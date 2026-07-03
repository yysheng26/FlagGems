#include "flag_gems/flash_attn_varlen_fa3_triton.h"

#include <cmath>
#include <tuple>

#include "flag_gems/backend_utils.h"
#include "flag_gems/device_info.h"
#include "flag_gems/utils.h"
#include "torch/torch.h"
#include "triton_jit/triton_jit_function.h"

namespace flag_gems {
namespace {

constexpr double kLog2e = 1.4426950408889634074;

// Tile selection: mirrors flash_api.py mha_varlan_fwd_fa3 cfg selection.
// Returns (block_m, block_n, num_warps, num_stages).
std::tuple<int64_t, int64_t, int64_t, int64_t> pick_fa3_tile_config(
    int64_t total_q,
    int64_t num_heads,
    int64_t batch_size,
    int64_t head_size,
    int64_t max_seqlen_q) {
  (void)max_seqlen_q;
  const double total_rows = static_cast<double>(total_q) * static_cast<double>(num_heads);
  int num_sms = device::current_sm_count();
  if (num_sms <= 0) num_sms = 1;
  const double avg_rows_per_sm = total_rows / static_cast<double>(num_sms);
  const double avg_rows_per_batch = static_cast<double>(total_q) / static_cast<double>(batch_size);
  const double avg_rows_per_cta = std::min(avg_rows_per_batch, avg_rows_per_sm);

  int64_t block_m, block_n;
  if (avg_rows_per_cta > 64.0) {
    block_m = 128;
    block_n = 32;
  } else if (avg_rows_per_cta > 32.0) {
    block_m = 64;
    block_n = 64;
  } else if (avg_rows_per_cta > 16.0) {
    block_m = 32;
    block_n = 64;
  } else {
    block_m = 16;
    block_n = 64;
  }
  // FA3 on Hopper favours larger tiles
  if (block_m <= 64 && head_size <= 128 && avg_rows_per_cta > 8.0) {
    block_m = 64;
  }
  if (head_size >= 256) {
    block_n = 32;
  }

  // Match Python FA3 heuristic: always 4 warps / 3 stages for Hopper warp_specialize
  const int64_t num_warps = 4;
  const int64_t num_stages = 3;
  return {block_m, block_n, num_warps, num_stages};
}

int64_t round_multiple(int64_t x, int64_t m) {
  return (x + m - 1) / m * m;
}

// Launch flash_varlen_fwd_fa3_kernel for non-paged inputs.
void launch_fa3_triton_nonpaged(const at::Tensor& q,
                                const at::Tensor& k,
                                const at::Tensor& v,
                                at::Tensor& out,
                                at::Tensor& lse,
                                const at::Tensor& cu_seqlens_q,
                                const at::Tensor& cu_seqlens_k,
                                int64_t max_seqlen_q,
                                int64_t max_seqlen_k,
                                float softmax_scale,
                                bool is_causal,
                                int64_t window_size_left,
                                int64_t window_size_right,
                                float softcap_val,
                                bool has_descale,
                                double qk_descale_val,
                                double v_descale_val) {
  const int64_t total_q = q.size(0);
  const int64_t num_heads = q.size(1);
  const int64_t head_size = q.size(2);
  const int64_t num_heads_k = k.size(1);
  const int64_t batch = cu_seqlens_q.numel() - 1;
  if (batch == 0) return;

  const int64_t h_hk_ratio = num_heads / num_heads_k;
  const float scale_log2 = softmax_scale * static_cast<float>(kLog2e);

  // Tile config
  auto [block_m, block_n, num_warps, num_stages] =
      pick_fa3_tile_config(total_q, num_heads, batch, head_size, max_seqlen_q);
  const int64_t block_k = utils::next_power_of_2(head_size);
  const int64_t head_size_rounded = head_size < 192 ? round_multiple(head_size, 32) : 256;
  const int64_t seqlen_q_rounded = round_multiple(max_seqlen_q, 128);
  const int64_t seqlen_k_rounded = round_multiple(max_seqlen_k, 32);

  // softcap
  const bool is_softcap = softcap_val > 0.0;
  const float adjusted_softcap = is_softcap ? (softmax_scale / softcap_val) : 0.0f;
  const float adjusted_scale_softmax = is_softcap ? softcap_val : softmax_scale;
  const float adjusted_scale_softmax_log2e = is_softcap
      ? (softcap_val * static_cast<float>(kLog2e))
      : scale_log2;

  // paged fields (unused for non-paged)
  const int64_t block_size = 1;
  const int64_t k_batch_size = 0;
  const int64_t page_table_batch_stride = 0;
  const int64_t k_page_stride = 0;
  at::Tensor page_table = at::empty({0, 0}, q.options().dtype(at::kInt));
  at::Tensor seqused_k_dummy = at::empty({0}, q.options().dtype(at::kInt));

  // dropout (disabled in FA3)
  at::Tensor philox_args = at::empty({2}, q.options().dtype(at::kLong));
  at::Tensor p_dummy = at::empty({}, q.options());
  // alibi (disabled in FA3)
  at::Tensor alibi_slopes = at::empty({0}, q.options().dtype(at::kFloat));

  const bool is_hopper = device::current_compute_capability_major() >= 9;

  // Strides
  const int64_t q_row_stride = q.stride(0);
  const int64_t k_row_stride = k.stride(0);
  const int64_t v_row_stride = v.stride(0);
  const int64_t o_row_stride = out.stride(0);
  const int64_t q_head_stride = q.stride(1);
  const int64_t k_head_stride = k.stride(1);
  const int64_t v_head_stride = v.stride(1);
  const int64_t o_head_stride = out.stride(1);
  const int64_t q_batch_stride = 0;
  const int64_t k_batch_stride = 0;
  const int64_t v_batch_stride = 0;
  const int64_t o_batch_stride = 0;

  const bool is_paged = false;
  const bool is_local = window_size_left >= 0;
  const bool seqlenq_ngroups_swapped = false;

  static const std::string kernel_py =
      (utils::get_flag_gems_src_path() / "ops" / "flash_kernel.py").string();
  const triton_jit::TritonJITFunction& f =
      triton_jit::TritonJITFunction::get_instance(kernel_py, "flash_varlen_fwd_fa3_kernel");

  c10::DeviceGuard guard(q.device());
  backend::StreamType stream = backend::getCurrentStream();
  backend::RawStreamType raw_stream = backend::getRawStream(stream);

  const unsigned grid_x = static_cast<unsigned>(utils::cdiv(max_seqlen_q, block_m));
  const unsigned grid_y = static_cast<unsigned>(batch);
  const unsigned grid_z = static_cast<unsigned>(num_heads);

  f(raw_stream,
    grid_x,
    grid_y,
    grid_z,
    static_cast<unsigned>(num_warps),
    static_cast<unsigned>(num_stages),
    // fwd_params.__slots__ order (59 args)
    q,                                           // q_ptr
    k,                                           // k_ptr
    v,                                           // v_ptr
    out,                                         // o_ptr
    p_dummy,                                     // p_ptr (no dropout)
    lse,                                         // softmax_lse_ptr
    q_row_stride,
    k_row_stride,
    v_row_stride,
    q_head_stride,
    k_head_stride,
    v_head_stride,
    o_row_stride,
    o_head_stride,
    q_batch_stride,
    k_batch_stride,
    v_batch_stride,
    o_batch_stride,
    true,                                        // is_cu_seqlens_q
    cu_seqlens_q,                                // cu_seqlens_q_ptr
    true,                                        // is_cu_seqlens_k
    cu_seqlens_k,                                // cu_seqlens_k_ptr
    false,                                       // is_seqused_k
    seqused_k_dummy,                             // seqused_k_ptr (dummy)
    // sizes
    batch,                                       // b
    k_batch_size,                                // bk
    num_heads,                                   // h
    num_heads_k,                                 // hk
    h_hk_ratio,                                  // h_hk_ratio
    max_seqlen_q,                                // seqlen_q
    max_seqlen_k,                                // seqlen_k
    seqlen_q_rounded,                            // seqlen_q_rounded
    seqlen_k_rounded,                            // seqlen_k_rounded
    head_size,                                   // d
    head_size_rounded,                           // d_rounded
    // scaling
    is_softcap,                                  // is_softcap
    adjusted_softcap,                            // softcap
    adjusted_scale_softmax,                      // scale_softmax
    adjusted_scale_softmax_log2e,                // scale_softmax_log2
    // dropout (disabled)
    false,                                       // is_dropout
    0.0,                                         // p_dropout
    1.0,                                         // rp_dropout
    255,                                         // p_dropout_in_uint8_t
    philox_args,                                 // philox_args
    false,                                       // return_softmax
    // causal / swa
    is_causal,                                   // is_causal
    is_local,                                    // is_local
    window_size_left,                            // window_size_left
    window_size_right,                           // window_size_right
    seqlenq_ngroups_swapped,                     // seqlenq_ngroups_swapped
    is_paged,                                    // is_paged
    // alibi (disabled in FA3)
    false,                                       // is_alibi
    alibi_slopes,                                // alibi_slopes_ptr
    static_cast<int64_t>(0),                    // alibi_slopes_batch_stride
    // block table (unused for non-paged)
    total_q,                                     // total_q
    page_table,                                  // page_table_ptr
    page_table_batch_stride,                     // page_table_batch_stride
    block_size,                                  // block_size
    k_page_stride,                               // k_page_stride
    // cfg_params (9 extra)
    has_descale,                                 // HAS_DESCALE
    qk_descale_val,                              // qk_descale
    v_descale_val,                               // v_descale
    is_hopper,                                   // IS_HOPPER
    block_m,                                     // BLOCK_M
    block_n,                                     // BLOCK_N
    block_k,                                     // BLOCK_K
    num_warps,                                   // num_warps
    num_stages                                   // num_stages
  );
}

// Launch flash_varlen_fwd_fa3_kernel for paged inputs.
void launch_fa3_triton_paged(const at::Tensor& q,
                             const at::Tensor& k,
                             const at::Tensor& v,
                             at::Tensor& out,
                             at::Tensor& lse,
                             const at::Tensor& cu_seqlens_q,
                             const at::Tensor& seqused_k,
                             const at::Tensor& page_table_in,
                             int64_t max_seqlen_q,
                             int64_t max_seqlen_k,
                             float softmax_scale,
                             bool is_causal,
                             int64_t window_size_left,
                             int64_t window_size_right,
                             float softcap_val,
                             bool has_descale,
                             double qk_descale_val,
                             double v_descale_val) {
  const int64_t total_q = q.size(0);
  const int64_t num_heads = q.size(1);
  const int64_t head_size = q.size(2);
  const int64_t num_heads_k = k.size(2);           // paged: [N, block, nk, d]
  const int64_t batch = cu_seqlens_q.numel() - 1;
  if (batch == 0) return;

  const int64_t block_size = k.size(1);
  const int64_t num_pages = k.size(0);
  const int64_t k_batch_size = num_pages;
  const int64_t h_hk_ratio = num_heads / num_heads_k;
  const float scale_log2 = softmax_scale * static_cast<float>(kLog2e);
  const int64_t page_table_batch_stride = page_table_in.stride(0);
  const int64_t k_page_stride = k.stride(0);

  // Tile config
  auto [block_m, block_n, num_warps, num_stages] =
      pick_fa3_tile_config(total_q, num_heads, batch, head_size, max_seqlen_q);
  const int64_t block_k = utils::next_power_of_2(head_size);
  const int64_t head_size_rounded = head_size < 192 ? round_multiple(head_size, 32) : 256;
  const int64_t seqlen_q_rounded = round_multiple(max_seqlen_q, 128);
  const int64_t seqlen_k_rounded = round_multiple(max_seqlen_k, 32);

  // softcap
  const bool is_softcap = softcap_val > 0.0;
  const float adjusted_softcap = is_softcap ? (softmax_scale / softcap_val) : 0.0f;
  const float adjusted_scale_softmax = is_softcap ? softcap_val : softmax_scale;
  const float adjusted_scale_softmax_log2e = is_softcap
      ? (softcap_val * static_cast<float>(kLog2e))
      : scale_log2;

  // dropout / alibi (disabled)
  at::Tensor philox_args = at::empty({2}, q.options().dtype(at::kLong));
  at::Tensor p_dummy = at::empty({}, q.options());
  at::Tensor alibi_slopes = at::empty({0}, q.options().dtype(at::kFloat));

  const bool is_hopper = device::current_compute_capability_major() >= 9;

  // Strides: mirror flash_api.py fwd_params pattern.
  // q: 3D [total_q, nq, d] → stride(-3)=stride(0), stride(-2)=stride(1)
  // k: 4D [num_pages, block_size, nk, d] → stride(-3)=stride(1), stride(-2)=stride(2)
  const int64_t q_row_stride = q.stride(0);
  const int64_t k_row_stride = k.stride(1);       // row stride within a page
  const int64_t v_row_stride = v.stride(1);
  const int64_t o_row_stride = out.stride(0);
  const int64_t q_head_stride = q.stride(1);
  const int64_t k_head_stride = k.stride(2);      // head stride within a page
  const int64_t v_head_stride = v.stride(2);
  const int64_t o_head_stride = out.stride(1);
  const int64_t q_batch_stride = 0;
  const int64_t k_batch_stride = 0;
  const int64_t v_batch_stride = 0;
  const int64_t o_batch_stride = 0;

  const bool is_paged = true;
  const bool is_local = window_size_left >= 0;
  const bool seqlenq_ngroups_swapped = false;

  // cu_seqlens_k dummy for paged path
  at::Tensor cu_seqlens_k_dummy = at::empty({batch + 1}, q.options().dtype(at::kInt));

  static const std::string kernel_py =
      (utils::get_flag_gems_src_path() / "ops" / "flash_kernel.py").string();
  const triton_jit::TritonJITFunction& f =
      triton_jit::TritonJITFunction::get_instance(kernel_py, "flash_varlen_fwd_fa3_kernel");

  c10::DeviceGuard guard(q.device());
  backend::StreamType stream = backend::getCurrentStream();
  backend::RawStreamType raw_stream = backend::getRawStream(stream);

  const unsigned grid_x = static_cast<unsigned>(utils::cdiv(max_seqlen_q, block_m));
  const unsigned grid_y = static_cast<unsigned>(batch);
  const unsigned grid_z = static_cast<unsigned>(num_heads);

  f(raw_stream,
    grid_x,
    grid_y,
    grid_z,
    static_cast<unsigned>(num_warps),
    static_cast<unsigned>(num_stages),
    // fwd_params.__slots__ (59 args)
    q,
    k,
    v,
    out,
    p_dummy,                                      // p_ptr
    lse,                                          // softmax_lse_ptr
    q_row_stride,
    k_row_stride,
    v_row_stride,
    q_head_stride,
    k_head_stride,
    v_head_stride,
    o_row_stride,
    o_head_stride,
    q_batch_stride,
    k_batch_stride,
    v_batch_stride,
    o_batch_stride,
    true,                                         // is_cu_seqlens_q
    cu_seqlens_q,
    false,                                        // is_cu_seqlens_k (paged uses seqused_k)
    cu_seqlens_k_dummy,                           // cu_seqlens_k_ptr (dummy)
    true,                                         // is_seqused_k
    seqused_k,
    batch,
    k_batch_size,
    num_heads,
    num_heads_k,
    h_hk_ratio,
    max_seqlen_q,
    max_seqlen_k,
    seqlen_q_rounded,
    seqlen_k_rounded,
    head_size,
    head_size_rounded,
    is_softcap,
    adjusted_softcap,
    adjusted_scale_softmax,
    adjusted_scale_softmax_log2e,
    false,                                        // is_dropout
    0.0,                                          // p_dropout
    1.0,                                          // rp_dropout
    255,                                          // p_dropout_in_uint8_t
    philox_args,
    false,                                        // return_softmax
    is_causal,
    is_local,
    window_size_left,
    window_size_right,
    seqlenq_ngroups_swapped,
    is_paged,
    false,                                        // is_alibi
    alibi_slopes,
    static_cast<int64_t>(0),
    total_q,
    page_table_in,
    page_table_batch_stride,
    block_size,
    k_page_stride,
    // cfg_params
    has_descale,
    qk_descale_val,
    v_descale_val,
    is_hopper,
    block_m,
    block_n,
    block_k,
    num_warps,
    num_stages
  );
}

}  // namespace

std::tuple<at::Tensor, at::Tensor> flash_attn_varlen_fa3_triton_fwd(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& cu_seqlens_q,
    const std::optional<at::Tensor>& cu_seqlens_k,
    const std::optional<at::Tensor>& seqused_k,
    const std::optional<at::Tensor>& page_table,
    int64_t max_seqlen_q,
    int64_t max_seqlen_k,
    double softmax_scale,
    bool is_causal,
    int64_t window_size_left,
    int64_t window_size_right,
    double softcap,
    const std::optional<at::Tensor>& q_descale,
    const std::optional<at::Tensor>& k_descale,
    const std::optional<at::Tensor>& v_descale,
    const std::optional<at::Tensor>& out_opt) {
  // Basic validation
  TORCH_CHECK(q.device() == k.device() && k.device() == v.device(), "q, k, v must be on same device");
  TORCH_CHECK(q.scalar_type() == at::kHalf || q.scalar_type() == at::kBFloat16, "only fp16/bf16");
  TORCH_CHECK(q.scalar_type() == k.scalar_type() && q.scalar_type() == v.scalar_type());
  TORCH_CHECK(q.stride(-1) == 1 && k.stride(-1) == 1 && v.stride(-1) == 1);
  TORCH_CHECK(cu_seqlens_q.scalar_type() == at::kInt && cu_seqlens_q.is_contiguous());
  TORCH_CHECK(device::current_compute_capability_major() >= 9, "FA3 Triton requires Hopper sm90+");

  const int64_t total_q = q.size(0);
  const int64_t num_heads = q.size(1);
  const int64_t head_size = q.size(2);
  const int64_t batch = cu_seqlens_q.numel() - 1;
  TORCH_CHECK(head_size <= 256 && head_size % 8 == 0, "bad head_size");

  const bool is_paged = page_table.has_value() && page_table->defined() && page_table->numel() > 0;

  // causal adjustment
  bool final_is_causal = is_causal;
  if (max_seqlen_q == 1 && final_is_causal) {
    final_is_causal = false;
  }
  if (final_is_causal) {
    window_size_right = 0;
  }
  if (window_size_left >= max_seqlen_k) window_size_left = -1;
  if (window_size_right >= max_seqlen_k) window_size_right = -1;

  // out
  at::Tensor out;
  if (out_opt.has_value() && out_opt->defined()) {
    out = *out_opt;
    TORCH_CHECK(out.scalar_type() == q.scalar_type());
    TORCH_CHECK(out.sizes() == q.sizes());
  } else {
    out = at::empty_like(q);
  }

  // lse
  at::Tensor lse = at::empty({num_heads, total_q}, q.options().dtype(at::kFloat));

  // descale
  bool has_descale = q_descale.has_value() || k_descale.has_value() || v_descale.has_value();
  double qk_descale_val = 1.0;
  double v_descale_val = 1.0;
  if (q_descale.has_value()) qk_descale_val *= q_descale->item<double>();
  if (k_descale.has_value()) qk_descale_val *= k_descale->item<double>();
  if (v_descale.has_value()) v_descale_val = v_descale->item<double>();

  if (is_paged) {
    TORCH_CHECK(seqused_k.has_value(), "paged requires seqused_k");
    TORCH_CHECK(k.dim() == 4 && v.dim() == 4, "paged k/v must be 4D");
    const int64_t num_heads_k = k.size(2);
    TORCH_CHECK(num_heads % num_heads_k == 0);
    TORCH_CHECK(seqused_k->scalar_type() == at::kInt && seqused_k->is_contiguous());
    TORCH_CHECK(seqused_k->numel() == batch);
    launch_fa3_triton_paged(q,
                            k,
                            v,
                            out,
                            lse,
                            cu_seqlens_q,
                            seqused_k.value(),
                            page_table.value(),
                            max_seqlen_q,
                            max_seqlen_k,
                            static_cast<float>(softmax_scale),
                            final_is_causal,
                            window_size_left,
                            window_size_right,
                            static_cast<float>(softcap),
                            has_descale,
                            qk_descale_val,
                            v_descale_val);
  } else {
    TORCH_CHECK(cu_seqlens_k.has_value(), "non-paged requires cu_seqlens_k");
    TORCH_CHECK(k.dim() == 3 && v.dim() == 3, "non-paged k/v must be 3D");
    const int64_t num_heads_k = k.size(1);
    TORCH_CHECK(num_heads % num_heads_k == 0);
    TORCH_CHECK(cu_seqlens_k->scalar_type() == at::kInt && cu_seqlens_k->is_contiguous());
    TORCH_CHECK(cu_seqlens_k->numel() == batch + 1);
    launch_fa3_triton_nonpaged(q,
                               k,
                               v,
                               out,
                               lse,
                               cu_seqlens_q,
                               cu_seqlens_k.value(),
                               max_seqlen_q,
                               max_seqlen_k,
                               static_cast<float>(softmax_scale),
                               final_is_causal,
                               window_size_left,
                               window_size_right,
                               static_cast<float>(softcap),
                               has_descale,
                               qk_descale_val,
                               v_descale_val);
  }

  return {std::move(out), std::move(lse)};
}

}  // namespace flag_gems
