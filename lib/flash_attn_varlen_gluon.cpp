#include "flag_gems/flash_attn_varlen_gluon.h"

#include <cmath>
#include <tuple>
#include <vector>

#include "flag_gems/backend_utils.h"
#include "flag_gems/device_info.h"
#include "flag_gems/utils.h"
#include "torch/torch.h"
#include "triton_jit/triton_jit_function.h"

namespace flag_gems {
namespace {

constexpr int kSmemLimit = 232448;
constexpr int kGluonTargetConsumerWarps = 8;
constexpr double kLog2e = 1.4426950408889634074;

int64_t round_up_pow2(int64_t n) {
  if (n <= 1) {
    return 1;
  }
  int64_t p = 1;
  while (p < n) {
    p <<= 1;
  }
  return p;
}

int64_t round_down_pow2(int64_t n) {
  if (n <= 1) {
    return 1;
  }
  int64_t p = 1;
  while ((p << 1) <= n) {
    p <<= 1;
  }
  return p;
}

int64_t round_up_headdim(int64_t d) {
  if (d <= 64) return 64;
  if (d <= 96) return 96;
  if (d <= 128) return 128;
  if (d <= 192) return 192;
  return 256;
}

bool use_one_mma_wg(int64_t headdim, int64_t seqlen_q, bool pack_gqa, int64_t num_heads, int64_t num_heads_k) {
  if (headdim != 128) {
    return false;
  }
  const int64_t qhead_per_khead = pack_gqa ? (num_heads / num_heads_k) : 1;
  const int64_t effective_seqlen_q = seqlen_q * qhead_per_khead;
  return effective_seqlen_q <= 64;
}

std::pair<int64_t, int64_t> tile_size_fwd_sm90(int64_t headdim,
                                               bool is_causal,
                                               bool is_local,
                                               bool paged_kv_non_tma,
                                               bool use_one_mma_wg_flag) {
  if (headdim <= 64) {
    const bool use_block_n_128 = is_causal || is_local;
    return {192, use_block_n_128 ? 128 : 192};
  }
  if (headdim <= 96) {
    return {192, (is_local || paged_kv_non_tma) ? 128 : 144};
  }
  if (headdim <= 128) {
    if (use_one_mma_wg_flag) {
      return {64, (is_causal || is_local || paged_kv_non_tma) ? 128 : 176};
    }
    return {128, (is_causal || is_local || paged_kv_non_tma) ? 128 : 176};
  }
  if (headdim <= 192) {
    return {128, paged_kv_non_tma || is_local ? 96 : 128};
  }
  return {128, 32};
}

bool should_pack_gqa_varlen() {
  return true;
}

bool get_pack_gqa(int64_t num_heads,
                  int64_t num_heads_k,
                  int64_t /*max_seqlen_q*/,
                  int64_t /*block_m*/,
                  bool is_paged,
                  bool use_kv_tma) {
  if (num_heads == num_heads_k) {
    return false;
  }
  if (is_paged && !use_kv_tma) {
    return true;
  }
  return should_pack_gqa_varlen();
}

bool get_pagedkv_tma(int64_t page_size,
                     int64_t max_seqlen_q,
                     int64_t num_heads,
                     int64_t num_heads_k,
                     int64_t d_rounded,
                     bool is_causal) {
  auto [k_block_m, k_block_n] =
      tile_size_fwd_sm90(d_rounded, is_causal, /*is_local=*/false, /*paged_kv_non_tma=*/false,
                         /*use_one_mma_wg_flag=*/false);
  if (page_size % k_block_n != 0) {
    return false;
  }
  const int64_t seqlen_q_packgqa = max_seqlen_q * (num_heads / num_heads_k);
  return seqlen_q_packgqa > k_block_m;
}

int64_t gluon_varlen_smem_bytes(int64_t block_m,
                                int64_t block_n,
                                int64_t d,
                                int64_t elem_bytes,
                                int64_t num_stages) {
  const int64_t qo = 2 * block_m * d * elem_bytes;
  const int64_t pp = block_m * block_n * elem_bytes;
  const int64_t qk = block_m * block_n * 4;
  const int64_t kv = num_stages * 2 * block_n * d * elem_bytes;
  return qo + pp + qk + kv + 2048;
}

std::tuple<int64_t, int64_t, int64_t> fit_gluon_varlen_tiles(int64_t block_m,
                                                              int64_t block_n,
                                                              int64_t d,
                                                              int64_t elem_bytes,
                                                              int64_t num_stages) {
  const int64_t bm = round_up_pow2(block_m);
  int64_t bn_up = round_up_pow2(block_n);
  std::vector<int64_t> bn_opts;
  // paged block_size can be 16; bn must go down to 16 not 32 only.
  for (int64_t n = bn_up; n >= 16; n /= 2) {
    bn_opts.push_back(n);
  }
  std::vector<int64_t> bm_opts;
  for (int64_t candidate : {128, 64, 32}) {
    if (candidate <= bm) {
      bm_opts.push_back(candidate);
    }
  }
  if (bm_opts.empty()) {
    bm_opts.push_back(32);
  }
  std::vector<int64_t> ns_opts = {1};
  if (num_stages > 1) {
    ns_opts.push_back(num_stages);
  }
  for (int64_t bm_try : bm_opts) {
    for (int64_t ns : ns_opts) {
      for (int64_t bn_try : bn_opts) {
        if (gluon_varlen_smem_bytes(bm_try, bn_try, d, elem_bytes, ns) <= kSmemLimit) {
          return {bm_try, bn_try, ns};
        }
      }
    }
  }
  TORCH_CHECK(false, "No Gluon varlen tile fits smem budget");
}

std::pair<int64_t, int64_t> gluon_varlen_consumer_warps(int64_t block_m) {
  const int64_t bm = round_up_pow2(block_m);
  TORCH_CHECK(bm >= 64, "BLOCK_M must be >= 64 for WGMMA");
  const int64_t consumer_warps = (bm / 64) * 4;
  const int64_t launch_warps = round_up_pow2(consumer_warps);
  TORCH_CHECK(launch_warps == consumer_warps, "Unsupported BLOCK_M for Gluon consumer warps");
  return {launch_warps, consumer_warps};
}

struct GluonVarlenLaunchConfig {
  int64_t block_m = 0;
  int64_t block_n = 0;
  int64_t compile_num_warps = 0;
  int64_t consumer_warps = 0;
  int64_t producer_warps = 0;
  int64_t num_stages = 0;
  bool pack_gqa = false;
  int64_t n_heads_grid = 0;
};

GluonVarlenLaunchConfig resolve_gluon_varlen_launch_params(int64_t d,
                                                           int64_t max_seqlen_q,
                                                           int64_t max_seqlen_k,
                                                           int64_t batch,
                                                           int64_t num_heads,
                                                           int64_t num_heads_k,
                                                           bool is_causal,
                                                           bool is_paged,
                                                           bool use_kv_tma,
                                                           int64_t elem_bytes,
                                                           int64_t num_stages_in,
                                                           int64_t block_size = 0) {
  (void)max_seqlen_k;
  (void)batch;
  const int64_t h_hk_ratio = num_heads / num_heads_k;
  (void)h_hk_ratio;
  const bool paged_kv_non_tma = is_paged && !use_kv_tma;

  bool pack_gqa = get_pack_gqa(num_heads, num_heads_k, max_seqlen_q, 128, is_paged, use_kv_tma);
  const int64_t d_rounded = round_up_headdim(d);
  const bool uomw = use_one_mma_wg(d_rounded, max_seqlen_q, pack_gqa, num_heads, num_heads_k);
  auto [block_m_raw, block_n_raw] =
      tile_size_fwd_sm90(d_rounded, is_causal, /*is_local=*/false, paged_kv_non_tma, uomw);
  int64_t block_m = round_up_pow2(block_m_raw);
  int64_t block_n = round_up_pow2(block_n_raw);

  pack_gqa = get_pack_gqa(num_heads, num_heads_k, max_seqlen_q, block_m, is_paged, use_kv_tma);
  int64_t producer_warps = (paged_kv_non_tma || pack_gqa) ? 4 : 1;
  const int64_t d_nvmma = round_up_pow2(d);
  // Decode / short-q: keep uomw BLOCK_M=64; do not boost to 128.
  if (!uomw) {
    const int64_t target_bm = (d_nvmma > d) ? 64LL : (kGluonTargetConsumerWarps / 4) * 64LL;
    block_m = std::max(block_m, target_bm);
  }
  if (d_nvmma > d && block_m > 64) {
    block_m = 64;
  }
  if (is_paged && block_size > 0) {
    block_n = std::min(block_n, round_down_pow2(block_size));
  }
  int64_t num_stages = num_stages_in;
  std::tie(block_m, block_n, num_stages) =
      fit_gluon_varlen_tiles(block_m, block_n, d_nvmma, elem_bytes, num_stages);
  auto [launch_num_warps, consumer_warps] = gluon_varlen_consumer_warps(block_m);
  const int64_t n_heads_grid = pack_gqa ? num_heads_k : num_heads;

  GluonVarlenLaunchConfig cfg;
  cfg.block_m = block_m;
  cfg.block_n = block_n;
  cfg.compile_num_warps = launch_num_warps;
  cfg.consumer_warps = consumer_warps;
  cfg.producer_warps = producer_warps;
  cfg.num_stages = num_stages;
  cfg.pack_gqa = pack_gqa;
  cfg.n_heads_grid = n_heads_grid;
  return cfg;
}

void launch_gluon_nonpaged_varlen_wgmma(const at::Tensor& q,
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
                                        int64_t num_stages) {
  const int64_t total_q = q.size(0);
  const int64_t num_heads = q.size(1);
  const int64_t d = q.size(2);
  const int64_t num_heads_k = k.size(1);
  const int64_t batch = cu_seqlens_q.numel() - 1;
  if (batch == 0) {
    return;
  }
  const int64_t h_hk_ratio = num_heads / num_heads_k;
  const float scale_log2 = static_cast<float>(softmax_scale * kLog2e);
  const int64_t elem_bytes = q.element_size();

  const GluonVarlenLaunchConfig cfg = resolve_gluon_varlen_launch_params(
      d, max_seqlen_q, max_seqlen_k, batch, num_heads, num_heads_k, is_causal, /*is_paged=*/false,
      /*use_kv_tma=*/false, elem_bytes, num_stages);

  const int64_t grid_m =
      cfg.pack_gqa ? utils::cdiv(max_seqlen_q * h_hk_ratio, cfg.block_m) : utils::cdiv(max_seqlen_q, cfg.block_m);
  const int64_t grid_h = cfg.n_heads_grid;

  static const std::string kernel_py =
      (utils::get_flag_gems_src_path() / "ops" / "flash_kernel_gluon_fwd_jit_poc.py").string();
  const triton_jit::TritonJITFunction& f =
      triton_jit::TritonJITFunction::get_instance(kernel_py, "flash_varlen_fwd_gluon_kernel");

  c10::DeviceGuard guard(q.device());
  backend::StreamType stream = backend::getCurrentStream();
  backend::RawStreamType raw_stream = backend::getRawStream(stream);

  f(raw_stream,
    static_cast<unsigned>(grid_m),
    static_cast<unsigned>(batch),
    static_cast<unsigned>(grid_h),
    static_cast<unsigned>(cfg.compile_num_warps),
    static_cast<unsigned>(cfg.num_stages),
    q,
    k,
    v,
    out,
    lse,
    out,
    lse,
    static_cast<int64_t>(0),
    static_cast<int64_t>(0),
    cu_seqlens_q,
    cu_seqlens_k,
    cu_seqlens_q,
    q.stride(0),
    k.stride(0),
    v.stride(0),
    out.stride(0),
    q.stride(1),
    k.stride(1),
    v.stride(1),
    out.stride(1),
    total_q,
    num_heads,
    num_heads_k,
    h_hk_ratio,
    d,
    scale_log2,
    is_causal,
    cfg.block_m,
    cfg.block_n,
    cfg.compile_num_warps,
    cfg.num_stages,
    cfg.pack_gqa,
    static_cast<int64_t>(1),
    cfg.producer_warps,
    cfg.consumer_warps);
}

void launch_gluon_paged_varlen_wgmma(const at::Tensor& q,
                                     const at::Tensor& k,
                                     const at::Tensor& v,
                                     at::Tensor& out,
                                     at::Tensor& lse,
                                     const at::Tensor& cu_seqlens_q,
                                     const at::Tensor& seqused_k,
                                     const at::Tensor& page_table,
                                     int64_t max_seqlen_q,
                                     int64_t max_seqlen_k,
                                     float softmax_scale,
                                     bool is_causal,
                                     int64_t num_stages) {
  const int64_t total_q = q.size(0);
  const int64_t num_heads = q.size(1);
  const int64_t d = q.size(2);
  const int64_t block_size = k.size(1);
  const int64_t num_heads_k = k.size(2);
  const int64_t batch = cu_seqlens_q.numel() - 1;
  if (batch == 0) {
    return;
  }
  const int64_t h_hk_ratio = num_heads / num_heads_k;
  const float scale_log2 = static_cast<float>(softmax_scale * kLog2e);
  const int64_t elem_bytes = q.element_size();

  const int64_t d_rounded = round_up_headdim(d);
  bool use_kv_tma =
      get_pagedkv_tma(block_size, max_seqlen_q, num_heads, num_heads_k, d_rounded, is_causal);
  GluonVarlenLaunchConfig cfg = resolve_gluon_varlen_launch_params(
      d, max_seqlen_q, max_seqlen_k, batch, num_heads, num_heads_k, is_causal, /*is_paged=*/true, use_kv_tma,
      elem_bytes, num_stages, block_size);
  if (use_kv_tma && block_size % cfg.block_n != 0) {
    use_kv_tma = false;
    cfg = resolve_gluon_varlen_launch_params(d, max_seqlen_q, max_seqlen_k, batch, num_heads, num_heads_k, is_causal,
                                             /*is_paged=*/true, use_kv_tma, elem_bytes, num_stages, block_size);
  }

  const int64_t m_dim = max_seqlen_q * (cfg.pack_gqa ? h_hk_ratio : 1);
  const int64_t grid_m = utils::cdiv(m_dim, cfg.block_m);
  const int64_t grid_h = cfg.n_heads_grid;

  static const std::string kernel_py =
      (utils::get_flag_gems_src_path() / "ops" / "flash_kernel_gluon_paged_jit_poc.py").string();
  const triton_jit::TritonJITFunction& f =
      triton_jit::TritonJITFunction::get_instance(kernel_py, "flash_paged_fwd_gluon_kernel");

  c10::DeviceGuard guard(q.device());
  backend::StreamType stream = backend::getCurrentStream();
  backend::RawStreamType raw_stream = backend::getRawStream(stream);

  f(raw_stream,
    static_cast<unsigned>(grid_m),
    static_cast<unsigned>(batch),
    static_cast<unsigned>(grid_h),
    static_cast<unsigned>(cfg.compile_num_warps),
    static_cast<unsigned>(cfg.num_stages),
    q,
    k,
    v,
    out,
    lse,
    out,
    lse,
    static_cast<int64_t>(0),
    static_cast<int64_t>(0),
    cu_seqlens_q,
    seqused_k,
    page_table,
    cu_seqlens_q,
    q.stride(0),
    q.stride(1),
    out.stride(0),
    out.stride(1),
    k.stride(0),
    k.stride(1),
    k.stride(2),
    page_table.stride(0),
    total_q,
    num_heads,
    num_heads_k,
    h_hk_ratio,
    d,
    block_size,
    scale_log2,
    is_causal,
    cfg.block_m,
    cfg.block_n,
    cfg.compile_num_warps,
    cfg.num_stages,
    static_cast<int64_t>(use_kv_tma ? 1 : 0),
    cfg.pack_gqa,
    static_cast<int64_t>(1),
    cfg.producer_warps,
    cfg.consumer_warps);
}

}  // namespace

bool flash_attn_varlen_fa3_gluon_capable(bool is_paged,
                                         bool is_softcap,
                                         bool is_local,
                                         bool has_descale) {
  if (device::current_compute_capability_major() < 9) {
    return false;
  }
  if (is_softcap || is_local || has_descale) {
    return false;
  }
  (void)is_paged;
  return true;
}

std::tuple<at::Tensor, at::Tensor> flash_attn_varlen_fa3_gluon_fwd(
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
    const std::optional<at::Tensor>& out_opt) {
  TORCH_CHECK(q.device() == k.device() && k.device() == v.device(), "q, k, v must be on the same device");
  TORCH_CHECK(q.scalar_type() == at::kHalf || q.scalar_type() == at::kBFloat16, "only fp16/bf16 supported");
  TORCH_CHECK(q.scalar_type() == k.scalar_type() && q.scalar_type() == v.scalar_type(), "dtype mismatch");
  TORCH_CHECK(q.stride(-1) == 1 && k.stride(-1) == 1 && v.stride(-1) == 1, "last dim must be contiguous");
  TORCH_CHECK(cu_seqlens_q.scalar_type() == at::kInt && cu_seqlens_q.is_contiguous(), "cu_seqlens_q must be int32");
  TORCH_CHECK(device::current_compute_capability_major() >= 9, "Gluon FA3 requires Hopper sm90+");

  const int64_t total_q = q.size(0);
  const int64_t num_heads = q.size(1);
  const int64_t head_size = q.size(2);
  const int64_t batch = cu_seqlens_q.numel() - 1;
  TORCH_CHECK(head_size <= 256 && head_size % 8 == 0, "invalid head_size");
  TORCH_CHECK(cu_seqlens_q.numel() == batch + 1, "cu_seqlens_q size mismatch");

  const bool is_paged = page_table.has_value() && page_table->defined() && page_table->numel() > 0;

  bool final_is_causal = is_causal;
  if (max_seqlen_q == 1) {
    final_is_causal = false;
  }

  at::Tensor out;
  if (out_opt.has_value() && out_opt->defined()) {
    out = *out_opt;
    TORCH_CHECK(out.scalar_type() == q.scalar_type(), "out dtype mismatch");
    TORCH_CHECK(out.sizes() == q.sizes(), "out shape mismatch");
  } else {
    out = at::empty_like(q);
  }
  at::Tensor lse = at::empty({num_heads, total_q}, q.options().dtype(at::kFloat));

  if (is_paged) {
    TORCH_CHECK(seqused_k.has_value(), "paged Gluon path requires seqused_k");
    TORCH_CHECK(!cu_seqlens_k.has_value(), "paged path must not pass cu_seqlens_k");
    TORCH_CHECK(k.dim() == 4 && v.dim() == 4, "paged k/v must be 4D");
    TORCH_CHECK(k.sizes() == v.sizes(), "k and v shape mismatch");
    const int64_t num_heads_k = k.size(2);
    TORCH_CHECK(num_heads % num_heads_k == 0, "GQA ratio invalid");
    TORCH_CHECK(seqused_k->scalar_type() == at::kInt && seqused_k->is_contiguous(), "seqused_k must be int32");
    TORCH_CHECK(seqused_k->numel() == batch, "seqused_k size mismatch");
    TORCH_CHECK(page_table->scalar_type() == at::kInt && page_table->is_contiguous(), "page_table must be int32");
    launch_gluon_paged_varlen_wgmma(q,
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
                                  /*num_stages=*/2);
  } else {
    TORCH_CHECK(cu_seqlens_k.has_value(), "non-paged Gluon path requires cu_seqlens_k");
    TORCH_CHECK(!seqused_k.has_value(), "non-paged path must not pass seqused_k");
    TORCH_CHECK(k.dim() == 3 && v.dim() == 3, "non-paged k/v must be 3D");
    TORCH_CHECK(k.sizes() == v.sizes(), "k and v shape mismatch");
    const int64_t num_heads_k = k.size(1);
    TORCH_CHECK(num_heads % num_heads_k == 0, "GQA ratio invalid");
    TORCH_CHECK(cu_seqlens_k->scalar_type() == at::kInt && cu_seqlens_k->is_contiguous(),
                "cu_seqlens_k must be int32");
    TORCH_CHECK(cu_seqlens_k->numel() == batch + 1, "cu_seqlens_k size mismatch");
    launch_gluon_nonpaged_varlen_wgmma(q,
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
                                       /*num_stages=*/2);
  }
  return {std::move(out), std::move(lse)};
}

}  // namespace flag_gems