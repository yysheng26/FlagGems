#include <gtest/gtest.h>

#include <cmath>
#include <string>
#include <vector>

#include "flag_gems/accuracy_utils.h"
#include "flag_gems/backend_utils.h"
#include "flag_gems/device_info.h"
#include "flag_gems/test_utils.h"
#include "flag_gems/utils.h"
#include "torch/torch.h"
#include "triton_jit/triton_jit_function.h"

namespace {

constexpr double kLog2e = 1.4426950408889634074;

bool is_hopper_or_newer() {
  return flag_gems::device::current_compute_capability_major() >= 9;
}

at::Tensor build_cu_seqlens(const std::vector<int32_t>& lens, const torch::Device& device) {
  std::vector<int32_t> cu;
  cu.reserve(lens.size() + 1);
  cu.push_back(0);
  for (int32_t x : lens) {
    cu.push_back(cu.back() + x);
  }
  return torch::tensor(cu, torch::TensorOptions().dtype(torch::kInt32).device(device));
}

at::Tensor ref_varlen_non_paged(const at::Tensor& q,
                                 const at::Tensor& k,
                                 const at::Tensor& v,
                                 const at::Tensor& cu_q,
                                 const at::Tensor& cu_k,
                                 double scale,
                                 bool causal) {
  const int64_t batch = cu_q.numel() - 1;
  const int64_t ratio = q.size(1) / k.size(1);
  at::Tensor out = torch::empty_like(q);
  for (int64_t b = 0; b < batch; ++b) {
    const int64_t qs = cu_q[b].item<int64_t>();
    const int64_t qe = cu_q[b + 1].item<int64_t>();
    const int64_t ks = cu_k[b].item<int64_t>();
    const int64_t ke = cu_k[b + 1].item<int64_t>();
    auto qi = q.index({at::indexing::Slice(qs, qe)}).to(torch::kFloat32);
    auto ki = k.index({at::indexing::Slice(ks, ke)}).to(torch::kFloat32);
    auto vi = v.index({at::indexing::Slice(ks, ke)}).to(torch::kFloat32);
    if (ratio > 1) {
      ki = at::repeat_interleave(ki, ratio, /*dim=*/1);
      vi = at::repeat_interleave(vi, ratio, /*dim=*/1);
    }
    auto attn = at::einsum("qhd,khd->hqk", {qi, ki}) * scale;
    if (causal) {
      const int64_t ql = qe - qs;
      const int64_t kl = ke - ks;
      auto mask = at::triu(torch::ones({ql, kl}, qi.options().dtype(torch::kBool)),
                           /*diagonal=*/kl - ql + 1);
      attn.masked_fill_(mask.unsqueeze(0), -std::numeric_limits<float>::infinity());
    }
    attn = at::softmax(attn, /*dim=*/-1).to(q.dtype());
    out.index_put_({at::indexing::Slice(qs, qe)},
                   at::einsum("hqk,khd->qhd", {attn, vi.to(q.dtype())}));
  }
  return out;
}

const std::string decode_kernel_poc_py_path() {
  return (flag_gems::utils::get_flag_gems_src_path() / "ops" / "flash_kernel_gluon_decode_jit_poc.py")
      .string();
}

const std::string fwd_kernel_poc_py_path() {
  return (flag_gems::utils::get_flag_gems_src_path() / "ops" / "flash_kernel_gluon_fwd_jit_poc.py")
      .string();
}

void launch_decode_kernel_cpp(const at::Tensor& q,
                              const at::Tensor& k,
                              const at::Tensor& v,
                              at::Tensor& out,
                              at::Tensor& lse,
                              const at::Tensor& cu_q,
                              const at::Tensor& cu_k,
                              const at::Tensor& batch_ids,
                              double softmax_scale) {
  const int64_t total_q = q.size(0);
  const int64_t d = q.size(2);
  const int64_t n_decode = batch_ids.size(0);
  const int64_t num_heads_k = k.size(1);
  const int64_t h_hk_ratio = q.size(1) / num_heads_k;
  const int64_t block_n = (d >= 256) ? 32 : 64;
  const int64_t d_pad = flag_gems::utils::next_power_of_2(d);

  const triton_jit::TritonJITFunction& f =
      triton_jit::TritonJITFunction::get_instance(decode_kernel_poc_py_path(),
                                                  "flash_varlen_decode_gluon_kernel_jit");

  c10::DeviceGuard guard(q.device());
  flag_gems::backend::StreamType stream = flag_gems::backend::getCurrentStream();
  flag_gems::backend::RawStreamType raw_stream = flag_gems::backend::getRawStream(stream);

  const unsigned grid_x = static_cast<unsigned>(n_decode);
  const unsigned grid_y = static_cast<unsigned>(num_heads_k);
  const unsigned num_warps = 4;
  const unsigned num_stages = 1;

  f(raw_stream,
    grid_x,
    grid_y,
    1,
    num_warps,
    num_stages,
    q,
    k,
    v,
    out,
    lse,
    cu_q,
    cu_k,
    batch_ids,
    q.stride(0),
    q.stride(1),
    k.stride(0),
    v.stride(0),
    out.stride(0),
    out.stride(1),
    k.stride(1),
    v.stride(1),
    total_q,
    static_cast<float>(softmax_scale),
    /*is_causal=*/true,
    /*PACK_GQA=*/true,
    h_hk_ratio,
    /*MAX_Q_LEN=*/4,
    block_n,
    d,
    d_pad);
}

void launch_fwd_kernel_cpp(const at::Tensor& q,
                           const at::Tensor& k,
                           const at::Tensor& v,
                           at::Tensor& out,
                           at::Tensor& lse,
                           const at::Tensor& cu_q,
                           const at::Tensor& cu_k,
                           int64_t max_seqlen_q,
                           double softmax_scale) {
  const int64_t total_q = q.size(0);
  const int64_t num_heads = q.size(1);
  const int64_t d = q.size(2);
  const int64_t num_heads_k = k.size(1);
  const int64_t h_hk_ratio = num_heads / num_heads_k;
  const int64_t batch = cu_q.numel() - 1;
  const double scale_log2 = softmax_scale * kLog2e;

  // POC test shape (d=128, max_seqlen_q=64, GQA): matches _resolve_varlen_launch_params.
  const int64_t block_m = 128;
  const int64_t block_n = 128;
  const int64_t grid_m = flag_gems::utils::cdiv(max_seqlen_q * h_hk_ratio, block_m);
  const int64_t grid_h = num_heads_k;

  const triton_jit::TritonJITFunction& f = triton_jit::TritonJITFunction::get_instance(
      fwd_kernel_poc_py_path(), "flash_varlen_fwd_gluon_kernel");

  c10::DeviceGuard guard(q.device());
  flag_gems::backend::StreamType stream = flag_gems::backend::getCurrentStream();
  flag_gems::backend::RawStreamType raw_stream = flag_gems::backend::getRawStream(stream);

  // Compile option must be power-of-2; actual launch warps come from kernel metadata (e.g. 12).
  const unsigned compile_num_warps = 8;
  const unsigned num_stages = 1;
  const unsigned constexpr_num_warps = 8;

  f(raw_stream,
    static_cast<unsigned>(grid_m),
    static_cast<unsigned>(batch),
    static_cast<unsigned>(grid_h),
    compile_num_warps,
    num_stages,
    q,
    k,
    v,
    out,
    lse,
    out,
    lse,
    static_cast<int64_t>(0),
    static_cast<int64_t>(0),
    cu_q,
    cu_k,
    cu_q,
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
    static_cast<float>(scale_log2),
    /*is_causal=*/true,
    block_m,
    block_n,
    constexpr_num_warps,
    num_stages,
    /*PACK_GQA=*/true,
    /*NUM_SPLITS=*/1,
    /*PRODUCER_WARPS=*/4,
    /*CONSUMER_WARPS=*/8);
}

}  // namespace

class GluonVarlenJitPocTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (!flag_gems::test::is_device_available()) {
      GTEST_SKIP() << "CUDA device not available";
    }
    if (!is_hopper_or_newer()) {
      GTEST_SKIP() << "Gluon POC requires Hopper sm90+";
    }
  }
};

TEST_F(GluonVarlenJitPocTest, DecodeKernelViaLibTritonJit) {
  torch::manual_seed(20260630);
  const torch::Device device = flag_gems::test::default_device();
  const auto fp16 = torch::TensorOptions().dtype(torch::kFloat16).device(device);

  const std::vector<int32_t> q_lens{1, 1};
  const std::vector<int32_t> k_lens{64, 128};
  const int64_t num_heads = 16;
  const int64_t num_heads_k = 8;
  const int64_t head_size = 128;
  const double scale = 1.0 / std::sqrt(static_cast<double>(head_size));

  const int64_t total_q = q_lens[0] + q_lens[1];
  const int64_t total_k = k_lens[0] + k_lens[1];
  at::Tensor q = torch::randn({total_q, num_heads, head_size}, fp16);
  at::Tensor k = torch::randn({total_k, num_heads_k, head_size}, fp16);
  at::Tensor v = torch::randn_like(k);
  at::Tensor cu_q = build_cu_seqlens(q_lens, device);
  at::Tensor cu_k = build_cu_seqlens(k_lens, device);
  at::Tensor batch_ids = torch::tensor({0, 1}, torch::TensorOptions().dtype(torch::kInt32).device(device));

  at::Tensor out = torch::empty_like(q);
  at::Tensor lse = torch::empty({num_heads, total_q}, fp16.dtype(torch::kFloat32));

  launch_decode_kernel_cpp(q, k, v, out, lse, cu_q, cu_k, batch_ids, scale);
  flag_gems::test::synchronize();

  at::Tensor ref = ref_varlen_non_paged(q, k, v, cu_q, cu_k, scale, /*causal=*/true);
  auto result = flag_gems::accuracy_utils::gems_assert_close(
      out, ref, torch::kFloat16, false, 1, 2e-2f);
  EXPECT_TRUE(result.ok) << result.message;
}

TEST_F(GluonVarlenJitPocTest, FwdKernelViaLibTritonJit) {
  torch::manual_seed(20260630);
  const torch::Device device = flag_gems::test::default_device();
  const auto fp16 = torch::TensorOptions().dtype(torch::kFloat16).device(device);

  const std::vector<int32_t> q_lens{64};
  const std::vector<int32_t> k_lens{64};
  const int64_t num_heads = 16;
  const int64_t num_heads_k = 8;
  const int64_t head_size = 128;
  const int64_t max_seqlen_q = 64;
  const double scale = 1.0 / std::sqrt(static_cast<double>(head_size));

  at::Tensor q = torch::randn({64, num_heads, head_size}, fp16);
  at::Tensor k = torch::randn({64, num_heads_k, head_size}, fp16);
  at::Tensor v = torch::randn_like(k);
  at::Tensor cu_q = build_cu_seqlens(q_lens, device);
  at::Tensor cu_k = build_cu_seqlens(k_lens, device);

  at::Tensor out = torch::empty_like(q);
  at::Tensor lse = torch::empty({num_heads, 64}, fp16.dtype(torch::kFloat32));

  launch_fwd_kernel_cpp(q, k, v, out, lse, cu_q, cu_k, max_seqlen_q, scale);
  flag_gems::test::synchronize();

  at::Tensor ref = ref_varlen_non_paged(q, k, v, cu_q, cu_k, scale, /*causal=*/true);
  auto result = flag_gems::accuracy_utils::gems_assert_close(
      out, ref, torch::kFloat16, false, 1, 2e-2f);
  EXPECT_TRUE(result.ok) << result.message;
}