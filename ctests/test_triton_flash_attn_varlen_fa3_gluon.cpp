#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "flag_gems/accuracy_utils.h"
#include "flag_gems/device_info.h"
#include "flag_gems/operators.h"
#include "flag_gems/test_utils.h"
#include "torch/torch.h"

namespace {

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

at::Tensor build_seqused_k(const std::vector<int32_t>& lens, const torch::Device& device) {
  return torch::tensor(lens, torch::TensorOptions().dtype(torch::kInt32).device(device));
}

at::Tensor build_block_table(const std::vector<int32_t>& kv_lens,
                             int64_t block_size,
                             int64_t num_blocks,
                             const torch::Device& device) {
  int64_t max_pages = 0;
  for (int32_t kl : kv_lens) {
    max_pages = std::max(max_pages, (static_cast<int64_t>(kl) + block_size - 1) / block_size);
  }
  return torch::randint(0,
                        num_blocks,
                        {static_cast<int64_t>(kv_lens.size()), max_pages},
                        torch::TensorOptions().dtype(torch::kInt32).device(device));
}

at::Tensor ref_varlen_paged(const at::Tensor& q,
                            const at::Tensor& k_cache,
                            const at::Tensor& v_cache,
                            const at::Tensor& cu_q,
                            const at::Tensor& seqused_k,
                            const at::Tensor& block_table,
                            double scale,
                            bool causal) {
  const int64_t batch = cu_q.numel() - 1;
  const int64_t block_size = k_cache.size(1);
  const int64_t ratio = q.size(1) / k_cache.size(2);
  at::Tensor out = torch::empty_like(q);
  for (int64_t b = 0; b < batch; ++b) {
    const int64_t qs = cu_q[b].item<int64_t>();
    const int64_t qe = cu_q[b + 1].item<int64_t>();
    const int64_t kl = seqused_k[b].item<int64_t>();
    const int64_t npages = (kl + block_size - 1) / block_size;
    auto pids = block_table[b].slice(0, 0, npages);
    auto kd = k_cache.index_select(0, pids)
                  .reshape({-1, k_cache.size(2), k_cache.size(3)})
                  .slice(0, 0, kl);
    auto vd = v_cache.index_select(0, pids)
                  .reshape({-1, v_cache.size(2), v_cache.size(3)})
                  .slice(0, 0, kl);
    auto qi = q.index({at::indexing::Slice(qs, qe)}).to(torch::kFloat32);
    auto ki = kd.to(torch::kFloat32);
    auto vi = vd.to(torch::kFloat32);
    if (ratio > 1) {
      ki = at::repeat_interleave(ki, ratio, /*dim=*/1);
      vi = at::repeat_interleave(vi, ratio, /*dim=*/1);
    }
    auto attn = at::einsum("qhd,khd->hqk", {qi, ki}) * scale;
    if (causal) {
      const int64_t ql = qe - qs;
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

std::tuple<at::Tensor, at::Tensor> call_fa3_varlen(at::Tensor q,
                                                   at::Tensor k,
                                                   at::Tensor v,
                                                   at::Tensor cu_q,
                                                   int64_t max_seqlen_q,
                                                   int64_t max_seqlen_k,
                                                   double scale,
                                                   bool causal,
                                                   const std::optional<at::Tensor>& cu_k = std::nullopt,
                                                   const std::optional<at::Tensor>& seqused_k = std::nullopt,
                                                   const std::optional<at::Tensor>& block_table = std::nullopt) {
  return flag_gems::flash_attn_varlen_func(q,
                                           k,
                                           v,
                                           max_seqlen_q,
                                           cu_q,
                                           max_seqlen_k,
                                           cu_k,
                                           seqused_k,
                                           /*q_v=*/std::nullopt,
                                           /*dropout_p=*/0.0,
                                           /*softmax_scale=*/scale,
                                           causal,
                                           /*window_size=*/std::nullopt,
                                           /*softcap=*/0.0,
                                           /*alibi_slopes=*/std::nullopt,
                                           /*deterministic=*/false,
                                           /*return_attn_probs=*/false,
                                           block_table,
                                           /*return_softmax_lse=*/false,
                                           /*out=*/std::nullopt,
                                           /*scheduler_metadata=*/std::nullopt,
                                           /*q_descale=*/std::nullopt,
                                           /*k_descale=*/std::nullopt,
                                           /*v_descale=*/std::nullopt,
                                           /*s_aux=*/std::nullopt,
                                           /*num_splits=*/0,
                                           /*cp_world_size=*/1,
                                           /*cp_rank=*/0,
                                           /*cp_tot_seqused_k=*/std::nullopt,
                                           /*fa_version=*/3);
}

}  // namespace

class FlashAttnVarlenFa3GluonWrapperTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (!flag_gems::test::is_device_available()) {
      GTEST_SKIP() << "CUDA device not available";
    }
    if (!is_hopper_or_newer()) {
      GTEST_SKIP() << "FA3 Gluon C++ wrapper requires Hopper sm90+";
    }
  }
};

TEST_F(FlashAttnVarlenFa3GluonWrapperTest, NonPagedViaFlashAttnVarlenFunc) {
  torch::manual_seed(20260630);
  const torch::Device device = flag_gems::test::default_device();
  const auto fp16 = torch::TensorOptions().dtype(torch::kFloat16).device(device);

  const std::vector<int32_t> q_lens{64};
  const std::vector<int32_t> k_lens{64};
  const int64_t num_heads = 16;
  const int64_t num_heads_k = 8;
  const int64_t head_size = 128;
  const double scale = 1.0 / std::sqrt(static_cast<double>(head_size));

  at::Tensor q = torch::randn({64, num_heads, head_size}, fp16);
  at::Tensor k = torch::randn({64, num_heads_k, head_size}, fp16);
  at::Tensor v = torch::randn_like(k);
  at::Tensor cu_q = build_cu_seqlens(q_lens, device);
  at::Tensor cu_k = build_cu_seqlens(k_lens, device);

  auto [out, lse] = call_fa3_varlen(q, k, v, cu_q, 64, 64, scale, /*causal=*/true, cu_k);
  flag_gems::test::synchronize();

  at::Tensor ref = ref_varlen_non_paged(q, k, v, cu_q, cu_k, scale, /*causal=*/true);
  auto result = flag_gems::accuracy_utils::gems_assert_close(
      out, ref, torch::kFloat16, false, 1, 2e-2f);
  EXPECT_TRUE(result.ok) << result.message;
  (void)lse;
}

TEST_F(FlashAttnVarlenFa3GluonWrapperTest, PagedViaFlashAttnVarlenFunc) {
  torch::manual_seed(20260630);
  const torch::Device device = flag_gems::test::default_device();
  const auto fp16 = torch::TensorOptions().dtype(torch::kFloat16).device(device);

  const std::vector<int32_t> q_lens{64, 32};
  const std::vector<int32_t> kv_lens{128, 96};
  const int64_t num_heads = 16;
  const int64_t num_heads_k = 8;
  const int64_t head_size = 128;
  const int64_t block_size = 64;
  const int64_t num_blocks = 512;
  const double scale = 1.0 / std::sqrt(static_cast<double>(head_size));

  at::Tensor q = torch::randn({96, num_heads, head_size}, fp16);
  at::Tensor k_cache = torch::randn({num_blocks, block_size, num_heads_k, head_size}, fp16);
  at::Tensor v_cache = torch::randn_like(k_cache);
  at::Tensor cu_q = build_cu_seqlens(q_lens, device);
  at::Tensor seqused_k = build_seqused_k(kv_lens, device);
  at::Tensor block_table = build_block_table(kv_lens, block_size, num_blocks, device);

  auto [out, lse] = call_fa3_varlen(q,
                                    k_cache,
                                    v_cache,
                                    cu_q,
                                    /*max_seqlen_q=*/64,
                                    /*max_seqlen_k=*/128,
                                    scale,
                                    /*causal=*/true,
                                    std::nullopt,
                                    seqused_k,
                                    block_table);
  flag_gems::test::synchronize();

  at::Tensor ref = ref_varlen_paged(q, k_cache, v_cache, cu_q, seqused_k, block_table, scale, /*causal=*/true);
  auto result = flag_gems::accuracy_utils::gems_assert_close(
      out, ref, torch::kFloat16, false, 1, 2e-2f);
  EXPECT_TRUE(result.ok) << result.message;
  (void)lse;
}

TEST_F(FlashAttnVarlenFa3GluonWrapperTest, DecodeViaPrefillPath) {
  torch::manual_seed(42);
  const torch::Device device = flag_gems::test::default_device();
  const auto fp16 = torch::TensorOptions().dtype(torch::kFloat16).device(device);

  const std::vector<int32_t> q_lens{1, 1, 1};
  const std::vector<int32_t> k_lens{64, 128, 32};
  const int64_t num_heads = 16;
  const int64_t num_heads_k = 8;
  const int64_t head_size = 128;
  const double scale = 1.0 / std::sqrt(static_cast<double>(head_size));

  at::Tensor q = torch::randn({3, num_heads, head_size}, fp16);
  at::Tensor k = torch::randn({224, num_heads_k, head_size}, fp16);
  at::Tensor v = torch::randn_like(k);
  at::Tensor cu_q = build_cu_seqlens(q_lens, device);
  at::Tensor cu_k = build_cu_seqlens(k_lens, device);

  auto [out, lse] = call_fa3_varlen(q, k, v, cu_q, 1, 128, scale, /*causal=*/true, cu_k);
  flag_gems::test::synchronize();

  at::Tensor ref = ref_varlen_non_paged(q, k, v, cu_q, cu_k, scale, /*causal=*/false);
  auto result = flag_gems::accuracy_utils::gems_assert_close(
      out, ref, torch::kFloat16, false, 1, 2e-2f);
  EXPECT_TRUE(result.ok) << result.message;
  (void)lse;
}