#include <gtest/gtest.h>

#include <cmath>
#include <numeric>
#include <vector>

#include "flag_gems/accuracy_utils.h"
#include "flag_gems/backend_utils.h"
#include "flag_gems/device_info.h"
#include "flag_gems/flash_attn_varlen_fa3_triton.h"
#include "flag_gems/test_utils.h"
#include "flag_gems/utils.h"
#include "torch/torch.h"

namespace {

bool is_hopper() { return flag_gems::device::current_compute_capability_major() >= 9; }

at::Tensor cu_from_lens(const std::vector<int32_t>& lens, const torch::Device& d) {
  std::vector<int32_t> c{0};
  for (auto x : lens) c.push_back(c.back() + x);
  return torch::tensor(c, torch::TensorOptions().dtype(torch::kInt32).device(d));
}

// ── non-paged reference ────────────────────────────────────────────────
at::Tensor ref_nonpaged(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                        const at::Tensor& cu_q, const at::Tensor& cu_k,
                        double scale, bool causal) {
  int64_t B = cu_q.numel() - 1, R = q.size(1) / k.size(1);
  at::Tensor out = torch::empty_like(q);
  for (int64_t b = 0; b < B; ++b) {
    int64_t qs = cu_q[b].item<int64_t>(), qe = cu_q[b + 1].item<int64_t>();
    int64_t ks = cu_k[b].item<int64_t>(), ke = cu_k[b + 1].item<int64_t>();
    auto qi = q.slice(0, qs, qe).to(torch::kFloat32);
    auto ki = k.slice(0, ks, ke).to(torch::kFloat32);
    auto vi = v.slice(0, ks, ke).to(torch::kFloat32);
    if (R > 1) { ki = at::repeat_interleave(ki, R, 1); vi = at::repeat_interleave(vi, R, 1); }
    auto a = at::einsum("qhd,khd->hqk", {qi, ki}) * scale;
    if (causal) {
      auto m = at::triu(torch::ones({qe - qs, ke - ks}, qi.options().dtype(torch::kBool)), ke - ks - qe + qs + 1);
      a.masked_fill_(m.unsqueeze(0), -std::numeric_limits<float>::infinity());
    }
    a = at::softmax(a, -1).to(q.dtype());
    out.slice(0, qs, qe) = at::einsum("hqk,khd->qhd", {a, vi.to(q.dtype())});
  }
  return out;
}

// ── paged reference ────────────────────────────────────────────────────
at::Tensor ref_paged(const at::Tensor& q, const at::Tensor& kc, const at::Tensor& vc,
                     const at::Tensor& cu_q, const at::Tensor& seqused_k,
                     const at::Tensor& bt, double scale, bool causal) {
  int64_t B = cu_q.numel() - 1, bs = kc.size(1), R = q.size(1) / kc.size(2);
  at::Tensor out = torch::empty_like(q);
  for (int64_t b = 0; b < B; ++b) {
    int64_t qs = cu_q[b].item<int64_t>(), qe = cu_q[b + 1].item<int64_t>();
    int64_t kl = seqused_k[b].item<int64_t>();
    int64_t np = (kl + bs - 1) / bs;
    auto pids = bt[b].slice(0, 0, np);
    auto kd = kc.index_select(0, pids).reshape({-1, kc.size(2), kc.size(3)}).slice(0, 0, kl);
    auto vd = vc.index_select(0, pids).reshape({-1, vc.size(2), vc.size(3)}).slice(0, 0, kl);
    auto qi = q.slice(0, qs, qe).to(torch::kFloat32);
    auto ki = kd.to(torch::kFloat32), vi = vd.to(torch::kFloat32);
    if (R > 1) { ki = at::repeat_interleave(ki, R, 1); vi = at::repeat_interleave(vi, R, 1); }
    auto a = at::einsum("qhd,khd->hqk", {qi, ki}) * scale;
    if (causal) {
      int64_t ql = qe - qs;
      auto m = at::triu(torch::ones({ql, kl}, qi.options().dtype(torch::kBool)), kl - ql + 1);
      a.masked_fill_(m.unsqueeze(0), -std::numeric_limits<float>::infinity());
    }
    a = at::softmax(a, -1).to(q.dtype());
    out.slice(0, qs, qe) = at::einsum("hqk,khd->qhd", {a, vi.to(q.dtype())});
  }
  return out;
}

struct TestOpts {
  torch::TensorOptions fp16;
  torch::Device dev;
  explicit TestOpts() : dev(flag_gems::test::default_device()) { fp16 = torch::TensorOptions().dtype(torch::kFloat16).device(dev); }
};

#define _AC(out, ref, atol) do {                                   \
  auto _r = flag_gems::accuracy_utils::gems_assert_close(          \
      out, ref, (out).scalar_type(), false, 1, atol);             \
  EXPECT_TRUE(_r.ok) << _r.message;                                 \
} while (0)

}  // namespace

class Fa3TritonWrapperTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (!flag_gems::test::is_device_available()) GTEST_SKIP() << "CUDA unavailable";
    if (!is_hopper()) GTEST_SKIP() << "FA3 Triton needs sm90+";
  }
};

// ── non-paged tests ────────────────────────────────────────────────────

TEST_F(Fa3TritonWrapperTest, NonPagedBs1) {
  TestOpts T; torch::manual_seed(42);
  std::vector<int32_t> ql{64}, kl{64};
  double sc = 1.0 / std::sqrt(128.0);
  auto q = torch::randn({64, 16, 128}, T.fp16), k = torch::randn({64, 8, 128}, T.fp16), v = torch::randn_like(k);
  auto cuq = cu_from_lens(ql, T.dev), cuk = cu_from_lens(kl, T.dev);
  c10::DeviceGuard g(T.dev);
  auto [out, lse] = flag_gems::flash_attn_varlen_fa3_triton_fwd(q,k,v,cuq,cuk,{},{},64,64,sc,true,-1,-1,0,{},{},{});
  flag_gems::test::synchronize();
  _AC(out, ref_nonpaged(q,k,v,cuq,cuk,sc,true), 2e-2f);
}

TEST_F(Fa3TritonWrapperTest, NonPagedBs3) {
  TestOpts T; torch::manual_seed(42);
  std::vector<int32_t> ql{1,1,70}, kl{1,1,70};
  double sc = 1.0 / std::sqrt(128.0);
  auto q = torch::randn({72, 16, 128}, T.fp16), k = torch::randn({72, 8, 128}, T.fp16), v = torch::randn_like(k);
  auto cuq = cu_from_lens(ql, T.dev), cuk = cu_from_lens(kl, T.dev);
  c10::DeviceGuard g(T.dev);
  auto [out, lse] = flag_gems::flash_attn_varlen_fa3_triton_fwd(q,k,v,cuq,cuk,{},{},70,70,sc,true,-1,-1,0,{},{},{});
  flag_gems::test::synchronize();
  _AC(out, ref_nonpaged(q,k,v,cuq,cuk,sc,true), 2e-2f);
}

TEST_F(Fa3TritonWrapperTest, NonPagedSoftcap) {
  TestOpts T; torch::manual_seed(42);
  double sc = 1.0 / std::sqrt(128.0), softcap = 10.0;
  auto q = torch::randn({128, 16, 128}, T.fp16), k = torch::randn({128, 8, 128}, T.fp16), v = torch::randn_like(k);
  auto cuq = cu_from_lens({128}, T.dev), cuk = cu_from_lens({128}, T.dev);
  c10::DeviceGuard g(T.dev);
  auto [out, lse] = flag_gems::flash_attn_varlen_fa3_triton_fwd(q,k,v,cuq,cuk,{},{},128,128,sc,true,-1,-1,softcap,{},{},{});
  flag_gems::test::synchronize();
  // softcap ref
  auto qi = q.to(torch::kFloat32) * sc, ki = k.to(torch::kFloat32), vi = v.to(torch::kFloat32);
  ki = at::repeat_interleave(ki, 2, 1); vi = at::repeat_interleave(vi, 2, 1);
  auto a = at::einsum("qhd,khd->hqk", {qi, ki});
  a = softcap * torch::tanh(a / softcap);
  a.masked_fill_(at::triu(torch::ones({128L,128L},T.dev).to(torch::kBool),1).unsqueeze(0), -std::numeric_limits<float>::infinity());
  a = at::softmax(a, -1).to(q.dtype());
  auto ref = at::einsum("hqk,khd->qhd", {a, vi.to(q.dtype())});
  _AC(out, ref, 2e-2f);
}

TEST_F(Fa3TritonWrapperTest, NonPagedTrace3) {
  TestOpts T; torch::manual_seed(42);
  std::vector<int32_t> ql, kl = {515};
  for (int i=1;i<=44;++i) ql.push_back(i);
  ql.insert(ql.end(),{105,121,137,153,169,185,201,217,233,249,265});
  kl.insert(kl.end(),20,514); kl.insert(kl.end(),20,513); kl.insert(kl.end(),14,512);
  int64_t tq=std::accumulate(ql.begin(),ql.end(),0L), tk=std::accumulate(kl.begin(),kl.end(),0L);
  int64_t mq=*std::max_element(ql.begin(),ql.end()), mk=*std::max_element(kl.begin(),kl.end());
  double sc=1.0/std::sqrt(128.0);
  auto q=torch::randn({tq,16,128},T.fp16),k=torch::randn({tk,8,128},T.fp16),v=torch::randn_like(k);
  auto cuq=cu_from_lens(ql,T.dev),cuk=cu_from_lens(kl,T.dev);
  c10::DeviceGuard g(T.dev);
  auto [out,lse] = flag_gems::flash_attn_varlen_fa3_triton_fwd(q,k,v,cuq,cuk,{},{},mq,mk,sc,true,-1,-1,0,{},{},{});
  flag_gems::test::synchronize();
  _AC(out, ref_nonpaged(q,k,v,cuq,cuk,sc,true), 8e-2f);  // mixed prefill+decode tolerance
}

// ── paged tests ─────────────────────────────────────────────────────────

TEST_F(Fa3TritonWrapperTest, PagedBasic) {
  TestOpts T; torch::manual_seed(42);
  std::vector<int32_t> ql{64,32}, kl{128,96};
  double sc=1.0/std::sqrt(128.0);
  auto q=torch::randn({96,16,128},T.fp16);
  auto kc=torch::randn({512,64,8,128},T.fp16), vc=torch::randn_like(kc);
  auto cuq=cu_from_lens(ql,T.dev);
  auto sk=torch::tensor(kl, torch::TensorOptions().dtype(torch::kInt32).device(T.dev));
  auto bt=torch::randint(0,512,{2L,3L}, torch::TensorOptions().dtype(torch::kInt32).device(T.dev));
  c10::DeviceGuard g(T.dev);
  auto [out,lse] = flag_gems::flash_attn_varlen_fa3_triton_fwd(q,kc,vc,cuq,{},sk,bt,64,128,sc,true,-1,-1,0,{},{},{});
  flag_gems::test::synchronize();
  _AC(out, ref_paged(q,kc,vc,cuq,sk,bt,sc,true), 2e-2f);
}

TEST_F(Fa3TritonWrapperTest, NonPagedLargeBf16) {
  TestOpts T; torch::manual_seed(42);
  int64_t sq = 8192;
  auto bf16 = T.fp16.dtype(torch::kBFloat16);
  auto q = torch::randn({sq * 2, 16, 128}, bf16);
  auto k = torch::randn({sq * 2, 8, 128}, bf16);
  auto v = torch::randn_like(k);
  auto cuq = cu_from_lens({sq, sq}, T.dev);
  auto cuk = cu_from_lens({sq, sq}, T.dev);
  c10::DeviceGuard g(T.dev);
  auto [out, lse] = flag_gems::flash_attn_varlen_fa3_triton_fwd(
      q, k, v, cuq, cuk, {}, {}, sq, sq, 0.088, true, -1, 0, 0.0, {}, {}, {});
  flag_gems::test::synchronize();
  auto ref = ref_nonpaged(q, k, v, cuq, cuk, 0.088, true);
  _AC(out, ref, 5e-2f);
}

TEST_F(Fa3TritonWrapperTest, NonPagedLargeFp16) {
  TestOpts T; torch::manual_seed(42);
  int64_t sq = 8192;
  auto q = torch::randn({sq * 2, 16, 128}, T.fp16);
  auto k = torch::randn({sq * 2, 8, 128}, T.fp16);
  auto v = torch::randn_like(k);
  auto cuq = cu_from_lens({sq, sq}, T.dev);
  auto cuk = cu_from_lens({sq, sq}, T.dev);
  c10::DeviceGuard g(T.dev);
  auto [out, lse] = flag_gems::flash_attn_varlen_fa3_triton_fwd(
      q, k, v, cuq, cuk, {}, {}, sq, sq, 0.088, true, -1, 0, 0.0, {}, {}, {});
  flag_gems::test::synchronize();
  auto ref = ref_nonpaged(q, k, v, cuq, cuk, 0.088, true);
  _AC(out, ref, 5e-2f);
}

// PR #4494: prefill_b4_s4k_d128_mha (32/32 MHA, cu_seqlens_k dense path)
TEST_F(Fa3TritonWrapperTest, NonPagedPr4494Mha4k) {
  TestOpts T;
  torch::manual_seed(2031);
  int64_t sq = 4096;
  auto q = torch::randn({sq * 4, 32, 128}, T.fp16) * 0.5;
  auto k = torch::randn({sq * 4, 32, 128}, T.fp16) * 0.5;
  auto v = torch::randn_like(k);
  auto cuq = cu_from_lens({sq, sq, sq, sq}, T.dev);
  auto cuk = cu_from_lens({sq, sq, sq, sq}, T.dev);
  double sc = 1.0 / std::sqrt(128.0);
  c10::DeviceGuard g(T.dev);
  auto [out, lse] = flag_gems::flash_attn_varlen_fa3_triton_fwd(
      q, k, v, cuq, cuk, {}, {}, sq, sq, sc, true, -1, -1, 0.0, {}, {}, {});
  flag_gems::test::synchronize();
  auto ref = ref_nonpaged(q, k, v, cuq, cuk, sc, true);
  _AC(out, ref, 5e-2f);
}

#undef _AC
