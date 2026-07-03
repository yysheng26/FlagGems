# """
# benchmark/test_flash_attn_varlen_gluon_func.py
# -----------------------------------------------
# Benchmark for the Gluon TMA+WGMMA flash-attention paths:
#   - non-paged varlen  (cu_seqlens_k, flat K/V)
#   - paged varlen      (cu_seqlens_k + block_table)

# Shapes mirror test_flash_attn_varlen_fa3_func.py as closely as possible.
# The baseline (torch_op) is the vllm FA3 implementation with fa_version=3.

# Shape format
# ------------
# non-paged: (cu_q, cu_k, nhead_q, nhead_k, head_size, window_size)
# paged:     (cu_q, kv_lens, nhead_q, nhead_k, head_size, block_size, num_blocks, window_size)
# """
# from typing import Any, List, Optional

# import pytest
# import torch

# import flag_gems

# from . import base, utils

# vendor_name = flag_gems.vendor_name


# def _is_hopper():
#     return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9


# # ---------------------------------------------------------------------------
# # Non-paged benchmark
# # ---------------------------------------------------------------------------

# class GluonVarlenNonPagedBenchmark(base.Benchmark):
#     """
#     Benchmark for the Gluon non-paged varlen path.

#     Shape categories (mirrors FA3 NonPaged benchmark)
#     --------------------------------------------------
#     A. Real trace (Qwen3-1.7B)
#     B. Synthetic prefill: bs=1 various seqlen / head_size
#     C. Synthetic prefill: bs=4/8
#     D. Synthetic decode: seqlen_q=1, various batch / kvcache
#     E. GQA variants: decode bs=32 sk=2048
#     F. head_size variants
#     """

#     def set_shapes(self, shape_file_path=None):
#         shapes = []

#         def cu(lens):
#             r = [0]
#             for l in lens:
#                 r.append(r[-1] + l)
#             return tuple(r)

#         # ── A. Real trace: Qwen3-1.7B ─────────────────────────────────────
#         # (cu_seqlens_q, cu_seqlens_k, nhead_q, nhead_k, head_size, window_size)
#         all_cu_q = [
#             (0, 512),
#             (0, 1, 2, 72),
#             tuple(range(0, 45)) + (105, 121, 137, 153, 169, 185, 201, 217, 233, 249, 265),
#             tuple(range(0, 196)) + (211, 226, 240, 253, 265),
#         ]
#         all_kv = [
#             (512,),
#             (1, 1, 70),
#             (515,) + (514,) * 20 + (513,) * 20 + (512,) * 14,
#             (2333,)
#             + (2331,) * 20 + (2330,) * 20 + (2329,) * 14
#             + (2328,) * 18 + (2327,) * 15 + (2326,) * 17
#             + (2325,) * 18 + (2324,) * 21 + (2323,) * 22
#             + (2322,) * 24 + (2321,) * 5
#             + (2320, 2319, 2318, 2317, 2316),
#         ]
#         for cu_q, kv in zip(all_cu_q, all_kv):
#             cu_k = cu(kv)
#             shapes.append((cu_q, cu_k, 16, 8, 128, (-1, -1)))

#         # ── B. Synthetic prefill: bs=1 various seqlen ─────────────────────
#         for seqlen in (128, 256, 512, 1024, 2048, 4096):
#             shapes.append(((0, seqlen), (0, seqlen), 16, 8, 128, (-1, -1)))

#         # prefill bs=1 various head_size
#         for hd in (64, 128, 256):
#             shapes.append(((0, 1024), (0, 1024), 16, 8, hd, (-1, -1)))

#         # ── C. Synthetic prefill: bs=4/8 ──────────────────────────────────
#         for seqlen in (256, 512, 1024, 2048):
#             shapes.append((cu(seqlen for _ in range(4)), cu(seqlen for _ in range(4)),
#                            16, 8, 128, (-1, -1)))
#         shapes.append((cu(512 for _ in range(8)), cu(512 for _ in range(8)),
#                        16, 8, 128, (-1, -1)))

#         # ── D. Synthetic decode: seqlen_q=1 ───────────────────────────────
#         for bs, kv_len in (
#             (1,   512), (1,  2048),
#             (8,   512), (8,  2048),
#             (16,  512), (16, 2048),
#             (32,  512), (32, 1024), (32, 2048),
#             (64, 1024), (64, 2048),
#             (128, 1024), (128, 2048),
#         ):
#             shapes.append((cu(1 for _ in range(bs)), cu(kv_len for _ in range(bs)),
#                            16, 8, 128, (-1, -1)))

#         # ── E. GQA variants: decode bs=32 sk=2048 ─────────────────────────
#         cu_q_d32 = cu(1 for _ in range(32))
#         cu_k_d32 = cu(2048 for _ in range(32))
#         for nq, nk in ((4, 4), (8, 2), (8, 1), (16, 2), (16, 1), (32, 8), (32, 4)):
#             shapes.append((cu_q_d32, cu_k_d32, nq, nk, 128, (-1, -1)))

#         # prefill bs=1 sq=1024
#         for nq, nk in ((4, 4), (8, 2), (16, 2), (32, 8)):
#             shapes.append(((0, 1024), (0, 1024), nq, nk, 128, (-1, -1)))

#         # ── F. head_size variants: decode bs=32 sk=2048 ───────────────────
#         for hd in (64, 128, 256):
#             shapes.append((cu_q_d32, cu_k_d32, 16, 8, hd, (-1, -1)))

#         self.shapes = shapes

#     def get_input_iter(self, dtype):
#         for cfg in self.shapes:
#             inp = self._make_input(cfg, dtype, self.device)
#             if inp is not None:
#                 yield inp

#     def _make_input(self, config, dtype, device):
#         cu_q_tup, cu_k_tup, nq, nk, hd, window_size = config

#         cu_q_list = list(cu_q_tup)
#         cu_k_list = list(cu_k_tup)
#         num_seqs   = len(cu_q_list) - 1
#         total_q    = cu_q_list[-1]
#         total_k    = cu_k_list[-1]
#         max_q_len  = max(cu_q_list[i+1] - cu_q_list[i] for i in range(num_seqs))
#         max_k_len  = max(cu_k_list[i+1] - cu_k_list[i] for i in range(num_seqs))
#         scale      = hd ** -0.5

#         q   = torch.randn(total_q, nq, hd, dtype=dtype, device=device)
#         k   = torch.randn(total_k, nk, hd, dtype=dtype, device=device)
#         v   = torch.randn_like(k)
#         out = torch.empty_like(q)

#         cu_q_t = torch.tensor(cu_q_list, dtype=torch.int32, device=device)
#         cu_k_t = torch.tensor(cu_k_list, dtype=torch.int32, device=device)

#         return (
#             q, k, v,
#             max_q_len,           # max_seqlen_q
#             cu_q_t,              # cu_seqlens_q
#             max_k_len,           # max_seqlen_k
#             cu_k_t,              # cu_seqlens_k  <- non-paged marker
#             None,                # seqused_k
#             None,                # q_v
#             0.0,                 # dropout_p
#             scale,               # softmax_scale
#             True,                # causal
#             list(window_size),   # window_size
#             0.0,                 # softcap
#             None,                # alibi_slopes
#             False,               # deterministic
#             False,               # return_attn_probs
#             None,                # block_table  <- None = non-paged
#             False,               # return_softmax_lse
#             out,                 # out
#             None,                # scheduler_metadata
#             None, None, None,    # q/k/v_descale
#             {"fa_version": 3, "use_gluon": True},
#         )


# # ---------------------------------------------------------------------------
# # Paged benchmark
# # ---------------------------------------------------------------------------

# class GluonVarlenPagedBenchmark(base.Benchmark):
#     """
#     Benchmark for the Gluon paged varlen path.

#     Uses seqused_k + block_table (not cu_seqlens_k) to route through Gluon paged kernel.

#     Shape categories (mirrors FA3 paged benchmark)
#     -----------------------------------------------
#     A. Real trace (Qwen3-1.7B)
#     B. Synthetic prefill: bs=1 various seqlen
#     C. Synthetic decode: seqlen_q=1, various batch/kvcache
#     D. GQA variants
#     E. head_size variants
#     F. block_size variants: 64 / 128
#     """

#     def set_shapes(self, shape_file_path=None):
#         # format: (cu_q, kv_lens_tuple, nq, nk, head_size, block_size, num_blocks, window)
#         shapes = []

#         def cu(lens):
#             r = [0]
#             for l in lens:
#                 r.append(r[-1] + l)
#             return tuple(r)

#         # ── A. Real trace: Qwen3-1.7B ─────────────────────────────────────
#         # same sequence shapes as FA3 paged benchmark
#         all_cu_q = [
#             (0, 512),
#             (0, 1, 2, 72),
#             tuple(range(0, 45)) + (105, 121, 137, 153, 169, 185, 201, 217, 233, 249, 265),
#         ]
#         all_sk = [
#             (512,),
#             (1, 1, 70),
#             (515,) + (514,) * 20 + (513,) * 20 + (512,) * 14,
#         ]
#         for cu_q, sk in zip(all_cu_q, all_sk):
#             max_kv    = max(sk)
#             block_size = 64
#             num_blocks = sum((k + block_size - 1) // block_size for k in sk) + 64
#             shapes.append((cu_q, sk, 16, 8, 128, block_size, num_blocks, (-1, -1)))

#         # ── B. Synthetic prefill: bs=1 various seqlen ─────────────────────
#         for seqlen in (128, 256, 512, 1024, 2048, 4096):
#             block_size = 64
#             num_blocks = (seqlen + block_size - 1) // block_size + 32
#             shapes.append(((0, seqlen), (seqlen,), 16, 8, 128, block_size, num_blocks, (-1, -1)))

#         # prefill bs=1 various head_size
#         for hd in (64, 128, 256):
#             block_size = 64
#             num_blocks = (1024 + block_size - 1) // block_size + 32
#             shapes.append(((0, 1024), (1024,), 16, 8, hd, block_size, num_blocks, (-1, -1)))

#         # prefill bs=4 various seqlen
#         for seqlen in (256, 512, 1024, 2048):
#             block_size = 64
#             num_blocks = 4 * (seqlen + block_size - 1) // block_size + 64
#             shapes.append((cu(seqlen for _ in range(4)), (seqlen,) * 4,
#                            16, 8, 128, block_size, num_blocks, (-1, -1)))

#         # prefill bs=8, seqlen=512
#         block_size = 64
#         num_blocks = 8 * (512 + block_size - 1) // block_size + 64
#         shapes.append((cu(512 for _ in range(8)), (512,) * 8,
#                        16, 8, 128, block_size, num_blocks, (-1, -1)))

#         # ── C. Synthetic decode: seqlen_q=1 ───────────────────────────────
#         for bs, kv_len in (
#             (1,   512), (1,  2048),
#             (8,   512), (8,  2048),
#             (16,  512), (16, 2048),
#             (32,  512), (32, 1024), (32, 2048),
#             (64, 1024), (64, 2048),
#             (128, 1024), (128, 2048),
#         ):
#             block_size = 64
#             num_blocks = bs * (kv_len + block_size - 1) // block_size + 64
#             shapes.append((cu(1 for _ in range(bs)), (kv_len,) * bs,
#                            16, 8, 128, block_size, num_blocks, (-1, -1)))

#         # ── D. GQA variants: decode bs=32 sk=2048 ─────────────────────────
#         cu_q_d32 = cu(1 for _ in range(32))
#         sk_d32   = (2048,) * 32
#         block_d  = 64
#         nb_d     = 32 * (2048 + block_d - 1) // block_d + 64
#         for nq, nk in ((4, 4), (8, 2), (8, 1), (16, 2), (16, 1), (32, 8), (32, 4)):
#             shapes.append((cu_q_d32, sk_d32, nq, nk, 128, block_d, nb_d, (-1, -1)))

#         # prefill bs=1 sq=1024 GQA variants
#         for nq, nk in ((4, 4), (8, 2), (16, 2), (32, 8)):
#             num_blocks_p = (1024 + 63) // 64 + 32
#             shapes.append(((0, 1024), (1024,), nq, nk, 128, 64, num_blocks_p, (-1, -1)))

#         # ── E. head_size variants: decode bs=32 sk=2048 ───────────────────
#         for hd in (64, 128, 256):
#             shapes.append((cu_q_d32, sk_d32, 16, 8, hd, block_d, nb_d, (-1, -1)))

#         # ── F. block_size variants ─────────────────────────────────────────
#         for block_size in (64, 128):
#             nb = 32 * (2048 + block_size - 1) // block_size + 64
#             shapes.append((cu_q_d32, sk_d32, 16, 8, 128, block_size, nb, (-1, -1)))
#             # prefill
#             nb_p = (1024 + block_size - 1) // block_size + 32
#             shapes.append(((0, 1024), (1024,), 16, 8, 128, block_size, nb_p, (-1, -1)))

#         self.shapes = shapes

#     def get_input_iter(self, dtype):
#         for cfg in self.shapes:
#             inp = self._make_input(cfg, dtype, self.device)
#             if inp is not None:
#                 yield inp

#     def _make_input(self, config, dtype, device):
#         cu_q_tup, kv_lens, nq, nk, hd, block_size, num_blocks, window_size = config

#         cu_q_list  = list(cu_q_tup)
#         kv_list    = list(kv_lens)
#         num_seqs   = len(cu_q_list) - 1
#         total_q    = cu_q_list[-1]
#         max_q_len  = max(cu_q_list[i+1] - cu_q_list[i] for i in range(num_seqs))
#         max_kv_len = max(kv_list)
#         scale      = hd ** -0.5

#         q       = torch.randn(total_q,   nq, hd, dtype=dtype, device=device)
#         k_cache = torch.randn(num_blocks, block_size, nk, hd, dtype=dtype, device=device)
#         v_cache = torch.randn_like(k_cache)
#         out     = torch.empty_like(q)

#         cu_q_t   = torch.tensor(cu_q_list, dtype=torch.int32, device=device)
#         sk_t     = torch.tensor(kv_list,   dtype=torch.int32, device=device)

#         max_pgs = (max_kv_len + block_size - 1) // block_size
#         bt = torch.randint(0, num_blocks, (num_seqs, max_pgs),
#                            dtype=torch.int32, device=device)

#         return (
#             q, k_cache, v_cache,
#             max_q_len,           # max_seqlen_q
#             cu_q_t,              # cu_seqlens_q
#             max_kv_len,          # max_seqlen_k
#             None,                # cu_seqlens_k  <- None for paged
#             sk_t,                # seqused_k     <- paged path
#             None,                # q_v
#             0.0,                 # dropout_p
#             scale,               # softmax_scale
#             True,                # causal
#             list(window_size),   # window_size
#             0.0,                 # softcap
#             None,                # alibi_slopes
#             False,               # deterministic
#             False,               # return_attn_probs
#             bt,                  # block_table
#             False,               # return_softmax_lse
#             out,                 # out
#             None,                # scheduler_metadata
#             None, None, None,    # q/k/v_descale
#             {"fa_version": 3, "use_gluon": True},
#         )


# # ---------------------------------------------------------------------------
# # pytest entry points
# # ---------------------------------------------------------------------------

# @pytest.mark.skipif(not _is_hopper(), reason="Gluon requires Hopper GPU (sm_90+)")
# @pytest.mark.skipif(
#     utils.SkipVersion("vllm", "<0.9"),
#     reason="vLLM < 0.9 does not expose flash_attn_varlen_func",
# )
# @pytest.mark.skipif(vendor_name in ("hygon", "cambricon"), reason="Not working")
# @pytest.mark.flash_attn_varlen_func
# def test_gluon_varlen_non_paged_benchmark(monkeypatch):
#     monkeypatch.setenv("VLLM_CONFIGURE_LOGGING", "0")

#     from vllm.vllm_flash_attn.flash_attn_interface import flash_attn_varlen_func as _vllm_fa

#     def vllm_fa3(*args, **kwargs):
#         kwargs.pop("fa_version", None)
#         kwargs.pop("use_gluon", None)  # FlagGems-only; vLLM does not accept it
#         return _vllm_fa(*args, fa_version=3, **kwargs)

#     bench = GluonVarlenNonPagedBenchmark(
#         op_name="flash_attn_varlen_gluon_non_paged",
#         torch_op=vllm_fa3,
#         gems_op=flag_gems.ops.flash_attn_varlen_func,
#         dtypes=[torch.float16, torch.bfloat16],
#     )
#     bench.run()


# @pytest.mark.skipif(not _is_hopper(), reason="Gluon requires Hopper GPU (sm_90+)")
# @pytest.mark.skipif(
#     utils.SkipVersion("vllm", "<0.9"),
#     reason="vLLM < 0.9 does not expose flash_attn_varlen_func",
# )
# @pytest.mark.skipif(vendor_name in ("hygon", "cambricon"), reason="Not working")
# @pytest.mark.flash_attn_varlen_func
# def test_gluon_varlen_paged_benchmark(monkeypatch):
#     monkeypatch.setenv("VLLM_CONFIGURE_LOGGING", "0")

#     from vllm.vllm_flash_attn.flash_attn_interface import flash_attn_varlen_func as _vllm_fa

#     def vllm_fa3(*args, **kwargs):
#         kwargs.pop("fa_version", None)
#         kwargs.pop("use_gluon", None)  # FlagGems-only; vLLM does not accept it
#         # Benchmark passes all params as positional args; vllm's paged path
#         # requires seqused_k (index 7) instead of cu_seqlens_k (index 6).
#         # Convert when block_table (index 17) is present.
#         args = list(args)
#         CU_K_IDX, SEQUSED_K_IDX, BLOCK_TABLE_IDX = 6, 7, 17
#         if (len(args) > BLOCK_TABLE_IDX
#                 and args[BLOCK_TABLE_IDX] is not None
#                 and args[CU_K_IDX] is not None):
#             cu_k = args[CU_K_IDX]
#             args[SEQUSED_K_IDX] = (cu_k[1:] - cu_k[:-1]).to(torch.int32)
#             args[CU_K_IDX] = None   # vllm paged: cu_seqlens_k must be None
#         return _vllm_fa(*args, fa_version=3, **kwargs)

#     bench = GluonVarlenPagedBenchmark(
#         op_name="flash_attn_varlen_gluon_paged",
#         torch_op=vllm_fa3,
#         gems_op=flag_gems.ops.flash_attn_varlen_func,
#         dtypes=[torch.float16, torch.bfloat16],
#     )
#     bench.run()


# from typing import Any, List, Optional

# import pytest
# import torch

# import flag_gems

# from . import base, utils

# vendor_name = flag_gems.vendor_name


# def _is_hopper():
#     return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9


# def _cu(lens):
#     r = [0]
#     for l in lens:
#         r.append(r[-1] + l)
#     return tuple(r)


# # ---------------------------------------------------------------------------
# # Quick non-paged benchmark：Qwen3-1.7B 真实 trace（4 条）
# # ---------------------------------------------------------------------------

# class GluonVarlenNonPagedQuickBenchmark(base.Benchmark):

#     def set_shapes(self, shape_file_path=None):
#         all_cu_q = [
#             (0, 512),
#             (0, 1, 2, 72),
#             tuple(range(0, 45)) + (105, 121, 137, 153, 169, 185, 201, 217, 233, 249, 265),
#             tuple(range(0, 196)) + (211, 226, 240, 253, 265),
#         ]
#         all_kv = [
#             (512,),
#             (1, 1, 70),
#             (515,) + (514,) * 20 + (513,) * 20 + (512,) * 14,
#             (2333,)
#             + (2331,) * 20 + (2330,) * 20 + (2329,) * 14
#             + (2328,) * 18 + (2327,) * 15 + (2326,) * 17
#             + (2325,) * 18 + (2324,) * 21 + (2323,) * 22
#             + (2322,) * 24 + (2321,) * 5
#             + (2320, 2319, 2318, 2317, 2316),
#         ]
#         self.shapes = [
#             (cu_q, _cu(kv), 16, 8, 128, (-1, -1))
#             for cu_q, kv in zip(all_cu_q, all_kv)
#         ]

#     def get_input_iter(self, dtype):
#         for cfg in self.shapes:
#             inp = self._make_input(cfg, dtype, self.device)
#             if inp is not None:
#                 yield inp

#     def _make_input(self, config, dtype, device):
#         cu_q_tup, cu_k_tup, nq, nk, hd, window_size = config
#         cu_q_list = list(cu_q_tup)
#         cu_k_list = list(cu_k_tup)
#         num_seqs  = len(cu_q_list) - 1
#         total_q   = cu_q_list[-1]
#         total_k   = cu_k_list[-1]
#         max_q_len = max(cu_q_list[i+1] - cu_q_list[i] for i in range(num_seqs))
#         max_k_len = max(cu_k_list[i+1] - cu_k_list[i] for i in range(num_seqs))
#         scale     = hd ** -0.5

#         q   = torch.randn(total_q, nq, hd, dtype=dtype, device=device)
#         k   = torch.randn(total_k, nk, hd, dtype=dtype, device=device)
#         v   = torch.randn_like(k)
#         out = torch.empty_like(q)
#         cu_q_t = torch.tensor(cu_q_list, dtype=torch.int32, device=device)
#         cu_k_t = torch.tensor(cu_k_list, dtype=torch.int32, device=device)

#         return (
#             q, k, v,
#             max_q_len, cu_q_t, max_k_len, cu_k_t,
#             None, None, 0.0, scale, True, list(window_size),
#             0.0, None, False, False, None, False, out, None,
#             None, None, None,
#             {"fa_version": 3, "use_gluon": True},
#         )


# # ---------------------------------------------------------------------------
# # Quick paged benchmark：Qwen3-1.7B 真实 trace（前 3 条）
# # ---------------------------------------------------------------------------

# class GluonVarlenPagedQuickBenchmark(base.Benchmark):

#     def set_shapes(self, shape_file_path=None):
#         all_cu_q = [
#             (0, 512),
#             (0, 1, 2, 72),
#             tuple(range(0, 45)) + (105, 121, 137, 153, 169, 185, 201, 217, 233, 249, 265),
#         ]
#         all_sk = [
#             (512,),
#             (1, 1, 70),
#             (515,) + (514,) * 20 + (513,) * 20 + (512,) * 14,
#         ]
#         block_size = 64
#         self.shapes = [
#             (cu_q, sk, 16, 8, 128, block_size,
#              sum((k + block_size - 1) // block_size for k in sk) + 64,
#              (-1, -1))
#             for cu_q, sk in zip(all_cu_q, all_sk)
#         ]

#     def get_input_iter(self, dtype):
#         for cfg in self.shapes:
#             inp = self._make_input(cfg, dtype, self.device)
#             if inp is not None:
#                 yield inp

#     def _make_input(self, config, dtype, device):
#         cu_q_tup, kv_lens, nq, nk, hd, block_size, num_blocks, window_size = config
#         cu_q_list  = list(cu_q_tup)
#         kv_list    = list(kv_lens)
#         num_seqs   = len(cu_q_list) - 1
#         total_q    = cu_q_list[-1]
#         max_q_len  = max(cu_q_list[i+1] - cu_q_list[i] for i in range(num_seqs))
#         max_kv_len = max(kv_list)
#         scale      = hd ** -0.5

#         q       = torch.randn(total_q,    nq, hd, dtype=dtype, device=device)
#         k_cache = torch.randn(num_blocks, block_size, nk, hd, dtype=dtype, device=device)
#         v_cache = torch.randn_like(k_cache)
#         out     = torch.empty_like(q)
#         cu_q_t  = torch.tensor(cu_q_list, dtype=torch.int32, device=device)
#         sk_t    = torch.tensor(kv_list,   dtype=torch.int32, device=device)
#         max_pgs = (max_kv_len + block_size - 1) // block_size
#         bt      = torch.randint(0, num_blocks, (num_seqs, max_pgs),
#                                 dtype=torch.int32, device=device)

#         return (
#             q, k_cache, v_cache,
#             max_q_len, cu_q_t, max_kv_len, None,
#             sk_t, None, 0.0, scale, True, list(window_size),
#             0.0, None, False, False, bt, False, out, None,
#             None, None, None,
#             {"fa_version": 3, "use_gluon": True},
#         )


# # ---------------------------------------------------------------------------
# # pytest entry points
# # ---------------------------------------------------------------------------

# @pytest.mark.skipif(not _is_hopper(), reason="Gluon requires Hopper GPU (sm_90+)")
# @pytest.mark.skipif(
#     utils.SkipVersion("vllm", "<0.9"),
#     reason="vLLM < 0.9 does not expose flash_attn_varlen_func",
# )
# @pytest.mark.skipif(vendor_name in ("hygon", "cambricon"), reason="Not working")
# @pytest.mark.flash_attn_varlen_func
# def test_gluon_varlen_non_paged_quick_benchmark(monkeypatch):
#     monkeypatch.setenv("VLLM_CONFIGURE_LOGGING", "0")

#     from vllm.vllm_flash_attn.flash_attn_interface import flash_attn_varlen_func as _vllm_fa

#     def vllm_fa3(*args, **kwargs):
#         kwargs.pop("fa_version", None)
#         kwargs.pop("use_gluon", None)
#         return _vllm_fa(*args, fa_version=3, **kwargs)

#     bench = GluonVarlenNonPagedQuickBenchmark(
#         op_name="flash_attn_varlen_gluon_non_paged_quick",
#         torch_op=vllm_fa3,
#         gems_op=flag_gems.ops.flash_attn_varlen_func,
#         dtypes=[torch.float16, torch.bfloat16],
#     )
#     bench.run()


# @pytest.mark.skipif(not _is_hopper(), reason="Gluon requires Hopper GPU (sm_90+)")
# @pytest.mark.skipif(
#     utils.SkipVersion("vllm", "<0.9"),
#     reason="vLLM < 0.9 does not expose flash_attn_varlen_func",
# )
# @pytest.mark.skipif(vendor_name in ("hygon", "cambricon"), reason="Not working")
# @pytest.mark.flash_attn_varlen_func
# def test_gluon_varlen_paged_quick_benchmark(monkeypatch):
#     monkeypatch.setenv("VLLM_CONFIGURE_LOGGING", "0")

#     from vllm.vllm_flash_attn.flash_attn_interface import flash_attn_varlen_func as _vllm_fa

#     def vllm_fa3(*args, **kwargs):
#         kwargs.pop("fa_version", None)
#         kwargs.pop("use_gluon", None)
#         args = list(args)
#         CU_K_IDX, SEQUSED_K_IDX, BLOCK_TABLE_IDX = 6, 7, 17
#         if (len(args) > BLOCK_TABLE_IDX
#                 and args[BLOCK_TABLE_IDX] is not None
#                 and args[CU_K_IDX] is not None):
#             cu_k = args[CU_K_IDX]
#             args[SEQUSED_K_IDX] = (cu_k[1:] - cu_k[:-1]).to(torch.int32)
#             args[CU_K_IDX] = None
#         return _vllm_fa(*args, fa_version=3, **kwargs)

#     bench = GluonVarlenPagedQuickBenchmark(
#         op_name="flash_attn_varlen_gluon_paged_quick",
#         torch_op=vllm_fa3,
#         gems_op=flag_gems.ops.flash_attn_varlen_func,
#         dtypes=[torch.float16, torch.bfloat16],
#     )
#     bench.run()


# =============================================================================
# FlagTree PR #707 perf table: 25 workloads × 2 dtype (vLLM FA3 vs Gems Gluon)
#
# PR #707 只有表名/顺序；shape 倒推自 FlagGems PR #4494：
#   4 条 paged trace → benchmark/test_flash_attn_varlen_func.py (16/8, d=128)
#   21 条合成 case  → benchmark/test_flash_attn_varlen_fa3_func.py (32/32, 32/8)
#
# Usage:
#   USE_C_EXTENSION=1 TORCH_USE_RTLD_GLOBAL=1 CUDA_VISIBLE_DEVICES=2 \\
#   FLAGGEMS_SOURCE_DIR=/root/workspace/FlagGems/src/flag_gems \\
#   PYTHONPATH=/root/workspace/FlagGems/src \\
#   pytest benchmark/test_flash_attn_varlen_gluon_func.py::test_flash_attn_varlen_pr707_gluon_benchmark -v -s
# =============================================================================
from __future__ import annotations

import gc
import math
from dataclasses import asdict, dataclass
from typing import Any, List, Optional, Tuple, Union

import pytest
import torch

import flag_gems

from . import base, consts, utils
from .conftest import Config, emit_record_logger, update_result
from .consts import BenchmarkMetrics, BenchmarkResult

vendor_name = flag_gems.vendor_name


def _pr4494_is_hopper() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9


@dataclass(frozen=True)
class _Pr4494HopperShape:
    name: str
    seq_lens: List[Tuple[int, int]]
    nh_q: int
    nh_k: int
    head_dim: int
    causal: bool
    paged: bool = False
    block_size: int = 16
    overcommit: float = 1.5


@dataclass(frozen=True)
class _Pr4494QwenShape:
    name: str
    cu_seqlens_q: Tuple[int, ...]
    seqused_k: Tuple[int, ...]
    nh_q: int = 16
    nh_k: int = 8
    head_dim: int = 128
    block_size: int = 16
    num_blocks: int = 2000


@dataclass(frozen=True)
class _Pr4494Workload:
    suite: str
    shape: Union[_Pr4494QwenShape, _Pr4494HopperShape]


def _pr4494_qwen_shapes() -> List[_Pr4494QwenShape]:
    all_cu_q = [
        (0, 512),
        (0, 1, 2, 72),
        tuple(range(0, 45))
        + (105, 121, 137, 153, 169, 185, 201, 217, 233, 249, 265),
        tuple(range(0, 196)) + (211, 226, 240, 253, 265),
    ]
    all_sk = [
        (512,),
        (1, 1, 70),
        (515,) + (514,) * 20 + (513,) * 20 + (512,) * 14,
        (2333,)
        + (2331,) * 20
        + (2330,) * 20
        + (2329,) * 14
        + (2328,) * 18
        + (2327,) * 15
        + (2326,) * 17
        + (2325,) * 18
        + (2324,) * 21
        + (2323,) * 22
        + (2322,) * 24
        + (2321,) * 5
        + (2320, 2319, 2318, 2317, 2316),
    ]
    names = [
        "qwen_trace0_prefill_bs1_q512",
        "qwen_trace1_mixed_bs3_maxq70",
        "qwen_trace2_decode_bs55_maxq61",
        "qwen_trace3_decode_bs201_maxq16",
    ]
    return [
        _Pr4494QwenShape(names[i], all_cu_q[i], all_sk[i])
        for i in range(len(all_cu_q))
    ]


def _pr4494_hopper_benchmark_shapes() -> List[_Pr4494HopperShape]:
    prefill = [
        _Pr4494HopperShape("prefill_b4_s2k_d128_mha", [(2048, 2048)] * 4, 32, 32, 128, True),
        _Pr4494HopperShape("prefill_b4_s4k_d128_mha", [(4096, 4096)] * 4, 32, 32, 128, True),
        _Pr4494HopperShape("prefill_b4_s8k_d128_mha", [(8192, 8192)] * 4, 32, 32, 128, True),
        _Pr4494HopperShape("prefill_b2_s16k_d128_mha", [(16384, 16384)] * 2, 32, 32, 128, True),
        _Pr4494HopperShape("prefill_b4_s4k_d128_gqa4", [(4096, 4096)] * 4, 32, 8, 128, True),
        _Pr4494HopperShape("prefill_b4_s8k_d128_gqa4", [(8192, 8192)] * 4, 32, 8, 128, True),
        _Pr4494HopperShape("prefill_b8_s2k_d64_mha", [(2048, 2048)] * 8, 16, 16, 64, False),
    ]
    decode = [
        _Pr4494HopperShape("decode_b16_kv1k_d128_gqa4", [(1, 1024)] * 16, 32, 8, 128, True),
        _Pr4494HopperShape("decode_b8_kv1k_d192_gqa4", [(1, 1024)] * 8, 32, 8, 192, True),
        _Pr4494HopperShape("decode_b8_kv1k_d256_gqa4", [(1, 1024)] * 8, 32, 8, 256, True),
        _Pr4494HopperShape(
            "decode_b16_mixed_d128_gqa4",
            [(1, 512), (1, 1024), (1, 2048), (1, 4096)] * 4,
            32,
            8,
            128,
            True,
        ),
        _Pr4494HopperShape("decode_b32_kv2k_d128_gqa4", [(1, 2048)] * 32, 32, 8, 128, True),
    ]
    varlen = [
        _Pr4494HopperShape(
            "varlen_mixed_d128_gqa4",
            [(2048, 2048), (1, 4096), (1, 4096), (1024, 1024), (1, 8192), (1, 1024)],
            32,
            8,
            128,
            True,
        ),
        _Pr4494HopperShape(
            "varlen_serve_b32_1pf_31dec_d128_gqa4",
            [(2048, 2048)] + [(1, 1024 + 64 * i) for i in range(31)],
            32,
            8,
            128,
            True,
        ),
        _Pr4494HopperShape(
            "varlen_longtail_d128_gqa4",
            [(16384, 16384)] + [(256, 256)] * 16,
            32,
            8,
            128,
            True,
        ),
    ]
    paged = [
        _Pr4494HopperShape(
            "paged_decode_b16_kvmix_bs16_d128_gqa4",
            [(1, 1024 + 256 * i) for i in range(16)],
            32,
            8,
            128,
            True,
            paged=True,
        ),
        _Pr4494HopperShape(
            "paged_decode_b8_bs16_d192_gqa4",
            [(1, 1024 + 128 * i) for i in range(8)],
            32,
            8,
            192,
            True,
            paged=True,
        ),
        _Pr4494HopperShape(
            "paged_decode_b8_bs16_d256_gqa4",
            [(1, 1024 + 128 * i) for i in range(8)],
            32,
            8,
            256,
            True,
            paged=True,
        ),
        _Pr4494HopperShape(
            "paged_decode_b64_bs16_d128_gqa4",
            [(1, 512 + 128 * i) for i in range(64)],
            32,
            8,
            128,
            True,
            paged=True,
        ),
        _Pr4494HopperShape(
            "paged_serve_b32_1pf_31dec_bs16_d128_gqa4",
            [(2048, 2048)] + [(1, 1024 + 96 * i) for i in range(31)],
            32,
            8,
            128,
            True,
            paged=True,
        ),
        _Pr4494HopperShape(
            "paged_uniform_b4_s4k_bs16_d128_mha",
            [(4096, 4096)] * 4,
            32,
            32,
            128,
            True,
            paged=True,
        ),
    ]
    return prefill + decode + varlen + paged


_PR707_QWEN_TRACE_ORDER = (3, 0, 2, 1)
_PR707_QWEN_DISPLAY_NAMES = (
    "paged_decodeish_long_k_tq265_q16_k2333_h16_hk8_d128",
    "paged_medium_or_prefill_tq512_q512_k512_h16_hk8_d128",
    "paged_mixed_short_tq265_q61_k515_h16_hk8_d128",
    "paged_short_tq72_q70_k70_h16_hk8_d128",
)
_PR707_HOPPER_CASE_ORDER = (
    "decode_b16_kv1k_d128_gqa4",
    "decode_b16_mixed_d128_gqa4",
    "decode_b32_kv2k_d128_gqa4",
    "decode_b8_kv1k_d192_gqa4",
    "decode_b8_kv1k_d256_gqa4",
    "paged_decode_b16_kvmix_bs16_d128_gqa4",
    "paged_decode_b64_bs16_d128_gqa4",
    "paged_decode_b8_bs16_d192_gqa4",
    "paged_decode_b8_bs16_d256_gqa4",
    "paged_serve_b32_1pf_31dec_bs16_d128_gqa4",
    "paged_uniform_b4_s4k_bs16_d128_mha",
    "prefill_b2_s16k_d128_mha",
    "prefill_b4_s2k_d128_mha",
    "prefill_b4_s4k_d128_gqa4",
    "prefill_b4_s4k_d128_mha",
    "prefill_b4_s8k_d128_gqa4",
    "prefill_b4_s8k_d128_mha",
    "prefill_b8_s2k_d64_mha",
    "varlen_longtail_d128_gqa4",
    "varlen_mixed_d128_gqa4",
    "varlen_serve_b32_1pf_31dec_d128_gqa4",
)


def _pr4494_all_workloads() -> List[_Pr4494Workload]:
    qwen_shapes = _pr4494_qwen_shapes()
    workloads = []
    for i, trace_idx in enumerate(_PR707_QWEN_TRACE_ORDER):
        base = qwen_shapes[trace_idx]
        shape = _Pr4494QwenShape(
            _PR707_QWEN_DISPLAY_NAMES[i],
            base.cu_seqlens_q,
            base.seqused_k,
            base.nh_q,
            base.nh_k,
            base.head_dim,
            base.block_size,
            base.num_blocks,
        )
        workloads.append(_Pr4494Workload("qwenCase", shape))
    hopper_by_name = {s.name: s for s in _pr4494_hopper_benchmark_shapes()}
    workloads.extend(
        _Pr4494Workload("prefillDecodePageCase", hopper_by_name[name])
        for name in _PR707_HOPPER_CASE_ORDER
    )
    return workloads


def _pr4494_make_qwen_input(
    shape: _Pr4494QwenShape, dtype: torch.dtype, device: str, seed: int
) -> tuple:
    gen = torch.Generator(device=device).manual_seed(seed)
    cu_q = list(shape.cu_seqlens_q)
    sk = list(shape.seqused_k)
    num_seqs = len(cu_q) - 1
    total_q = cu_q[-1]
    max_q = max(cu_q[i + 1] - cu_q[i] for i in range(num_seqs))
    max_k = max(sk)
    scale = shape.head_dim**-0.5

    q = torch.randn(
        total_q, shape.nh_q, shape.head_dim, dtype=dtype, device=device, generator=gen
    )
    out = torch.empty_like(q)
    k_cache = torch.randn(
        shape.num_blocks,
        shape.block_size,
        shape.nh_k,
        shape.head_dim,
        dtype=dtype,
        device=device,
        generator=gen,
    )
    v_cache = torch.randn_like(k_cache)
    cu_q_t = torch.tensor(cu_q, dtype=torch.int32, device=device)
    sk_t = torch.tensor(sk, dtype=torch.int32, device=device)
    max_pages = (max_k + shape.block_size - 1) // shape.block_size
    block_table = torch.randint(
        0,
        shape.num_blocks,
        (num_seqs, max_pages),
        dtype=torch.int32,
        device=device,
        generator=gen,
    )

    return (
        q,
        k_cache,
        v_cache,
        max_q,
        cu_q_t,
        max_k,
        None,
        sk_t,
        None,
        0.0,
        scale,
        True,
        [-1, -1],
        0.0,
        None,
        False,
        False,
        block_table,
        False,
        out,
        None,
        None,
        None,
        None,
        {"fa_version": 3, "use_gluon": True},
    )


def _pr4494_make_hopper_dense_input(
    shape: _Pr4494HopperShape, dtype: torch.dtype, device: str, seed: int
) -> tuple:
    gen = torch.Generator(device=device).manual_seed(seed)
    cu_q = [0]
    cu_k = [0]
    for q_len, k_len in shape.seq_lens:
        cu_q.append(cu_q[-1] + q_len)
        cu_k.append(cu_k[-1] + k_len)

    q = torch.randn(
        (cu_q[-1], shape.nh_q, shape.head_dim),
        dtype=dtype,
        device=device,
        generator=gen,
    ) * 0.5
    k = torch.randn(
        (cu_k[-1], shape.nh_k, shape.head_dim),
        dtype=dtype,
        device=device,
        generator=gen,
    ) * 0.5
    v = torch.randn_like(k)
    out = torch.empty_like(q)
    cu_q_t = torch.tensor(cu_q, dtype=torch.int32, device=device)
    cu_k_t = torch.tensor(cu_k, dtype=torch.int32, device=device)
    max_q = max(q_len for q_len, _ in shape.seq_lens)
    max_k = max(k_len for _, k_len in shape.seq_lens)
    scale = 1.0 / math.sqrt(shape.head_dim)

    return (
        q,
        k,
        v,
        max_q,
        cu_q_t,
        max_k,
        cu_k_t,
        None,
        None,
        0.0,
        scale,
        shape.causal,
        [-1, -1],
        0.0,
        None,
        False,
        False,
        None,
        False,
        out,
        None,
        None,
        None,
        None,
        {"fa_version": 3, "use_gluon": True},
    )


def _pr4494_make_hopper_paged_input(
    shape: _Pr4494HopperShape, dtype: torch.dtype, device: str, seed: int
) -> tuple:
    gen = torch.Generator(device=device).manual_seed(seed)
    cpu_gen = torch.Generator().manual_seed(seed + 1)
    block_size = shape.block_size

    cu_q = [0]
    for q_len, _ in shape.seq_lens:
        cu_q.append(cu_q[-1] + q_len)
    q = torch.randn(
        (cu_q[-1], shape.nh_q, shape.head_dim),
        dtype=dtype,
        device=device,
        generator=gen,
    ) * 0.5
    out = torch.empty_like(q)

    blocks_per_req = [
        (k_len + block_size - 1) // block_size for _, k_len in shape.seq_lens
    ]
    max_blocks_per_req = max(blocks_per_req)
    total_virtual_blocks = sum(blocks_per_req)
    num_physical_blocks = max(1, int(total_virtual_blocks * shape.overcommit))

    perm = torch.randperm(num_physical_blocks, generator=cpu_gen)[:total_virtual_blocks]
    block_table = torch.zeros(
        (len(shape.seq_lens), max_blocks_per_req), dtype=torch.int32, device=device
    )
    offset = 0
    for req_idx, num_blocks in enumerate(blocks_per_req):
        block_table[req_idx, :num_blocks] = perm[offset : offset + num_blocks].to(
            dtype=torch.int32, device=device
        )
        offset += num_blocks

    k_cache = torch.randn(
        (num_physical_blocks, block_size, shape.nh_k, shape.head_dim),
        dtype=dtype,
        device=device,
        generator=gen,
    ) * 0.5
    v_cache = torch.randn_like(k_cache)
    sk_t = torch.tensor(
        [k_len for _, k_len in shape.seq_lens], dtype=torch.int32, device=device
    )
    cu_q_t = torch.tensor(cu_q, dtype=torch.int32, device=device)
    max_q = max(q_len for q_len, _ in shape.seq_lens)
    max_k = max(k_len for _, k_len in shape.seq_lens)
    scale = 1.0 / math.sqrt(shape.head_dim)

    return (
        q,
        k_cache,
        v_cache,
        max_q,
        cu_q_t,
        max_k,
        None,
        sk_t,
        None,
        0.0,
        scale,
        shape.causal,
        [-1, -1],
        0.0,
        None,
        False,
        False,
        block_table,
        False,
        out,
        None,
        None,
        None,
        None,
        {"fa_version": 3, "use_gluon": True},
    )


def _pr4494_make_input(workload: _Pr4494Workload, dtype: torch.dtype, device: str, seed: int) -> tuple:
    shape = workload.shape
    if isinstance(shape, _Pr4494QwenShape):
        return _pr4494_make_qwen_input(shape, dtype, device, seed)
    if shape.paged:
        return _pr4494_make_hopper_paged_input(shape, dtype, device, seed)
    return _pr4494_make_hopper_dense_input(shape, dtype, device, seed)


def _pr4494_cuda_cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


class FlashAttnVarlenPr707GluonBenchmark(base.Benchmark):
    """PR #707 perf table: 4 paged trace + 21 hopper shapes, vLLM FA3 vs Gems Gluon."""

    DEFAULT_SHAPE_DESC = "suite, name, seq_lens_or_trace, paged"

    def set_shapes(self, shape_file_path: Optional[List[Any]] = None):
        del shape_file_path
        self.workloads = _pr4494_all_workloads()
        self.shapes = [w.shape.name for w in self.workloads]

    def get_input_iter(self, dtype):
        for idx, workload in enumerate(self.workloads):
            self._current_workload = workload
            inp = _pr4494_make_input(workload, dtype, self.device, seed=2026 + idx)
            yield inp

    def record_shapes(self, *args, **kwargs):
        workload = getattr(self, "_current_workload", None)
        if workload is None:
            return super().record_shapes(*args, **kwargs)
        shape = workload.shape
        if isinstance(shape, _Pr4494QwenShape):
            kind = "paged"
        else:
            kind = "paged" if shape.paged else "non_paged"
        detail = super().record_shapes(*args, **kwargs)
        return (shape.name, kind, detail)

    def run(self):
        """Gems first + fresh inputs per path; avoid vLLM→Gems GPU pollution."""
        if Config.query:
            self.init_default_config()
            return

        self.init_user_config()
        gems_op = self.gems_op
        vllm_op = self.torch_op

        for dtype in self.to_bench_dtypes:
            metrics: List[BenchmarkMetrics] = []
            for idx, workload in enumerate(self.workloads):
                self._current_workload = workload
                name = workload.shape.name
                metric = BenchmarkMetrics()
                try:
                    inp_gems = _pr4494_make_input(
                        workload, dtype, self.device, seed=2026 + idx
                    )
                    args_g, kwargs_g = self.unpack_to_args_kwargs(inp_gems)
                    metric.shape_detail = self.record_shapes(*args_g, **kwargs_g)

                    if "latency" in self.to_bench_metrics and gems_op:
                        metric.latency = self.get_latency(
                            gems_op, *args_g, **kwargs_g
                        )
                    _pr4494_cuda_cleanup()

                    if "latency_base" in self.to_bench_metrics:
                        inp_vllm = _pr4494_make_input(
                            workload, dtype, self.device, seed=9026 + idx
                        )
                        args_v, kwargs_v = self.unpack_to_args_kwargs(inp_vllm)
                        metric.latency_base = self.get_latency(
                            vllm_op, *args_v, **kwargs_v
                        )
                    _pr4494_cuda_cleanup()

                    if "speedup" in self.to_bench_metrics:
                        metric.speedup = metric.latency_base / metric.latency
                except (RuntimeError, Exception) as exc:
                    metric.error_msg = str(exc)
                    pytest.fail(
                        f"{workload.suite}/{name} dtype={dtype}: {exc}"
                    )
                finally:
                    metrics.append(metric)
                    _pr4494_cuda_cleanup()

            result = BenchmarkResult(
                level=Config.bench_level.value,
                op_name=self.op_name,
                dtype=str(dtype),
                mode=Config.mode.value,
                result=metrics,
            )
            print(result)
            update_result(self.op_name, asdict(result))
            emit_record_logger(result.to_json())


@pytest.mark.skipif(not _pr4494_is_hopper(), reason="Gluon requires Hopper GPU (sm_90+)")
@pytest.mark.skipif(
    utils.SkipVersion("vllm", "<0.9"),
    reason="vLLM version prior to 0.9 does not include the flash_attn_varlen_func API.",
)
@pytest.mark.skipif(vendor_name == "hygon", reason="Not working")
@pytest.mark.skipif(vendor_name == "cambricon", reason="Not supported")
@pytest.mark.flash_attn_varlen_func
def test_flash_attn_varlen_pr707_gluon_benchmark(monkeypatch):
    """PR #707 perf table: vLLM FA3 baseline vs Gems Gluon (use_gluon=True)."""
    monkeypatch.setenv("VLLM_CONFIGURE_LOGGING", "0")

    from vllm.vllm_flash_attn.flash_attn_interface import flash_attn_varlen_func as _vllm_fa

    def vllm_fa3(*args, **kwargs):
        kwargs.pop("fa_version", None)
        kwargs.pop("use_gluon", None)
        return _vllm_fa(*args, fa_version=3, **kwargs)

    bench = FlashAttnVarlenPr707GluonBenchmark(
        op_name="flash_attn_varlen_pr707_gluon",
        torch_op=vllm_fa3,
        gems_op=flag_gems.ops.flash_attn_varlen_func,
        dtypes=[torch.float16, torch.bfloat16],
    )
    bench.run()
