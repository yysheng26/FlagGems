# """
# Reproduce FlagTree PR #707 performance workloads via FlagGems benchmark.

# PR #707 itself does NOT ship a varlen perf script for the 25 table rows.
# What exists in the PR / branch:

#   FlagTree (compiler + TLE API smoke tests):
#     python/test/tle/integration/test_tle_tma_copy.py
#     python/test/tle/integration/test_tle_ws_tma_gemm.py
#     third_party/tle/tutorials/tle_hopper_fa_ws_pipelined_pingpong_persistent.py
#     third_party/tle/tutorials/test/test_fa.sh   # dense SDPA ZxHxNxD only

#   FlagGems (varlen shapes — source of PR perf table cases):
#     benchmark/test_flash_attn_varlen_fa3_func.py   # paged, block_size=16
#     benchmark/test_flash_attn_varlen_gluon_func.py
#     benchmark/test_flash_attn_varlen_gluon_fa2_shapes.py  # 4 trace × 5 paths

# PR table columns: vllm / torch / fa2 / fa3_ws  (normalized to vllm=100%).
# fa3_ws needs FlagTree #707 built into triton + a FlagGems ws kernel hookup.

# Workload names decode as:
#   tq{total_q} q{max_seqlen_q} k{max_seqlen_k} h{nq} hk{nk} d{head_dim}
#   bs{N} s{seqlen}  kv{len}  gqa4 -> nq=4,nk=4  mha -> nq=16,nk=8
#   bs16 in paged names -> block_size=16 (vLLM default)

# This script runs the same 25 shapes against:
#   vllm_fa3, torch_varlen (if available), gems_fa2, gems_fa3, gems_gluon

# fa3_ws in the PR requires FlagTree #707 + a dedicated ws kernel build; until that
# is wired into FlagGems, gems_fa3/gems_gluon are the closest local proxies.

# Usage (Hopper):
#   pytest -s benchmark/test_flash_attn_varlen_pr707_workloads.py \\
#     -m flash_attn_varlen_func --level core

#   # fp16 only, subset:
#   pytest -s benchmark/test_flash_attn_varlen_pr707_workloads.py \\
#     -m flash_attn_varlen_func --level core --dtypes float16 -k prefill
# """
# from __future__ import annotations

# import gc
# from dataclasses import dataclass
# from typing import Any, Callable, Dict, List, Optional, Tuple

# import pytest
# import torch
# import triton

# import flag_gems

# from . import base, conftest, utils
# from .test_flash_attn_varlen_fa3_func import (
#     FlashAttnVarlenFa3Benchmark,
#     FlashAttnVarlenFa3NonPagedBenchmark,
# )

# vendor_name = flag_gems.vendor_name


# def _is_hopper() -> bool:
#     return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9


# def _cu(lens: Tuple[int, ...]) -> Tuple[int, ...]:
#     r = [0]
#     for l in lens:
#         r.append(r[-1] + l)
#     return tuple(r)


# def _cu_arange(n: int) -> Tuple[int, ...]:
#     return tuple(range(n + 1))


# # ── Qwen3-1.7B trace tensors (shared with fa3_func.py) ─────────────────────

# _TRACE_CU_Q = [
#     (0, 512),
#     (0, 1, 2, 72),
#     tuple(range(0, 45)) + (105, 121, 137, 153, 169, 185, 201, 217, 233, 249, 265),
#     tuple(range(0, 196)) + (211, 226, 240, 253, 265),
# ]
# _TRACE_SK = [
#     (512,),
#     (1, 1, 70),
#     (515,) + (514,) * 20 + (513,) * 20 + (512,) * 14,
#     (2333,)
#     + (2331,) * 20 + (2330,) * 20 + (2329,) * 14
#     + (2328,) * 18 + (2327,) * 15 + (2326,) * 17
#     + (2325,) * 18 + (2324,) * 21 + (2323,) * 22
#     + (2322,) * 24 + (2321,) * 5
#     + (2320, 2319, 2318, 2317, 2316),
# ]

# MHA = (16, 8)
# GQA4 = (4, 4)


# @dataclass(frozen=True)
# class WorkloadSpec:
#     name: str
#     kind: str  # "paged" | "non_paged"
#     # paged: (cu_q, seqused_k, nq, nk, hd, block_size, num_blocks, softcap, window)
#     # non_paged: (cu_q, cu_k_lens_as_seq_lens, nq, nk, hd, softcap, window)
#     config: tuple
#     b2_hint: str = ""


# def _paged_trace(idx: int, name: str, b2_hint: str) -> WorkloadSpec:
#     cu_q, sk = _TRACE_CU_Q[idx], _TRACE_SK[idx]
#     max_kv = max(sk)
#     bs = 16
#     nb = sum((k + bs - 1) // bs for k in sk) + 64
#     return WorkloadSpec(
#         name=name,
#         kind="paged",
#         config=(cu_q, sk, *MHA, 128, bs, nb, None, (-1, -1)),
#         b2_hint=b2_hint,
#     )


# def _non_paged_trace(idx: int, name: str, b2_hint: str) -> WorkloadSpec:
#     cu_q, sk = _TRACE_CU_Q[idx], _TRACE_SK[idx]
#     return WorkloadSpec(
#         name=name,
#         kind="non_paged",
#         config=(cu_q, _cu(sk), *MHA, 128, None, (-1, -1)),
#         b2_hint=b2_hint,
#     )


# def _serve_32_1pf_31dec() -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
#     """32-seq batch: 1×prefill(sq=512) + 31×decode(sq=1)."""
#     cu_q = (0, 512) + tuple(512 + i for i in range(1, 32))
#     sk = (2048,) * 32
#     return cu_q, sk


# def _non_paged_uniform_qk(batch: int, seqlen: int) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
#     return _cu((seqlen,) * batch), _cu((seqlen,) * batch)


# PR707_WORKLOADS: List[WorkloadSpec] = [
#     # ── Qwen trace (paged, block_size=16) ───────────────────────────────────
#     _paged_trace(3, "paged_decodeish_long_k_tq265_q16_k2333_h16_hk8_d128",
#                  "fa3_func paged trace#3 (not in gluon paged b2)"),
#     _paged_trace(0, "paged_medium_or_prefill_tq512_q512_k512_h16_hk8_d128",
#                  "b2 paged L114 / non-paged L13 case0"),
#     _paged_trace(2, "paged_mixed_short_tq265_q61_k515_h16_hk8_d128",
#                  "b2 paged L116 / non-paged L15 case2"),
#     _paged_trace(1, "paged_short_tq72_q70_k70_h16_hk8_d128",
#                  "b2 paged L115 / non-paged L14 case1"),
#     # ── decode uniform (non-paged) ──────────────────────────────────────────
#     WorkloadSpec("decode_b16_kv1k_d128_gqa4", "non_paged",
#                  (_cu_arange(16), _cu((1024,) * 16), *GQA4, 128, None, (-1, -1)),
#                  "custom; closest decode bs=16 in b2 is kv=512/2048"),
#     WorkloadSpec("decode_b16_mixed_d128_gqa4", "non_paged",
#                  (_cu_arange(16), _cu((512,) * 8 + (2048,) * 8), *GQA4, 128, None, (-1, -1)),
#                  "custom mixed kv"),
#     WorkloadSpec("decode_b32_kv2k_d128_gqa4", "non_paged",
#                  (_cu_arange(32), _cu((2048,) * 32), *GQA4, 128, None, (-1, -1)),
#                  "b2 non-paged GQA idx31 nq=4,nk=4 bs=32 kv=2048"),
#     WorkloadSpec("decode_b8_kv1k_d192_gqa4", "non_paged",
#                  (_cu_arange(8), _cu((1024,) * 8), *GQA4, 192, None, (-1, -1)),
#                  "custom"),
#     WorkloadSpec("decode_b8_kv1k_d256_gqa4", "non_paged",
#                  (_cu_arange(8), _cu((1024,) * 8), *GQA4, 256, None, (-1, -1)),
#                  "custom"),
#     # ── decode paged (block_size=16) ────────────────────────────────────────
#     WorkloadSpec("paged_decode_b16_kvmix_bs16_d128_gqa4", "paged",
#                  (_cu_arange(16), (512,) * 8 + (2048,) * 8, *GQA4, 128, 16,
#                   16 * ((2048 + 15) // 16) + 64, None, (-1, -1)),
#                  "custom"),
#     WorkloadSpec("paged_decode_b64_bs16_d128_gqa4", "paged",
#                  (_cu_arange(64), (2048,) * 64, *GQA4, 128, 16,
#                   64 * ((2048 + 15) // 16) + 64, None, (-1, -1)),
#                  "custom"),
#     WorkloadSpec("paged_decode_b8_bs16_d192_gqa4", "paged",
#                  (_cu_arange(8), (1024,) * 8, *GQA4, 192, 16,
#                   8 * ((1024 + 15) // 16) + 64, None, (-1, -1)),
#                  "custom"),
#     WorkloadSpec("paged_decode_b8_bs16_d256_gqa4", "paged",
#                  (_cu_arange(8), (1024,) * 8, *GQA4, 256, 16,
#                   8 * ((1024 + 15) // 16) + 64, None, (-1, -1)),
#                  "custom"),
#     WorkloadSpec("paged_serve_b32_1pf_31dec_bs16_d128_gqa4", "paged",
#                  (*_serve_32_1pf_31dec(), *GQA4, 128, 16,
#                   32 * ((2048 + 15) // 16) + 64, None, (-1, -1)),
#                  "custom serve mix"),
#     WorkloadSpec("paged_uniform_b4_s4k_bs16_d128_mha", "paged",
#                  (_cu((4096,) * 4), (4096,) * 4, *MHA, 128, 16,
#                   4 * ((4096 + 15) // 16) + 64, None, (-1, -1)),
#                  "custom; b2 has bs=4 s=2048 not 4096"),
#     # ── prefill (non-paged) ─────────────────────────────────────────────────
#     WorkloadSpec("prefill_b2_s16k_d128_mha", "non_paged",
#                  (*_non_paged_uniform_qk(2, 16384), *MHA, 128, None, (-1, -1)),
#                  "custom 16k"),
#     WorkloadSpec("prefill_b4_s2k_d128_mha", "non_paged",
#                  (*_non_paged_uniform_qk(4, 2048), *MHA, 128, None, (-1, -1)),
#                  "b2 non-paged idx16"),
#     WorkloadSpec("prefill_b4_s4k_d128_gqa4", "non_paged",
#                  (*_non_paged_uniform_qk(4, 4096), *GQA4, 128, None, (-1, -1)),
#                  "custom"),
#     WorkloadSpec("prefill_b4_s4k_d128_mha", "non_paged",
#                  (*_non_paged_uniform_qk(4, 4096), *MHA, 128, None, (-1, -1)),
#                  "custom"),
#     WorkloadSpec("prefill_b4_s8k_d128_gqa4", "non_paged",
#                  (*_non_paged_uniform_qk(4, 8192), *GQA4, 128, None, (-1, -1)),
#                  "custom"),
#     WorkloadSpec("prefill_b4_s8k_d128_mha", "non_paged",
#                  (*_non_paged_uniform_qk(4, 8192), *MHA, 128, None, (-1, -1)),
#                  "custom"),
#     WorkloadSpec("prefill_b8_s2k_d64_mha", "non_paged",
#                  (*_non_paged_uniform_qk(8, 2048), *MHA, 64, None, (-1, -1)),
#                  "custom"),
#     # ── varlen trace (non-paged) ────────────────────────────────────────────
#     _non_paged_trace(3, "varlen_longtail_d128_gqa4",
#                      "b2 non-paged L16 case3 (fix vLLM baseline ~6ms not 1.17ms)"),
#     _non_paged_trace(2, "varlen_mixed_d128_gqa4",
#                      "b2 non-paged L15 case2"),
#     WorkloadSpec(
#         "varlen_serve_b32_1pf_31dec_d128_gqa4",
#         "non_paged",
#         (
#             _serve_32_1pf_31dec()[0],
#             _cu(_serve_32_1pf_31dec()[1]),
#             *GQA4,
#             128,
#             None,
#             (-1, -1),
#         ),
#         "custom serve mix non-paged",
#     ),
# ]


# def _unpack(inp: tuple) -> Tuple[list, dict]:
#     bench = base.Benchmark(op_name="unpack", torch_op=lambda: None)
#     return bench.unpack_to_args_kwargs(inp)


# def _with_fa_version(inp: tuple, fa_version: int) -> tuple:
#     args, kwargs = _unpack(inp)
#     kwargs = dict(kwargs)
#     kwargs["fa_version"] = fa_version
#     return tuple(args) + (kwargs,)


# def _seqused_k_to_cu_seqlens_k(seqused_k: torch.Tensor) -> torch.Tensor:
#     return torch.cat(
#         [
#             torch.zeros(1, dtype=torch.int32, device=seqused_k.device),
#             torch.cumsum(seqused_k, dim=0),
#         ]
#     )


# def _kv_num_heads(key: torch.Tensor, block_table: Optional[torch.Tensor]) -> int:
#     return key.size(2) if block_table is not None else key.size(1)


# def _flash_window_size_for_torch(causal: bool, window_size) -> Tuple[int, int]:
#     ws = tuple(window_size) if window_size is not None else (-1, -1)
#     if causal and ws == (-1, -1):
#         return (-1, 0)
#     return ws


# def _torch_varlen_from_flash(*args, **kwargs):
#     """Map flash_attn_varlen_func args to torch.nn.attention.varlen.varlen_attn."""
#     from torch.nn.attention.varlen import varlen_attn

#     kwargs.pop("fa_version", None)
#     q, k, v = args[0], args[1], args[2]
#     max_q, cu_q = args[3], args[4]
#     max_k = args[5]
#     cu_k = args[6]
#     seqused_k = args[7]
#     scale = args[10]
#     causal = args[11]
#     window_size = _flash_window_size_for_torch(causal, args[12])
#     block_table = args[17]

#     if cu_k is None and seqused_k is not None and block_table is None:
#         cu_k = _seqused_k_to_cu_seqlens_k(seqused_k)

#     enable_gqa = _kv_num_heads(k, block_table) != q.size(1)
#     return varlen_attn(
#         q,
#         k,
#         v,
#         cu_q,
#         cu_k,
#         max_q,
#         max_k,
#         scale=scale,
#         window_size=window_size,
#         enable_gqa=enable_gqa,
#         seqused_k=seqused_k if block_table is not None else None,
#         block_table=block_table,
#     )


# def _to_gluon_input(fa3_paged_inp: tuple) -> tuple:
#     args, kwargs = _unpack(fa3_paged_inp)
#     args = list(args)
#     seqused_k = args[7]
#     args[6] = _seqused_k_to_cu_seqlens_k(seqused_k)
#     args[7] = None
#     kwargs = dict(kwargs)
#     kwargs["fa_version"] = 3
#     return tuple(args) + (kwargs,)


# def _measure_latency(op: Callable, inp: tuple) -> float:
#     args, kwargs = _unpack(inp)
#     fn = lambda: op(*args, **kwargs)
#     return triton.testing.do_bench(
#         fn,
#         warmup=conftest.Config.warm_up,
#         rep=conftest.Config.repetition,
#         return_mode="median",
#     )


# def _fmt_ms(x: float) -> str:
#     return f"{x:.6f}"


# def _fmt_ratio(base: float, lat: float) -> str:
#     if lat <= 0:
#         return "n/a"
#     return f"{base / lat:.3f}"


# class Pr707WorkloadBenchmark(base.Benchmark):
#     """Run PR #707 table workloads and print a comparison table."""

#     def set_shapes(self, shape_file_path=None):
#         self.workloads = PR707_WORKLOADS
#         self._paged_bench = FlashAttnVarlenFa3Benchmark(
#             op_name="paged_builder", torch_op=lambda: None, gems_op=None,
#         )
#         self._non_paged_bench = FlashAttnVarlenFa3NonPagedBenchmark(
#             op_name="non_paged_builder", torch_op=lambda: None, gems_op=None,
#         )
#         self._paged_bench.set_shapes()
#         self._non_paged_bench.set_shapes()

#     def _build_input(self, spec: WorkloadSpec, dtype: torch.dtype) -> tuple:
#         if spec.kind == "paged":
#             return self._paged_bench._make_input(spec.config, dtype, self.device)
#         return self._non_paged_bench._make_input(spec.config, dtype, self.device)

#     def run(self):
#         if conftest.Config.query:
#             self.init_default_config()
#             return

#         self.init_user_config()
#         gems_op = flag_gems.ops.flash_attn_varlen_func

#         from vllm.vllm_flash_attn.flash_attn_interface import (
#             flash_attn_varlen_func as _vllm_fa,
#         )

#         def vllm_fa2(*args, **kwargs):
#             kwargs.pop("fa_version", None)
#             return _vllm_fa(*args, fa_version=2, **kwargs)

#         def vllm_fa3(*args, **kwargs):
#             kwargs.pop("fa_version", None)
#             return _vllm_fa(*args, fa_version=3, **kwargs)

#         def gems_fa2(*args, **kwargs):
#             kwargs["fa_version"] = 2
#             return gems_op(*args, **kwargs)

#         def gems_fa3(*args, **kwargs):
#             kwargs["fa_version"] = 3
#             return gems_op(*args, **kwargs)

#         torch_varlen_fn: Optional[Callable] = None
#         try:
#             from torch.nn.attention.varlen import varlen_attn  # noqa: F401

#             torch_varlen_fn = _torch_varlen_from_flash
#         except ImportError:
#             pass

#         for dtype in self.to_bench_dtypes:
#             print(
#                 f"\n{'=' * 130}\n"
#                 f"PR #707 workloads  dtype={dtype}  "
#                 f"mode={conftest.Config.mode.value}  level={conftest.Config.bench_level.value}\n"
#                 f"{'=' * 130}"
#             )
#             hdr = (
#                 f"{'workload':<52}"
#                 f"{'vllm':>10}{'torch':>10}{'fa2':>10}{'fa3':>10}{'gluon':>10} | "
#                 f"{'t_x':>6}{'f2_x':>6}{'f3_x':>6}{'gl_x':>6}"
#             )
#             print(hdr)
#             print("-" * len(hdr))

#             for spec in self.workloads:
#                 inp = self._build_input(spec, dtype)
#                 if inp is None:
#                     print(f"{spec.name:<52} SKIP (input build failed)")
#                     continue

#                 fa3_inp = _with_fa_version(inp, fa_version=3)
#                 fa2_inp = _with_fa_version(inp, fa_version=2)
#                 gluon_inp = (
#                     _to_gluon_input(inp) if spec.kind == "paged" else fa3_inp
#                 )

#                 lat: Dict[str, float] = {}
#                 try:
#                     lat["vllm"] = _measure_latency(vllm_fa3, fa3_inp)
#                     if torch_varlen_fn is not None:
#                         try:
#                             lat["torch"] = _measure_latency(torch_varlen_fn, fa3_inp)
#                         except (RuntimeError, TypeError, ValueError) as e:
#                             print(f"  torch skip ({spec.name}): {e}")
#                     lat["fa2"] = _measure_latency(gems_fa2, fa2_inp)
#                     lat["fa3"] = _measure_latency(gems_fa3, fa3_inp)
#                     lat["gluon"] = _measure_latency(gems_fa3, gluon_inp)
#                 except (RuntimeError, Exception) as e:
#                     pytest.fail(f"{spec.name} failed: {e}")
#                 finally:
#                     gc.collect()

#                 base_ms = lat["vllm"]
#                 torch_ms = lat.get("torch", float("nan"))
#                 print(
#                     f"{spec.name:<52}"
#                     f"{_fmt_ms(lat['vllm']):>10}"
#                     f"{_fmt_ms(torch_ms):>10}"
#                     f"{_fmt_ms(lat['fa2']):>10}"
#                     f"{_fmt_ms(lat['fa3']):>10}"
#                     f"{_fmt_ms(lat['gluon']):>10} | "
#                     f"{_fmt_ratio(base_ms, torch_ms):>6}"
#                     f"{_fmt_ratio(base_ms, lat['fa2']):>6}"
#                     f"{_fmt_ratio(base_ms, lat['fa3']):>6}"
#                     f"{_fmt_ratio(base_ms, lat['gluon']):>6}"
#                 )
#                 if spec.b2_hint:
#                     print(f"  hint: {spec.b2_hint}")

#             print(
#                 "\nColumns mirror PR #707: vllm=100% baseline; *_x = vllm/latency.\n"
#                 "  fa3  = FlagGems FA3 Triton (seqused_k paged / cu_seqlens_k non-paged)\n"
#                 "  gluon= cu_seqlens_k + block_table (paged) or same as fa3 (non-paged)\n"
#                 "  PR fa3_ws needs FlagTree #707 ws kernel — not measured here unless wired.\n"
#                 "  b2.txt line hints assume test_flash_attn_varlen_gluon_func comprehensive order."
#             )


# @pytest.mark.skipif(not _is_hopper(), reason="FA3 requires Hopper GPU (sm_90+)")
# @pytest.mark.skipif(
#     utils.SkipVersion("vllm", "<0.9"),
#     reason="vLLM < 0.9 does not expose flash_attn_varlen_func",
# )
# @pytest.mark.skipif(vendor_name in ("hygon", "cambricon"), reason="Not working")
# @pytest.mark.flash_attn_varlen_func
# def test_flash_attn_varlen_pr707_workloads(monkeypatch):
#     monkeypatch.setenv("VLLM_CONFIGURE_LOGGING", "0")

#     bench = Pr707WorkloadBenchmark(
#         op_name="flash_attn_varlen_pr707_workloads",
#         torch_op=None,
#         gems_op=flag_gems.ops.flash_attn_varlen_func,
#         dtypes=[torch.float16, torch.bfloat16],
#     )
#     bench.run()

# """
# PR #707 performance workloads — standard FlagGems benchmark.

# Runs the same 25 shapes as the commented table script above:
#   baseline = vLLM flash_attn_varlen_func (fa_version=3)
#   gems     = FlagGems flash_attn_varlen_func (fa_version=3, use_gluon=False)

# Usage (Hopper):
#   pytest -s benchmark/test_flash_attn_varlen_pr707_workloads.py \\
#     -m flash_attn_varlen_func --level core

#   pytest -s benchmark/test_flash_attn_varlen_pr707_workloads.py \\
#     -m flash_attn_varlen_func --level core --dtypes float16
# """
# from __future__ import annotations

# from dataclasses import dataclass
# from typing import Any, Generator, List, Optional, Tuple

# import pytest
# import torch

# import flag_gems

# from . import base, utils
# from .test_flash_attn_varlen_fa3_func import (
#     FlashAttnVarlenFa3Benchmark,
#     FlashAttnVarlenFa3NonPagedBenchmark,
# )

# vendor_name = flag_gems.vendor_name


# def _is_hopper() -> bool:
#     return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9


# def _cu(lens: Tuple[int, ...]) -> Tuple[int, ...]:
#     r = [0]
#     for length in lens:
#         r.append(r[-1] + length)
#     return tuple(r)


# def _cu_arange(n: int) -> Tuple[int, ...]:
#     return tuple(range(n + 1))


# _TRACE_CU_Q = [
#     (0, 512),
#     (0, 1, 2, 72),
#     tuple(range(0, 45)) + (105, 121, 137, 153, 169, 185, 201, 217, 233, 249, 265),
#     tuple(range(0, 196)) + (211, 226, 240, 253, 265),
# ]
# _TRACE_SK = [
#     (512,),
#     (1, 1, 70),
#     (515,) + (514,) * 20 + (513,) * 20 + (512,) * 14,
#     (2333,)
#     + (2331,) * 20 + (2330,) * 20 + (2329,) * 14
#     + (2328,) * 18 + (2327,) * 15 + (2326,) * 17
#     + (2325,) * 18 + (2324,) * 21 + (2323,) * 22
#     + (2322,) * 24 + (2321,) * 5
#     + (2320, 2319, 2318, 2317, 2316),
# ]

# MHA = (16, 8)
# GQA4 = (4, 4)


# @dataclass(frozen=True)
# class WorkloadSpec:
#     name: str
#     kind: str  # "paged" | "non_paged"
#     config: tuple
#     b2_hint: str = ""


# def _paged_trace(idx: int, name: str, b2_hint: str) -> WorkloadSpec:
#     cu_q, sk = _TRACE_CU_Q[idx], _TRACE_SK[idx]
#     bs = 16
#     nb = sum((k + bs - 1) // bs for k in sk) + 64
#     return WorkloadSpec(
#         name=name,
#         kind="paged",
#         config=(cu_q, sk, *MHA, 128, bs, nb, None, (-1, -1)),
#         b2_hint=b2_hint,
#     )


# def _non_paged_trace(idx: int, name: str, b2_hint: str) -> WorkloadSpec:
#     cu_q, sk = _TRACE_CU_Q[idx], _TRACE_SK[idx]
#     return WorkloadSpec(
#         name=name,
#         kind="non_paged",
#         config=(cu_q, _cu(sk), *MHA, 128, None, (-1, -1)),
#         b2_hint=b2_hint,
#     )


# def _serve_32_1pf_31dec() -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
#     """32-seq batch: 1×prefill(sq=512) + 31×decode(sq=1)."""
#     cu_q = (0, 512) + tuple(512 + i for i in range(1, 32))
#     sk = (2048,) * 32
#     return cu_q, sk


# def _non_paged_uniform_qk(batch: int, seqlen: int) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
#     return _cu((seqlen,) * batch), _cu((seqlen,) * batch)


# PR707_WORKLOADS: List[WorkloadSpec] = [
#     # ── Qwen trace (paged, block_size=16) ───────────────────────────────────
#     _paged_trace(3, "paged_decodeish_long_k_tq265_q16_k2333_h16_hk8_d128",
#                  "fa3_func paged trace#3 (not in gluon paged b2)"),
#     _paged_trace(0, "paged_medium_or_prefill_tq512_q512_k512_h16_hk8_d128",
#                  "b2 paged L114 / non-paged L13 case0"),
#     _paged_trace(2, "paged_mixed_short_tq265_q61_k515_h16_hk8_d128",
#                  "b2 paged L116 / non-paged L15 case2"),
#     _paged_trace(1, "paged_short_tq72_q70_k70_h16_hk8_d128",
#                  "b2 paged L115 / non-paged L14 case1"),
#     # ── decode uniform (non-paged) ──────────────────────────────────────────
#     WorkloadSpec("decode_b16_kv1k_d128_gqa4", "non_paged",
#                  (_cu_arange(16), _cu((1024,) * 16), *GQA4, 128, None, (-1, -1)),
#                  "custom; closest decode bs=16 in b2 is kv=512/2048"),
#     WorkloadSpec("decode_b16_mixed_d128_gqa4", "non_paged",
#                  (_cu_arange(16), _cu((512,) * 8 + (2048,) * 8), *GQA4, 128, None, (-1, -1)),
#                  "custom mixed kv"),
#     WorkloadSpec("decode_b32_kv2k_d128_gqa4", "non_paged",
#                  (_cu_arange(32), _cu((2048,) * 32), *GQA4, 128, None, (-1, -1)),
#                  "b2 non-paged GQA idx31 nq=4,nk=4 bs=32 kv=2048"),
#     WorkloadSpec("decode_b8_kv1k_d192_gqa4", "non_paged",
#                  (_cu_arange(8), _cu((1024,) * 8), *GQA4, 192, None, (-1, -1)),
#                  "custom"),
#     WorkloadSpec("decode_b8_kv1k_d256_gqa4", "non_paged",
#                  (_cu_arange(8), _cu((1024,) * 8), *GQA4, 256, None, (-1, -1)),
#                  "custom"),
#     # ── decode paged (block_size=16) ────────────────────────────────────────
#     WorkloadSpec("paged_decode_b16_kvmix_bs16_d128_gqa4", "paged",
#                  (_cu_arange(16), (512,) * 8 + (2048,) * 8, *GQA4, 128, 16,
#                   16 * ((2048 + 15) // 16) + 64, None, (-1, -1)),
#                  "custom"),
#     WorkloadSpec("paged_decode_b64_bs16_d128_gqa4", "paged",
#                  (_cu_arange(64), (2048,) * 64, *GQA4, 128, 16,
#                   64 * ((2048 + 15) // 16) + 64, None, (-1, -1)),
#                  "custom"),
#     WorkloadSpec("paged_decode_b8_bs16_d192_gqa4", "paged",
#                  (_cu_arange(8), (1024,) * 8, *GQA4, 192, 16,
#                   8 * ((1024 + 15) // 16) + 64, None, (-1, -1)),
#                  "custom"),
#     WorkloadSpec("paged_decode_b8_bs16_d256_gqa4", "paged",
#                  (_cu_arange(8), (1024,) * 8, *GQA4, 256, 16,
#                   8 * ((1024 + 15) // 16) + 64, None, (-1, -1)),
#                  "custom"),
#     WorkloadSpec("paged_serve_b32_1pf_31dec_bs16_d128_gqa4", "paged",
#                  (*_serve_32_1pf_31dec(), *GQA4, 128, 16,
#                   32 * ((2048 + 15) // 16) + 64, None, (-1, -1)),
#                  "custom serve mix"),
#     WorkloadSpec("paged_uniform_b4_s4k_bs16_d128_mha", "paged",
#                  (_cu((4096,) * 4), (4096,) * 4, *MHA, 128, 16,
#                   4 * ((4096 + 15) // 16) + 64, None, (-1, -1)),
#                  "custom; b2 has bs=4 s=2048 not 4096"),
#     # ── prefill (non-paged) ─────────────────────────────────────────────────
#     WorkloadSpec("prefill_b2_s16k_d128_mha", "non_paged",
#                  (*_non_paged_uniform_qk(2, 16384), *MHA, 128, None, (-1, -1)),
#                  "custom 16k"),
#     WorkloadSpec("prefill_b4_s2k_d128_mha", "non_paged",
#                  (*_non_paged_uniform_qk(4, 2048), *MHA, 128, None, (-1, -1)),
#                  "b2 non-paged idx16"),
#     WorkloadSpec("prefill_b4_s4k_d128_gqa4", "non_paged",
#                  (*_non_paged_uniform_qk(4, 4096), *GQA4, 128, None, (-1, -1)),
#                  "custom"),
#     WorkloadSpec("prefill_b4_s4k_d128_mha", "non_paged",
#                  (*_non_paged_uniform_qk(4, 4096), *MHA, 128, None, (-1, -1)),
#                  "custom"),
#     WorkloadSpec("prefill_b4_s8k_d128_gqa4", "non_paged",
#                  (*_non_paged_uniform_qk(4, 8192), *GQA4, 128, None, (-1, -1)),
#                  "custom"),
#     WorkloadSpec("prefill_b4_s8k_d128_mha", "non_paged",
#                  (*_non_paged_uniform_qk(4, 8192), *MHA, 128, None, (-1, -1)),
#                  "custom"),
#     WorkloadSpec("prefill_b8_s2k_d64_mha", "non_paged",
#                  (*_non_paged_uniform_qk(8, 2048), *MHA, 64, None, (-1, -1)),
#                  "custom"),
#     # ── varlen trace (non-paged) ────────────────────────────────────────────
#     _non_paged_trace(3, "varlen_longtail_d128_gqa4",
#                      "b2 non-paged L16 case3 (fix vLLM baseline ~6ms not 1.17ms)"),
#     _non_paged_trace(2, "varlen_mixed_d128_gqa4",
#                      "b2 non-paged L15 case2"),
#     WorkloadSpec(
#         "varlen_serve_b32_1pf_31dec_d128_gqa4",
#         "non_paged",
#         (
#             _serve_32_1pf_31dec()[0],
#             _cu(_serve_32_1pf_31dec()[1]),
#             *GQA4,
#             128,
#             None,
#             (-1, -1),
#         ),
#         "custom serve mix non-paged",
#     ),
# ]


# class FlashAttnVarlenPr707Fa3Benchmark(base.Benchmark):
#     """PR #707 table workloads: vLLM FA3 vs FlagGems Triton FA3 (not Gluon)."""

#     DEFAULT_SHAPE_DESC = "workload, kind, q/k shapes"

#     def set_shapes(self, shape_file_path: Optional[List[Any]] = None):
#         del shape_file_path  # PR #707 shapes are fixed; ignore yaml shape files.
#         self.workloads = PR707_WORKLOADS
#         self.shapes = [spec.name for spec in self.workloads]
#         self.shape_desc = self.DEFAULT_SHAPE_DESC
#         self._paged_builder = FlashAttnVarlenFa3Benchmark(
#             op_name="pr707_paged_builder",
#             torch_op=lambda: None,
#             gems_op=None,
#         )
#         self._non_paged_builder = FlashAttnVarlenFa3NonPagedBenchmark(
#             op_name="pr707_non_paged_builder",
#             torch_op=lambda: None,
#             gems_op=None,
#         )

#     def get_input_iter(self, dtype) -> Generator[tuple, None, None]:
#         for spec in self.workloads:
#             self._current_workload = spec.name
#             self._current_kind = spec.kind
#             if spec.kind == "paged":
#                 inp = self._paged_builder._make_input(spec.config, dtype, self.device)
#             else:
#                 inp = self._non_paged_builder._make_input(spec.config, dtype, self.device)
#             if inp is not None:
#                 yield inp

#     def record_shapes(self, *args, **kwargs):
#         detail = super().record_shapes(*args, **kwargs)
#         return (
#             getattr(self, "_current_workload", ""),
#             getattr(self, "_current_kind", ""),
#             detail,
#         )


# @pytest.mark.skipif(
#     not _is_hopper(),
#     reason="FA3 requires Hopper GPU (sm_90+)",
# )
# @pytest.mark.skipif(
#     utils.SkipVersion("vllm", "<0.9"),
#     reason="vLLM version prior to 0.9 does not include the flash_attn_varlen_func API.",
# )
# @pytest.mark.skipif(vendor_name == "hygon", reason="Not working")
# @pytest.mark.skipif(vendor_name == "cambricon", reason="Not supported")
# @pytest.mark.flash_attn_varlen_func
# def test_flash_attn_varlen_pr707_workloads(monkeypatch):
#     monkeypatch.setenv("VLLM_CONFIGURE_LOGGING", "0")

#     from vllm.vllm_flash_attn.flash_attn_interface import flash_attn_varlen_func as _vllm_fa

#     def vllm_fa3(*args, **kwargs):
#         kwargs.pop("fa_version", None)
#         kwargs.pop("use_gluon", None)
#         return _vllm_fa(*args, fa_version=3, **kwargs)

#     bench = FlashAttnVarlenPr707Fa3Benchmark(
#         op_name="flash_attn_varlen_pr707_workloads",
#         torch_op=vllm_fa3,
#         gems_op=flag_gems.ops.flash_attn_varlen_func,
#         dtypes=[torch.float16, torch.bfloat16],
#     )
#     bench.run()

"""
PR #707 workloads — Gluon FA3 varlen benchmark.

  baseline = vLLM flash_attn_varlen_func (fa_version=3)   # 与上方 Triton 版完全相同
  gems     = FlagGems flash_attn_varlen_func (fa_version=3, use_gluon=True)

Usage (Hopper):
  pytest -s benchmark/test_flash_attn_varlen_pr707_workloads.py \\
    -m flash_attn_varlen_func --level core -k gluon
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generator, List, Optional, Tuple

import pytest
import torch

import flag_gems

from . import base, utils
from .test_flash_attn_varlen_fa3_func import (
    FlashAttnVarlenFa3Benchmark,
    FlashAttnVarlenFa3NonPagedBenchmark,
)

vendor_name = flag_gems.vendor_name


def _is_hopper() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9


def _cu(lens: Tuple[int, ...]) -> Tuple[int, ...]:
    r = [0]
    for length in lens:
        r.append(r[-1] + length)
    return tuple(r)


def _cu_arange(n: int) -> Tuple[int, ...]:
    return tuple(range(n + 1))


_TRACE_CU_Q = [
    (0, 512),
    (0, 1, 2, 72),
    tuple(range(0, 45)) + (105, 121, 137, 153, 169, 185, 201, 217, 233, 249, 265),
    tuple(range(0, 196)) + (211, 226, 240, 253, 265),
]
_TRACE_SK = [
    (512,),
    (1, 1, 70),
    (515,) + (514,) * 20 + (513,) * 20 + (512,) * 14,
    (2333,)
    + (2331,) * 20 + (2330,) * 20 + (2329,) * 14
    + (2328,) * 18 + (2327,) * 15 + (2326,) * 17
    + (2325,) * 18 + (2324,) * 21 + (2323,) * 22
    + (2322,) * 24 + (2321,) * 5
    + (2320, 2319, 2318, 2317, 2316),
]

MHA = (16, 8)
GQA4 = (4, 4)


@dataclass(frozen=True)
class WorkloadSpec:
    name: str
    kind: str  # "paged" | "non_paged"
    config: tuple
    b2_hint: str = ""


def _paged_trace(idx: int, name: str, b2_hint: str) -> WorkloadSpec:
    cu_q, sk = _TRACE_CU_Q[idx], _TRACE_SK[idx]
    bs = 16
    nb = sum((k + bs - 1) // bs for k in sk) + 64
    return WorkloadSpec(
        name=name,
        kind="paged",
        config=(cu_q, sk, *MHA, 128, bs, nb, None, (-1, -1)),
        b2_hint=b2_hint,
    )


def _non_paged_trace(idx: int, name: str, b2_hint: str) -> WorkloadSpec:
    cu_q, sk = _TRACE_CU_Q[idx], _TRACE_SK[idx]
    return WorkloadSpec(
        name=name,
        kind="non_paged",
        config=(cu_q, _cu(sk), *MHA, 128, None, (-1, -1)),
        b2_hint=b2_hint,
    )


def _serve_32_1pf_31dec() -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    cu_q = (0, 512) + tuple(512 + i for i in range(1, 32))
    sk = (2048,) * 32
    return cu_q, sk


def _non_paged_uniform_qk(batch: int, seqlen: int) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    return _cu((seqlen,) * batch), _cu((seqlen,) * batch)


PR707_WORKLOADS: List[WorkloadSpec] = [
    _paged_trace(3, "paged_decodeish_long_k_tq265_q16_k2333_h16_hk8_d128", ""),
    _paged_trace(0, "paged_medium_or_prefill_tq512_q512_k512_h16_hk8_d128", ""),
    _paged_trace(2, "paged_mixed_short_tq265_q61_k515_h16_hk8_d128", ""),
    _paged_trace(1, "paged_short_tq72_q70_k70_h16_hk8_d128", ""),
    WorkloadSpec("decode_b16_kv1k_d128_gqa4", "non_paged",
                 (_cu_arange(16), _cu((1024,) * 16), *GQA4, 128, None, (-1, -1)), ""),
    WorkloadSpec("decode_b16_mixed_d128_gqa4", "non_paged",
                 (_cu_arange(16), _cu((512,) * 8 + (2048,) * 8), *GQA4, 128, None, (-1, -1)), ""),
    WorkloadSpec("decode_b32_kv2k_d128_gqa4", "non_paged",
                 (_cu_arange(32), _cu((2048,) * 32), *GQA4, 128, None, (-1, -1)), ""),
    WorkloadSpec("decode_b8_kv1k_d192_gqa4", "non_paged",
                 (_cu_arange(8), _cu((1024,) * 8), *GQA4, 192, None, (-1, -1)), ""),
    WorkloadSpec("decode_b8_kv1k_d256_gqa4", "non_paged",
                 (_cu_arange(8), _cu((1024,) * 8), *GQA4, 256, None, (-1, -1)), ""),
    WorkloadSpec("paged_decode_b16_kvmix_bs16_d128_gqa4", "paged",
                 (_cu_arange(16), (512,) * 8 + (2048,) * 8, *GQA4, 128, 16,
                  16 * ((2048 + 15) // 16) + 64, None, (-1, -1)), ""),
    WorkloadSpec("paged_decode_b64_bs16_d128_gqa4", "paged",
                 (_cu_arange(64), (2048,) * 64, *GQA4, 128, 16,
                  64 * ((2048 + 15) // 16) + 64, None, (-1, -1)), ""),
    WorkloadSpec("paged_decode_b8_bs16_d192_gqa4", "paged",
                 (_cu_arange(8), (1024,) * 8, *GQA4, 192, 16,
                  8 * ((1024 + 15) // 16) + 64, None, (-1, -1)), ""),
    WorkloadSpec("paged_decode_b8_bs16_d256_gqa4", "paged",
                 (_cu_arange(8), (1024,) * 8, *GQA4, 256, 16,
                  8 * ((1024 + 15) // 16) + 64, None, (-1, -1)), ""),
    WorkloadSpec("paged_serve_b32_1pf_31dec_bs16_d128_gqa4", "paged",
                 (*_serve_32_1pf_31dec(), *GQA4, 128, 16,
                  32 * ((2048 + 15) // 16) + 64, None, (-1, -1)), ""),
    WorkloadSpec("paged_uniform_b4_s4k_bs16_d128_mha", "paged",
                 (_cu((4096,) * 4), (4096,) * 4, *MHA, 128, 16,
                  4 * ((4096 + 15) // 16) + 64, None, (-1, -1)), ""),
    WorkloadSpec("prefill_b2_s16k_d128_mha", "non_paged",
                 (*_non_paged_uniform_qk(2, 16384), *MHA, 128, None, (-1, -1)), ""),
    WorkloadSpec("prefill_b4_s2k_d128_mha", "non_paged",
                 (*_non_paged_uniform_qk(4, 2048), *MHA, 128, None, (-1, -1)), ""),
    WorkloadSpec("prefill_b4_s4k_d128_gqa4", "non_paged",
                 (*_non_paged_uniform_qk(4, 4096), *GQA4, 128, None, (-1, -1)), ""),
    WorkloadSpec("prefill_b4_s4k_d128_mha", "non_paged",
                 (*_non_paged_uniform_qk(4, 4096), *MHA, 128, None, (-1, -1)), ""),
    WorkloadSpec("prefill_b4_s8k_d128_gqa4", "non_paged",
                 (*_non_paged_uniform_qk(4, 8192), *GQA4, 128, None, (-1, -1)), ""),
    WorkloadSpec("prefill_b4_s8k_d128_mha", "non_paged",
                 (*_non_paged_uniform_qk(4, 8192), *MHA, 128, None, (-1, -1)), ""),
    WorkloadSpec("prefill_b8_s2k_d64_mha", "non_paged",
                 (*_non_paged_uniform_qk(8, 2048), *MHA, 64, None, (-1, -1)), ""),
    _non_paged_trace(3, "varlen_longtail_d128_gqa4", ""),
    _non_paged_trace(2, "varlen_mixed_d128_gqa4", ""),
    WorkloadSpec(
        "varlen_serve_b32_1pf_31dec_d128_gqa4",
        "non_paged",
        (_serve_32_1pf_31dec()[0], _cu(_serve_32_1pf_31dec()[1]), *GQA4, 128, None, (-1, -1)),
        "",
    ),
]


def _gems_gluon_kwargs(kwargs: dict) -> dict:
    """Only Gems path differs; baseline still calls the same vLLM FA3 varlen."""
    out = dict(kwargs)
    out["fa_version"] = 3
    out["use_gluon"] = True
    return out


class FlashAttnVarlenPr707GluonBenchmark(base.Benchmark):
    """PR #707 workloads: vLLM FA3 varlen baseline vs FlagGems Gluon FA3 varlen."""

    DEFAULT_SHAPE_DESC = "workload, kind, q/k shapes"

    def set_shapes(self, shape_file_path: Optional[List[Any]] = None):
        del shape_file_path
        self.workloads = PR707_WORKLOADS
        self.shapes = [spec.name for spec in self.workloads]
        self.shape_desc = self.DEFAULT_SHAPE_DESC
        self._paged_builder = FlashAttnVarlenFa3Benchmark(
            op_name="pr707_paged_builder", torch_op=lambda: None, gems_op=None,
        )
        self._non_paged_builder = FlashAttnVarlenFa3NonPagedBenchmark(
            op_name="pr707_non_paged_builder", torch_op=lambda: None, gems_op=None,
        )

    def get_input_iter(self, dtype) -> Generator[tuple, None, None]:
        for spec in self.workloads:
            self._current_workload = spec.name
            self._current_kind = spec.kind
            if spec.kind == "paged":
                inp = self._paged_builder._make_input(spec.config, dtype, self.device)
            else:
                inp = self._non_paged_builder._make_input(spec.config, dtype, self.device)
            if inp is not None:
                # Same positional args / tensor layout as Triton FA3 benchmark;
                # kwargs only tags Gems as Gluon (vllm_fa3 strips use_gluon).
                args, kwargs = self.unpack_to_args_kwargs(inp)
                yield tuple(args) + (_gems_gluon_kwargs(kwargs),)

    def record_shapes(self, *args, **kwargs):
        detail = super().record_shapes(*args, **kwargs)
        return (
            getattr(self, "_current_workload", ""),
            getattr(self, "_current_kind", ""),
            detail,
        )


@pytest.mark.skipif(not _is_hopper(), reason="Gluon FA3 requires Hopper GPU (sm_90+)")
@pytest.mark.skipif(
    utils.SkipVersion("vllm", "<0.9"),
    reason="vLLM version prior to 0.9 does not include the flash_attn_varlen_func API.",
)
@pytest.mark.skipif(vendor_name == "hygon", reason="Not working")
@pytest.mark.skipif(vendor_name == "cambricon", reason="Not supported")
@pytest.mark.flash_attn_varlen_func
def test_flash_attn_varlen_pr707_gluon_workloads(monkeypatch):
    monkeypatch.setenv("VLLM_CONFIGURE_LOGGING", "0")

    from vllm.vllm_flash_attn.flash_attn_interface import flash_attn_varlen_func as _vllm_fa

    # 与注释里 test_flash_attn_varlen_pr707_workloads 的 baseline 完全一致
    def vllm_fa3(*args, **kwargs):
        kwargs.pop("fa_version", None)
        kwargs.pop("use_gluon", None)
        return _vllm_fa(*args, fa_version=3, **kwargs)

    bench = FlashAttnVarlenPr707GluonBenchmark(
        op_name="flash_attn_varlen_pr707_gluon_workloads",
        torch_op=vllm_fa3,
        gems_op=flag_gems.ops.flash_attn_varlen_func,
        dtypes=[torch.float16, torch.bfloat16],
    )
    bench.run()