"""
Five-way flash_attn_varlen quick benchmark (14 cases).

Shapes mirror test_flash_attn_varlen_fa3_func.py quick suites:
  non-paged: 4 × Qwen3-1.7B trace (cu_seqlens_k, flat K/V)
  paged:     3 × Qwen3-1.7B trace (seqused_k + block_table, block_size=64)
  total 7 shapes × 2 dtypes = 14 cases

Paths (5 latency columns):
  vLLM FA2 / vLLM FA3 / Gems FA2 / Gems FA3 Triton / Gems FA3 Gluon

Speedup columns (baseline / gems, >1 means Gems faster):
  vllm_fa2 / gems_fa2
  vllm_fa3 / gems_fa3_triton
  vllm_fa3 / gems_fa3_gluon

Usage:
  USE_C_EXTENSION=1 TORCH_USE_RTLD_GLOBAL=1 CUDA_VISIBLE_DEVICES=2 \\
  FLAGGEMS_SOURCE_DIR=/root/workspace/FlagGems/src/flag_gems \\
  PYTHONPATH=/root/workspace/FlagGems/src \\
  pytest benchmark/test_flash_attn_varlen_five_way_quick.py -v -s \\
    > /root/workspace/tmp/bench_fa3gl_$(date +%Y%m%d_%H%M%S).log 2>&1
"""
from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple

import pytest
import torch
import triton

import flag_gems

from . import base, conftest, utils

vendor_name = flag_gems.vendor_name

_TRACE_NAMES = [
    "trace0 prefill bs=1 q=512",
    "trace1 mixed bs=3 max_q=70",
    "trace2 decode bs=55 max_q=61",
    "trace3 decode bs=201 max_q=16",
]


def _is_hopper() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9


def _cu(lens):
    r = [0]
    for length in lens:
        r.append(r[-1] + length)
    return tuple(r)


@dataclass(frozen=True)
class WorkloadSpec:
    name: str
    kind: str  # "non_paged" | "paged"
    config: tuple


def _quick_workloads() -> List[WorkloadSpec]:
    all_cu_q = [
        (0, 512),
        (0, 1, 2, 72),
        tuple(range(0, 45)) + (105, 121, 137, 153, 169, 185, 201, 217, 233, 249, 265),
        tuple(range(0, 196)) + (211, 226, 240, 253, 265),
    ]
    all_kv = [
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
    workloads = [
        WorkloadSpec(
            name=f"non_paged_{i}_{_TRACE_NAMES[i]}",
            kind="non_paged",
            config=(cu_q, _cu(kv), 16, 8, 128, (-1, -1)),
        )
        for i, (cu_q, kv) in enumerate(zip(all_cu_q, all_kv))
    ]

    paged_cu_q = all_cu_q[:3]
    paged_sk = all_kv[:3]
    block_size = 64
    for i, (cu_q, sk) in enumerate(zip(paged_cu_q, paged_sk)):
        workloads.append(
            WorkloadSpec(
                name=f"paged_{i}_{_TRACE_NAMES[i]}",
                kind="paged",
                config=(
                    cu_q,
                    sk,
                    16,
                    8,
                    128,
                    block_size,
                    sum((k + block_size - 1) // block_size for k in sk) + 64,
                    (-1, -1),
                ),
            )
        )
    return workloads


def _unpack(inp: tuple) -> Tuple[list, dict]:
    bench = base.Benchmark(op_name="unpack", torch_op=lambda: None)
    return bench.unpack_to_args_kwargs(inp)


def _make_non_paged_input(config, dtype, device) -> tuple:
    cu_q_tup, cu_k_tup, nq, nk, hd, window_size = config
    cu_q_list = list(cu_q_tup)
    cu_k_list = list(cu_k_tup)
    num_seqs = len(cu_q_list) - 1
    total_q = cu_q_list[-1]
    total_k = cu_k_list[-1]
    max_q_len = max(cu_q_list[i + 1] - cu_q_list[i] for i in range(num_seqs))
    max_k_len = max(cu_k_list[i + 1] - cu_k_list[i] for i in range(num_seqs))
    scale = hd ** -0.5

    q = torch.randn(total_q, nq, hd, dtype=dtype, device=device)
    k = torch.randn(total_k, nk, hd, dtype=dtype, device=device)
    v = torch.randn_like(k)
    out = torch.empty_like(q)
    cu_q_t = torch.tensor(cu_q_list, dtype=torch.int32, device=device)
    cu_k_t = torch.tensor(cu_k_list, dtype=torch.int32, device=device)

    return (
        q,
        k,
        v,
        max_q_len,
        cu_q_t,
        max_k_len,
        cu_k_t,
        None,
        None,
        0.0,
        scale,
        True,
        list(window_size),
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
        {},
    )


def _make_paged_input(config, dtype, device) -> tuple:
    cu_q_tup, kv_lens, nq, nk, hd, block_size, num_blocks, window_size = config
    cu_q_list = list(cu_q_tup)
    kv_list = list(kv_lens)
    num_seqs = len(cu_q_list) - 1
    total_q = cu_q_list[-1]
    max_q_len = max(cu_q_list[i + 1] - cu_q_list[i] for i in range(num_seqs))
    max_kv_len = max(kv_list)
    scale = hd ** -0.5

    q = torch.randn(total_q, nq, hd, dtype=dtype, device=device)
    k_cache = torch.randn(num_blocks, block_size, nk, hd, dtype=dtype, device=device)
    v_cache = torch.randn_like(k_cache)
    out = torch.empty_like(q)
    cu_q_t = torch.tensor(cu_q_list, dtype=torch.int32, device=device)
    sk_t = torch.tensor(kv_list, dtype=torch.int32, device=device)
    max_pgs = (max_kv_len + block_size - 1) // block_size
    bt = torch.randint(
        0, num_blocks, (num_seqs, max_pgs), dtype=torch.int32, device=device
    )

    return (
        q,
        k_cache,
        v_cache,
        max_q_len,
        cu_q_t,
        max_kv_len,
        None,
        sk_t,
        None,
        0.0,
        scale,
        True,
        list(window_size),
        0.0,
        None,
        False,
        False,
        bt,
        False,
        out,
        None,
        None,
        None,
        None,
        {},
    )


def _measure_latency(op: Callable, inp: tuple) -> float:
    args, kwargs = _unpack(inp)
    fn = lambda: op(*args, **kwargs)
    return triton.testing.do_bench(
        fn,
        warmup=conftest.Config.warm_up,
        rep=conftest.Config.repetition,
        return_mode="median",
    )


def _fmt_ms(x: float) -> str:
    return f"{x:.6f}"


def _fmt_speedup(base_ms: float, lat_ms: float) -> str:
    if lat_ms <= 0:
        return "n/a"
    return f"{base_ms / lat_ms:.3f}"


def _cuda_cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def _record_shapes(inp: tuple) -> Any:
    args, kwargs = _unpack(inp)
    bench = base.Benchmark(op_name="record", torch_op=lambda: None)
    return bench.record_shapes(*args, **kwargs)


def _make_fa2_paged_from_non_paged_config(
    config, dtype, device, block_size: int = 64
) -> tuple:
    """FA2 only supports paged KV; build fresh paged inputs from non-paged config."""
    cu_q_tup, cu_k_tup, nq, nk, hd, window_size = config
    cu_q_list = list(cu_q_tup)
    cu_k_list = list(cu_k_tup)
    kv_lens = [cu_k_list[i + 1] - cu_k_list[i] for i in range(len(cu_k_list) - 1)]
    num_seqs = len(cu_q_list) - 1
    total_q = cu_q_list[-1]
    max_q_len = max(cu_q_list[i + 1] - cu_q_list[i] for i in range(num_seqs))
    max_kv_len = max(kv_lens)
    scale = hd ** -0.5

    q = torch.randn(total_q, nq, hd, dtype=dtype, device=device)
    out = torch.empty_like(q)
    num_blocks = (
        sum((k + block_size - 1) // block_size for k in kv_lens) + 64
    )
    k_cache = torch.randn(num_blocks, block_size, nk, hd, dtype=dtype, device=device)
    v_cache = torch.randn_like(k_cache)
    cu_q_t = torch.tensor(cu_q_list, dtype=torch.int32, device=device)
    sk_t = torch.tensor(kv_lens, dtype=torch.int32, device=device)
    max_pgs = (max_kv_len + block_size - 1) // block_size
    bt = torch.randint(
        0, num_blocks, (num_seqs, max_pgs), dtype=torch.int32, device=device
    )

    return (
        q,
        k_cache,
        v_cache,
        max_q_len,
        cu_q_t,
        max_kv_len,
        None,
        sk_t,
        None,
        0.0,
        scale,
        True,
        list(window_size),
        0.0,
        None,
        False,
        False,
        bt,
        False,
        out,
        None,
        None,
        None,
        None,
        {},
    )


class FiveWayQuickBenchmark(base.Benchmark):
    """14-case five-path compare for FA2/FA3/Gems Triton/Gems Gluon."""

    def set_shapes(self, shape_file_path=None):
        self.workloads = _quick_workloads()
        self.shapes = [w.name for w in self.workloads]

    def get_input_iter(self, dtype):
        for spec in self.workloads:
            if spec.kind == "non_paged":
                inp = _make_non_paged_input(spec.config, dtype, self.device)
            else:
                inp = _make_paged_input(spec.config, dtype, self.device)
            if inp is not None:
                yield spec, inp

    def run(self):
        if conftest.Config.query:
            self.init_default_config()
            return

        self.init_user_config()
        gems_op = flag_gems.ops.flash_attn_varlen_func

        from vllm.vllm_flash_attn.flash_attn_interface import (
            flash_attn_varlen_func as _vllm_fa,
        )

        def vllm_fa2(*args, **kwargs):
            kwargs.pop("fa_version", None)
            kwargs.pop("use_gluon", None)
            return _vllm_fa(*args, fa_version=2, **kwargs)

        def vllm_fa3(*args, **kwargs):
            kwargs.pop("fa_version", None)
            kwargs.pop("use_gluon", None)
            return _vllm_fa(*args, fa_version=3, **kwargs)

        def gems_fa2(*args, **kwargs):
            kwargs.pop("fa_version", None)
            kwargs.pop("use_gluon", None)
            return gems_op(*args, fa_version=2, **kwargs)

        def gems_fa3(*args, **kwargs):
            kwargs.pop("fa_version", None)
            kwargs.pop("use_gluon", None)
            return gems_op(*args, fa_version=3, use_gluon=False, **kwargs)

        def gems_gluon(*args, **kwargs):
            kwargs.pop("fa_version", None)
            kwargs.pop("use_gluon", None)
            return gems_op(*args, fa_version=3, use_gluon=True, **kwargs)

        gems_ops = {
            "gems_fa2": gems_fa2,
            "gems_fa3": gems_fa3,
            "gems_gluon": gems_gluon,
        }
        vllm_ops = {
            "vllm_fa2": vllm_fa2,
            "vllm_fa3": vllm_fa3,
        }

        print(
            f"\n{'=' * 140}\n"
            f"Five-way flash_attn_varlen quick  dtypes={self.to_bench_dtypes}  "
            f"mode={conftest.Config.mode.value}  "
            f"warmup={conftest.Config.warm_up}ms  rep={conftest.Config.repetition}ms\n"
            f"{'=' * 140}"
        )

        hdr = (
            f"{'Case':<52}"
            f"{'vllm_fa2':>10}{'vllm_fa3':>10}{'gems_fa2':>10}"
            f"{'gems_fa3':>11}{'gems_gluon':>11} | "
            f"{'sp_fa2':>8}{'sp_fa3_tri':>11}{'sp_fa3_glu':>11} | "
            f"size_detail"
        )
        print(hdr)
        print("-" * len(hdr))

        dtype_order = sorted(
            self.to_bench_dtypes,
            key=lambda d: 0 if d == torch.float16 else 1,
        )
        results: Dict[Tuple[str, torch.dtype], Dict[str, float]] = {}
        shape_details: Dict[Tuple[str, torch.dtype], Any] = {}

        def _inp_for(spec: WorkloadSpec, dtype: torch.dtype) -> tuple:
            if spec.kind == "non_paged":
                return _make_non_paged_input(spec.config, dtype, self.device)
            return _make_paged_input(spec.config, dtype, self.device)

        try:
            # Global phase 1: Gems FA3 on every case/dtype before any vLLM FA3.
            for spec in self.workloads:
                for dtype in dtype_order:
                    key = (spec.name, dtype)
                    inp = _inp_for(spec, dtype)
                    shape_details[key] = _record_shapes(inp)
                    results[key] = {
                        "gems_fa3": _measure_latency(gems_ops["gems_fa3"], inp),
                    }
                    _cuda_cleanup()

            # Global phase 2: vLLM FA3, Gems Gluon, FA2 per case/dtype.
            for spec in self.workloads:
                for dtype in dtype_order:
                    key = (spec.name, dtype)
                    inp = _inp_for(spec, dtype)
                    results[key]["vllm_fa3"] = _measure_latency(
                        vllm_ops["vllm_fa3"], inp
                    )
                    _cuda_cleanup()

                    results[key]["gems_gluon"] = _measure_latency(
                        gems_ops["gems_gluon"], inp
                    )
                    _cuda_cleanup()

                    fa2_inp = (
                        _make_fa2_paged_from_non_paged_config(
                            spec.config, dtype, self.device
                        )
                        if spec.kind == "non_paged"
                        else inp
                    )
                    results[key]["vllm_fa2"] = _measure_latency(
                        vllm_ops["vllm_fa2"], fa2_inp
                    )
                    results[key]["gems_fa2"] = _measure_latency(
                        gems_ops["gems_fa2"], fa2_inp
                    )
                    _cuda_cleanup()
        except (RuntimeError, Exception) as e:
            pytest.fail(str(e))

        for spec in self.workloads:
            for dtype in dtype_order:
                key = (spec.name, dtype)
                lat = results[key]
                shape_detail = shape_details[key]
                label = f"{spec.name} [{dtype}]"
                print(
                    f"{label:<52}"
                    f"{_fmt_ms(lat['vllm_fa2']):>10}{_fmt_ms(lat['vllm_fa3']):>10}"
                    f"{_fmt_ms(lat['gems_fa2']):>10}{_fmt_ms(lat['gems_fa3']):>11}"
                    f"{_fmt_ms(lat['gems_gluon']):>11} | "
                    f"{_fmt_speedup(lat['vllm_fa2'], lat['gems_fa2']):>8}"
                    f"{_fmt_speedup(lat['vllm_fa3'], lat['gems_fa3']):>11}"
                    f"{_fmt_speedup(lat['vllm_fa3'], lat['gems_gluon']):>11} | "
                    f"{shape_detail}"
                )

        print(
            "\nSpeedup = baseline_latency / gems_latency  (>1 means Gems faster)\n"
            "  sp_fa2     : vllm_fa2 / gems_fa2\n"
            "  sp_fa3_tri : vllm_fa3 / gems_fa3_triton (use_gluon=False)\n"
            "  sp_fa3_glu : vllm_fa3 / gems_fa3_gluon   (use_gluon=True)\n"
            "  non-paged: flat K/V + cu_seqlens_k (FA3/Gluon); FA2 uses paged blk=64 conversion\n"
            "  paged: block_table + seqused_k blk=64"
        )


@pytest.mark.skipif(not _is_hopper(), reason="FA3/Gluon requires Hopper GPU (sm_90+)")
@pytest.mark.skipif(
    utils.SkipVersion("vllm", "<0.9"),
    reason="vLLM < 0.9 does not expose flash_attn_varlen_func",
)
@pytest.mark.skipif(vendor_name in ("hygon", "cambricon"), reason="Not working")
@pytest.mark.flash_attn_varlen_func
def test_flash_attn_varlen_five_way_quick(monkeypatch):
    monkeypatch.setenv("VLLM_CONFIGURE_LOGGING", "0")

    bench = FiveWayQuickBenchmark(
        op_name="flash_attn_varlen_five_way_quick",
        torch_op=None,
        gems_op=flag_gems.ops.flash_attn_varlen_func,
        dtypes=[torch.float16, torch.bfloat16],
    )
    bench.run()