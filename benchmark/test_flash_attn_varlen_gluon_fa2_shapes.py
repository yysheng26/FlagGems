"""
Compare 4 Qwen3-1.7B trace shapes across FA2 / FA3 / Gluon paths.

Runs the same 4 traces twice: block_size=16 (vLLM FA2 default) and block_size=64
(Gluon-tuned page size) for apples-to-apples ws/Gluon comparison.

Paged KV shapes:
  - vLLM FA2 varlen
  - vLLM FA3 varlen
  - Gems FA2 varlen   (fa_version=2, seqused_k)
  - Gems FA3 varlen   (fa_version=3, seqused_k — Triton kernel, not Gluon)
  - Gems FA3 Gluon    (fa_version=3, cu_seqlens_k)

Usage (Hopper, fa3-varlen-gluon branch):
  pytest -s benchmark/test_flash_attn_varlen_gluon_fa2_shapes.py \
    -m flash_attn_varlen_func --level core

Quick fp16-only:
  pytest -s benchmark/test_flash_attn_varlen_gluon_fa2_shapes.py \
    -m flash_attn_varlen_func --level core --dtypes float16
"""
from __future__ import annotations

import gc
from typing import Any, Callable, Dict, Tuple

import pytest
import torch
import triton

import flag_gems

from . import base, conftest, utils
from .test_flash_attn_varlen_func import FlashAttnVarlenBenchmark

vendor_name = flag_gems.vendor_name

_TRACE_NAMES = [
    "case0 prefill bs=1 q=512",
    "case1 mixed bs=3 max_q=70",
    "case2 decode bs=55 max_q=61",
    "case3 decode bs=201 max_q=16",
]

_BLOCK_SIZES = (16, 64)

def _is_hopper() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9


def _seqused_k_to_cu_seqlens_k(seqused_k: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        [
            torch.zeros(1, dtype=torch.int32, device=seqused_k.device),
            torch.cumsum(seqused_k, dim=0),
        ]
    )


def _unpack(inp: tuple) -> Tuple[list, dict]:
    bench = base.Benchmark(op_name="unpack", torch_op=lambda: None)
    return bench.unpack_to_args_kwargs(inp)


def _with_fa_version(inp: tuple, fa_version: int) -> tuple:
    args, kwargs = _unpack(inp)
    kwargs = dict(kwargs)
    kwargs["fa_version"] = fa_version
    return tuple(args) + (kwargs,)


def _to_gluon_input(fa2_inp: tuple) -> tuple:
    args, kwargs = _unpack(fa2_inp)
    args = list(args)
    seqused_k = args[7]
    args[6] = _seqused_k_to_cu_seqlens_k(seqused_k)
    args[7] = None
    kwargs = dict(kwargs)
    kwargs["fa_version"] = 3
    kwargs["use_gluon"] = True
    return tuple(args) + (kwargs,)


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


def _case_summary(config: tuple) -> str:
    cu_q, seqused_k, block_size = config[0], config[1], config[5]
    num_seqs = len(cu_q) - 1
    max_q = max(cu_q[i + 1] - cu_q[i] for i in range(num_seqs))
    max_k = max(seqused_k)
    return f"blk={block_size} bs={num_seqs} max_q={max_q} max_k={max_k}"


def _case_label(case_idx: int) -> str:
    return _TRACE_NAMES[case_idx % len(_TRACE_NAMES)]


class Fa2ShapesMultiCompareBenchmark(FlashAttnVarlenBenchmark):
    """Run all 5 paths on FA2's 4 Qwen trace configs and print a compare table."""

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path)
        traces = [
            (cu_q, sk, nq, nk, hd, nb, alibi, sc)
            for cu_q, sk, nq, nk, hd, _, nb, alibi, sc in self.shapes
        ]
        # block_size outer: rows 0-3 blk=16, rows 4-7 blk=64 (matches case_idx % 4)
        self.shapes = [
            (cu_q, sk, nq, nk, hd, block_size, nb, alibi, sc)
            for block_size in _BLOCK_SIZES
            for cu_q, sk, nq, nk, hd, nb, alibi, sc in traces
        ]

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
            return _vllm_fa(*args, fa_version=2, **kwargs)

        def vllm_fa3(*args, **kwargs):
            kwargs.pop("fa_version", None)
            return _vllm_fa(*args, fa_version=3, **kwargs)

        def gems_fa2(*args, **kwargs):
            kwargs["fa_version"] = 2
            return gems_op(*args, **kwargs)

        def gems_fa3(*args, **kwargs):
            kwargs["fa_version"] = 3
            kwargs["use_gluon"] = False
            return gems_op(*args, **kwargs)

        def gems_gluon(*args, **kwargs):
            kwargs["fa_version"] = 3
            kwargs["use_gluon"] = True
            return gems_op(*args, **kwargs)

        for dtype in self.to_bench_dtypes:
            print(
                f"\n{'=' * 120}\n"
                f"FA2-shape multi-path compare  dtype={dtype}  "
                f"mode={conftest.Config.mode.value}  level={conftest.Config.bench_level.value}\n"
                f"{'=' * 120}"
            )
            case_w = 56
            lat_w = 10
            sp_w = 8
            print(
                "Columns: latency(ms)×5 "
                "| speedup vs vllm_fa2×4 "
                "| speedup vs vllm_fa3×2 "
                "| gluon/triton×1"
            )
            row_cols = (
                f"{'Case':<{case_w}}"
                f"{'vllm_fa2':>{lat_w}}{'vllm_fa3':>{lat_w}}{'gems_fa2':>{lat_w}}"
                f"{'gems_fa3':>{lat_w}}{'gems_gluon':>{lat_w}} | "
                f"{'vllm_fa3':>{sp_w}}{'gems_fa2':>{sp_w}}{'gems_fa3':>{sp_w}}{'gems_gluon':>{sp_w}} | "
                f"{'gems_fa3':>{sp_w}}{'gems_gluon':>{sp_w}} | "
                f"{'gems_gluon':>{sp_w}}"
            )
            print(row_cols)
            print("-" * len(row_cols))

            prev_block_size = None
            for case_idx, config in enumerate(self.shapes):
                block_size = config[5]
                if prev_block_size is not None and block_size != prev_block_size:
                    print(f"{'--- block_size=' + str(block_size) + ' ' + '-' * 80}")
                prev_block_size = block_size

                fa2_inp = self.flash_attn_varlen_input_fn(config, dtype, self.device)
                if fa2_inp is None:
                    continue

                fa3_inp = _with_fa_version(fa2_inp, fa_version=3)
                gluon_inp = _to_gluon_input(fa2_inp)

                label = _case_label(case_idx)
                summary = _case_summary(config)

                lat: Dict[str, float] = {}
                try:
                    lat["vllm_fa2"] = _measure_latency(vllm_fa2, fa2_inp)
                    lat["vllm_fa3"] = _measure_latency(vllm_fa3, fa3_inp)
                    lat["gems_fa2"] = _measure_latency(gems_fa2, fa2_inp)
                    lat["gems_fa3"] = _measure_latency(gems_fa3, fa3_inp)
                    lat["gems_gluon"] = _measure_latency(gems_gluon, gluon_inp)
                except (RuntimeError, Exception) as e:
                    pytest.fail(f"{label} ({summary}) failed: {e}")
                finally:
                    gc.collect()

                base_fa2 = lat["vllm_fa2"]
                base_fa3 = lat["vllm_fa3"]

                print(
                    f"{(label + ' | ' + summary):<{case_w}}"
                    f"{_fmt_ms(lat['vllm_fa2']):>{lat_w}}{_fmt_ms(lat['vllm_fa3']):>{lat_w}}"
                    f"{_fmt_ms(lat['gems_fa2']):>{lat_w}}{_fmt_ms(lat['gems_fa3']):>{lat_w}}"
                    f"{_fmt_ms(lat['gems_gluon']):>{lat_w}} | "
                    f"{_fmt_speedup(base_fa2, lat['vllm_fa3']):>{sp_w}}"
                    f"{_fmt_speedup(base_fa2, lat['gems_fa2']):>{sp_w}}"
                    f"{_fmt_speedup(base_fa2, lat['gems_fa3']):>{sp_w}}"
                    f"{_fmt_speedup(base_fa2, lat['gems_gluon']):>{sp_w}} | "
                    f"{_fmt_speedup(base_fa3, lat['gems_fa3']):>{sp_w}}"
                    f"{_fmt_speedup(base_fa3, lat['gems_gluon']):>{sp_w}} | "
                    f"{_fmt_speedup(lat['gems_fa3'], lat['gems_gluon']):>{sp_w}}"
                )

            print(
                "\nSpeedup = baseline_latency / path_latency  (>1 means faster than baseline)\n"
                "  vs vllm_fa2 : compare each path against vLLM FA2 latency\n"
                "  vs vllm_fa3 : Gems FA3 Triton / Gluon against vLLM FA3 only\n"
                "  gluon/triton: Gems Gluon against Gems FA3 Triton (same row)\n"
                "  gems_fa3 = seqused_k (Triton); gems_gluon = cu_seqlens_k (Gluon)\n"
                "  rows 0-3 block_size=16; rows 4-7 block_size=64"
            )


@pytest.mark.skipif(not _is_hopper(), reason="Gluon requires Hopper GPU (sm_90+)")
@pytest.mark.skipif(
    utils.SkipVersion("vllm", "<0.9"),
    reason="vLLM < 0.9 does not expose flash_attn_varlen_func",
)
@pytest.mark.skipif(vendor_name in ("hygon", "cambricon"), reason="Not working")
@pytest.mark.flash_attn_varlen_func
def test_gluon_fa2_shapes_compare(monkeypatch):
    monkeypatch.setenv("VLLM_CONFIGURE_LOGGING", "0")

    bench = Fa2ShapesMultiCompareBenchmark(
        op_name="flash_attn_varlen_gluon_fa2_shapes_compare",
        torch_op=None,
        gems_op=flag_gems.ops.flash_attn_varlen_func,
        dtypes=[torch.float16, torch.bfloat16],
    )
    bench.run()