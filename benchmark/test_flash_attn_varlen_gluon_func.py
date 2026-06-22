"""
benchmark/test_flash_attn_varlen_gluon_func.py
-----------------------------------------------
Benchmark for the Gluon TMA+WGMMA flash-attention paths:
  - non-paged varlen  (cu_seqlens_k, flat K/V)
  - paged varlen      (cu_seqlens_k + block_table)

Shapes mirror test_flash_attn_varlen_fa3_func.py as closely as possible.
The baseline (torch_op) is the vllm FA3 implementation with fa_version=3.

Shape format
------------
non-paged: (cu_q, cu_k, nhead_q, nhead_k, head_size, window_size)
paged:     (cu_q, kv_lens, nhead_q, nhead_k, head_size, block_size, num_blocks, window_size)
"""
from typing import Any, List, Optional

import pytest
import torch

import flag_gems

from . import base, utils

vendor_name = flag_gems.vendor_name


def _is_hopper():
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9


# ---------------------------------------------------------------------------
# Non-paged benchmark
# ---------------------------------------------------------------------------

class GluonVarlenNonPagedBenchmark(base.Benchmark):
    """
    Benchmark for the Gluon non-paged varlen path.

    Shape categories (mirrors FA3 NonPaged benchmark)
    --------------------------------------------------
    A. Real trace (Qwen3-1.7B)
    B. Synthetic prefill: bs=1 various seqlen / head_size
    C. Synthetic prefill: bs=4/8
    D. Synthetic decode: seqlen_q=1, various batch / kvcache
    E. GQA variants: decode bs=32 sk=2048
    F. head_size variants
    """

    def set_shapes(self, shape_file_path=None):
        shapes = []

        def cu(lens):
            r = [0]
            for l in lens:
                r.append(r[-1] + l)
            return tuple(r)

        # ── A. Real trace: Qwen3-1.7B ─────────────────────────────────────
        # (cu_seqlens_q, cu_seqlens_k, nhead_q, nhead_k, head_size, window_size)
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
        for cu_q, kv in zip(all_cu_q, all_kv):
            cu_k = cu(kv)
            shapes.append((cu_q, cu_k, 16, 8, 128, (-1, -1)))

        # ── B. Synthetic prefill: bs=1 various seqlen ─────────────────────
        for seqlen in (128, 256, 512, 1024, 2048, 4096):
            shapes.append(((0, seqlen), (0, seqlen), 16, 8, 128, (-1, -1)))

        # prefill bs=1 various head_size
        for hd in (64, 128, 256):
            shapes.append(((0, 1024), (0, 1024), 16, 8, hd, (-1, -1)))

        # ── C. Synthetic prefill: bs=4/8 ──────────────────────────────────
        for seqlen in (256, 512, 1024, 2048):
            shapes.append((cu(seqlen for _ in range(4)), cu(seqlen for _ in range(4)),
                           16, 8, 128, (-1, -1)))
        shapes.append((cu(512 for _ in range(8)), cu(512 for _ in range(8)),
                       16, 8, 128, (-1, -1)))

        # ── D. Synthetic decode: seqlen_q=1 ───────────────────────────────
        for bs, kv_len in (
            (1,   512), (1,  2048),
            (8,   512), (8,  2048),
            (16,  512), (16, 2048),
            (32,  512), (32, 1024), (32, 2048),
            (64, 1024), (64, 2048),
            (128, 1024), (128, 2048),
        ):
            shapes.append((cu(1 for _ in range(bs)), cu(kv_len for _ in range(bs)),
                           16, 8, 128, (-1, -1)))

        # ── E. GQA variants: decode bs=32 sk=2048 ─────────────────────────
        cu_q_d32 = cu(1 for _ in range(32))
        cu_k_d32 = cu(2048 for _ in range(32))
        for nq, nk in ((4, 4), (8, 2), (8, 1), (16, 2), (16, 1), (32, 8), (32, 4)):
            shapes.append((cu_q_d32, cu_k_d32, nq, nk, 128, (-1, -1)))

        # prefill bs=1 sq=1024
        for nq, nk in ((4, 4), (8, 2), (16, 2), (32, 8)):
            shapes.append(((0, 1024), (0, 1024), nq, nk, 128, (-1, -1)))

        # ── F. head_size variants: decode bs=32 sk=2048 ───────────────────
        for hd in (64, 128, 256):
            shapes.append((cu_q_d32, cu_k_d32, 16, 8, hd, (-1, -1)))

        self.shapes = shapes

    def get_input_iter(self, dtype):
        for cfg in self.shapes:
            inp = self._make_input(cfg, dtype, self.device)
            if inp is not None:
                yield inp

    def _make_input(self, config, dtype, device):
        cu_q_tup, cu_k_tup, nq, nk, hd, window_size = config

        cu_q_list = list(cu_q_tup)
        cu_k_list = list(cu_k_tup)
        num_seqs   = len(cu_q_list) - 1
        total_q    = cu_q_list[-1]
        total_k    = cu_k_list[-1]
        max_q_len  = max(cu_q_list[i+1] - cu_q_list[i] for i in range(num_seqs))
        max_k_len  = max(cu_k_list[i+1] - cu_k_list[i] for i in range(num_seqs))
        scale      = hd ** -0.5

        q   = torch.randn(total_q, nq, hd, dtype=dtype, device=device)
        k   = torch.randn(total_k, nk, hd, dtype=dtype, device=device)
        v   = torch.randn_like(k)
        out = torch.empty_like(q)

        cu_q_t = torch.tensor(cu_q_list, dtype=torch.int32, device=device)
        cu_k_t = torch.tensor(cu_k_list, dtype=torch.int32, device=device)

        return (
            q, k, v,
            max_q_len,           # max_seqlen_q
            cu_q_t,              # cu_seqlens_q
            max_k_len,           # max_seqlen_k
            cu_k_t,              # cu_seqlens_k  <- non-paged marker
            None,                # seqused_k
            None,                # q_v
            0.0,                 # dropout_p
            scale,               # softmax_scale
            True,                # causal
            list(window_size),   # window_size
            0.0,                 # softcap
            None,                # alibi_slopes
            False,               # deterministic
            False,               # return_attn_probs
            None,                # block_table  <- None = non-paged
            False,               # return_softmax_lse
            out,                 # out
            None,                # scheduler_metadata
            None, None, None,    # q/k/v_descale
            {"fa_version": 3},
        )


# ---------------------------------------------------------------------------
# Paged benchmark
# ---------------------------------------------------------------------------

class GluonVarlenPagedBenchmark(base.Benchmark):
    """
    Benchmark for the Gluon paged varlen path.

    Uses cu_seqlens_k (not seqused_k) + block_table to route through Gluon.

    Shape categories (mirrors FA3 paged benchmark)
    -----------------------------------------------
    A. Real trace (Qwen3-1.7B)
    B. Synthetic prefill: bs=1 various seqlen
    C. Synthetic decode: seqlen_q=1, various batch/kvcache
    D. GQA variants
    E. head_size variants
    F. block_size variants: 64 / 128
    """

    def set_shapes(self, shape_file_path=None):
        # format: (cu_q, kv_lens_tuple, nq, nk, head_size, block_size, num_blocks, window)
        shapes = []

        def cu(lens):
            r = [0]
            for l in lens:
                r.append(r[-1] + l)
            return tuple(r)

        # ── A. Real trace: Qwen3-1.7B ─────────────────────────────────────
        # same sequence shapes as FA3 paged benchmark
        all_cu_q = [
            (0, 512),
            (0, 1, 2, 72),
            tuple(range(0, 45)) + (105, 121, 137, 153, 169, 185, 201, 217, 233, 249, 265),
        ]
        all_sk = [
            (512,),
            (1, 1, 70),
            (515,) + (514,) * 20 + (513,) * 20 + (512,) * 14,
        ]
        for cu_q, sk in zip(all_cu_q, all_sk):
            max_kv    = max(sk)
            block_size = 64
            num_blocks = sum((k + block_size - 1) // block_size for k in sk) + 64
            shapes.append((cu_q, sk, 16, 8, 128, block_size, num_blocks, (-1, -1)))

        # ── B. Synthetic prefill: bs=1 various seqlen ─────────────────────
        for seqlen in (128, 256, 512, 1024, 2048, 4096):
            block_size = 64
            num_blocks = (seqlen + block_size - 1) // block_size + 32
            shapes.append(((0, seqlen), (seqlen,), 16, 8, 128, block_size, num_blocks, (-1, -1)))

        # prefill bs=1 various head_size
        for hd in (64, 128, 256):
            block_size = 64
            num_blocks = (1024 + block_size - 1) // block_size + 32
            shapes.append(((0, 1024), (1024,), 16, 8, hd, block_size, num_blocks, (-1, -1)))

        # prefill bs=4 various seqlen
        for seqlen in (256, 512, 1024, 2048):
            block_size = 64
            num_blocks = 4 * (seqlen + block_size - 1) // block_size + 64
            shapes.append((cu(seqlen for _ in range(4)), (seqlen,) * 4,
                           16, 8, 128, block_size, num_blocks, (-1, -1)))

        # prefill bs=8, seqlen=512
        block_size = 64
        num_blocks = 8 * (512 + block_size - 1) // block_size + 64
        shapes.append((cu(512 for _ in range(8)), (512,) * 8,
                       16, 8, 128, block_size, num_blocks, (-1, -1)))

        # ── C. Synthetic decode: seqlen_q=1 ───────────────────────────────
        for bs, kv_len in (
            (1,   512), (1,  2048),
            (8,   512), (8,  2048),
            (16,  512), (16, 2048),
            (32,  512), (32, 1024), (32, 2048),
            (64, 1024), (64, 2048),
            (128, 1024), (128, 2048),
        ):
            block_size = 64
            num_blocks = bs * (kv_len + block_size - 1) // block_size + 64
            shapes.append((cu(1 for _ in range(bs)), (kv_len,) * bs,
                           16, 8, 128, block_size, num_blocks, (-1, -1)))

        # ── D. GQA variants: decode bs=32 sk=2048 ─────────────────────────
        cu_q_d32 = cu(1 for _ in range(32))
        sk_d32   = (2048,) * 32
        block_d  = 64
        nb_d     = 32 * (2048 + block_d - 1) // block_d + 64
        for nq, nk in ((4, 4), (8, 2), (8, 1), (16, 2), (16, 1), (32, 8), (32, 4)):
            shapes.append((cu_q_d32, sk_d32, nq, nk, 128, block_d, nb_d, (-1, -1)))

        # prefill bs=1 sq=1024 GQA variants
        for nq, nk in ((4, 4), (8, 2), (16, 2), (32, 8)):
            num_blocks_p = (1024 + 63) // 64 + 32
            shapes.append(((0, 1024), (1024,), nq, nk, 128, 64, num_blocks_p, (-1, -1)))

        # ── E. head_size variants: decode bs=32 sk=2048 ───────────────────
        for hd in (64, 128, 256):
            shapes.append((cu_q_d32, sk_d32, 16, 8, hd, block_d, nb_d, (-1, -1)))

        # ── F. block_size variants ─────────────────────────────────────────
        for block_size in (64, 128):
            nb = 32 * (2048 + block_size - 1) // block_size + 64
            shapes.append((cu_q_d32, sk_d32, 16, 8, 128, block_size, nb, (-1, -1)))
            # prefill
            nb_p = (1024 + block_size - 1) // block_size + 32
            shapes.append(((0, 1024), (1024,), 16, 8, 128, block_size, nb_p, (-1, -1)))

        self.shapes = shapes

    def get_input_iter(self, dtype):
        for cfg in self.shapes:
            inp = self._make_input(cfg, dtype, self.device)
            if inp is not None:
                yield inp

    def _make_input(self, config, dtype, device):
        cu_q_tup, kv_lens, nq, nk, hd, block_size, num_blocks, window_size = config

        cu_q_list  = list(cu_q_tup)
        kv_list    = list(kv_lens)
        num_seqs   = len(cu_q_list) - 1
        total_q    = cu_q_list[-1]
        max_q_len  = max(cu_q_list[i+1] - cu_q_list[i] for i in range(num_seqs))
        max_kv_len = max(kv_list)
        scale      = hd ** -0.5

        q       = torch.randn(total_q,   nq, hd, dtype=dtype, device=device)
        k_cache = torch.randn(num_blocks, block_size, nk, hd, dtype=dtype, device=device)
        v_cache = torch.randn_like(k_cache)
        out     = torch.empty_like(q)

        cu_q_t = torch.tensor(cu_q_list, dtype=torch.int32, device=device)
        # build cu_seqlens_k from kv_lens (cumsum form, NOT seqused_k)
        cu_k_list = [0]
        for l in kv_list:
            cu_k_list.append(cu_k_list[-1] + l)
        cu_k_t = torch.tensor(cu_k_list, dtype=torch.int32, device=device)

        max_pgs = (max_kv_len + block_size - 1) // block_size
        bt = torch.randint(0, num_blocks, (num_seqs, max_pgs),
                           dtype=torch.int32, device=device)

        return (
            q, k_cache, v_cache,
            max_q_len,           # max_seqlen_q
            cu_q_t,              # cu_seqlens_q
            max_kv_len,          # max_seqlen_k
            cu_k_t,              # cu_seqlens_k  <- paged + cu_seqlens_k -> Gluon
            None,                # seqused_k
            None,                # q_v
            0.0,                 # dropout_p
            scale,               # softmax_scale
            True,                # causal
            list(window_size),   # window_size
            0.0,                 # softcap
            None,                # alibi_slopes
            False,               # deterministic
            False,               # return_attn_probs
            bt,                  # block_table  <- paged
            False,               # return_softmax_lse
            out,                 # out
            None,                # scheduler_metadata
            None, None, None,    # q/k/v_descale
            {"fa_version": 3},
        )


# ---------------------------------------------------------------------------
# pytest entry points
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _is_hopper(), reason="Gluon requires Hopper GPU (sm_90+)")
@pytest.mark.skipif(
    utils.SkipVersion("vllm", "<0.9"),
    reason="vLLM < 0.9 does not expose flash_attn_varlen_func",
)
@pytest.mark.skipif(vendor_name in ("hygon", "cambricon"), reason="Not working")
@pytest.mark.flash_attn_varlen_func
def test_gluon_varlen_non_paged_benchmark(monkeypatch):
    monkeypatch.setenv("VLLM_CONFIGURE_LOGGING", "0")

    from vllm.vllm_flash_attn.flash_attn_interface import flash_attn_varlen_func as _vllm_fa

    def vllm_fa3(*args, **kwargs):
        kwargs.pop("fa_version", None)
        return _vllm_fa(*args, fa_version=3, **kwargs)

    bench = GluonVarlenNonPagedBenchmark(
        op_name="flash_attn_varlen_gluon_non_paged",
        torch_op=vllm_fa3,
        gems_op=flag_gems.ops.flash_attn_varlen_func,
        dtypes=[torch.float16, torch.bfloat16],
    )
    bench.run()


@pytest.mark.skipif(not _is_hopper(), reason="Gluon requires Hopper GPU (sm_90+)")
@pytest.mark.skipif(
    utils.SkipVersion("vllm", "<0.9"),
    reason="vLLM < 0.9 does not expose flash_attn_varlen_func",
)
@pytest.mark.skipif(vendor_name in ("hygon", "cambricon"), reason="Not working")
@pytest.mark.flash_attn_varlen_func
def test_gluon_varlen_paged_benchmark(monkeypatch):
    monkeypatch.setenv("VLLM_CONFIGURE_LOGGING", "0")

    from vllm.vllm_flash_attn.flash_attn_interface import flash_attn_varlen_func as _vllm_fa

    def vllm_fa3(*args, **kwargs):
        kwargs.pop("fa_version", None)
        # Benchmark passes all params as positional args; vllm's paged path
        # requires seqused_k (index 7) instead of cu_seqlens_k (index 6).
        # Convert when block_table (index 17) is present.
        args = list(args)
        CU_K_IDX, SEQUSED_K_IDX, BLOCK_TABLE_IDX = 6, 7, 17
        if (len(args) > BLOCK_TABLE_IDX
                and args[BLOCK_TABLE_IDX] is not None
                and args[CU_K_IDX] is not None):
            cu_k = args[CU_K_IDX]
            args[SEQUSED_K_IDX] = (cu_k[1:] - cu_k[:-1]).to(torch.int32)
            args[CU_K_IDX] = None   # vllm paged: cu_seqlens_k must be None
        return _vllm_fa(*args, fa_version=3, **kwargs)

    bench = GluonVarlenPagedBenchmark(
        op_name="flash_attn_varlen_gluon_paged",
        torch_op=vllm_fa3,
        gems_op=flag_gems.ops.flash_attn_varlen_func,
        dtypes=[torch.float16, torch.bfloat16],
    )
    bench.run()
