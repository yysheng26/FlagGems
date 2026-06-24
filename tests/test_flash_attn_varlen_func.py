"""
test_flash_attn_varlen_gluon_func.py
-------------------------------------
正确性测试：Gluon TMA+WGMMA flash-attention 的 varlen (non-paged) 和 paged KV 路径。

与 test_flash_attn_varlen_fa3_func.py 的关键区别
------------------------------------------------
FA3 paged 测试用 seqused_k；Gluon paged 测试用 cu_seqlens_k，这是 FA3 warp_specialize
因 TaskIdPropagation 失败而需要 Gluon 接管的场景。

覆盖的场景
----------
non-paged : prefill 多序列、不同 head_size / GQA、causal / non-causal
paged     : prefill / decode、GQA、不同 block_size / head_size、causal / non-causal
不测       : softcap / sliding-window（Gluon kernel 暂不支持）
"""
from typing import List, Tuple

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from .conftest import QUICK_MODE

device      = flag_gems.device
vendor_name = flag_gems.vendor_name

_SKIP_VENDOR = pytest.mark.skipif(
    vendor_name in ("kunlunxin", "hygon"), reason="Not supported on this vendor"
)
_SKIP_GPU = pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9),
    reason="Gluon requires Hopper GPU (sm_90+)",
)


# ---------------------------------------------------------------------------
# 参数集
# ---------------------------------------------------------------------------
if QUICK_MODE:
    NUM_HEADS   = [(8, 2)]
    HEAD_SIZES  = [128]
    DTYPES      = [torch.float16]
    BLOCK_SIZES = [64]
else:
    NUM_HEADS   = [(4, 4), (8, 2), (16, 2)]
    HEAD_SIZES  = [64, 128, 256]
    DTYPES      = [torch.float16, torch.bfloat16]
    BLOCK_SIZES = [64, 128]   # Gluon paged: BLOCK_N == block_size


# ---------------------------------------------------------------------------
# reference 实现
# ---------------------------------------------------------------------------

def _ref_varlen(q, k, v, cu_q, cu_k, scale, causal):
    """Dense varlen reference (non-paged)."""
    S     = cu_q.numel() - 1
    ratio = q.shape[1] // k.shape[1]
    out   = torch.empty_like(q)
    for i in range(S):
        qs, qe = int(cu_q[i]), int(cu_q[i + 1])
        ks, ke = int(cu_k[i]), int(cu_k[i + 1])
        qi = q[qs:qe].float()
        ki = k[ks:ke].float()
        vi = v[ks:ke].float()
        if ratio > 1:
            ki = ki.repeat_interleave(ratio, dim=1)
            vi = vi.repeat_interleave(ratio, dim=1)
        a = torch.einsum("qhd,khd->hqk", qi, ki) * scale
        if causal:
            ql, kl = qe - qs, ke - ks
            m = torch.triu(torch.ones(ql, kl, device=q.device, dtype=torch.bool),
                           diagonal=kl - ql + 1)
            a.masked_fill_(m.unsqueeze(0), float("-inf"))
        a = torch.softmax(a, dim=-1).to(q.dtype)
        out[qs:qe] = torch.einsum("hqk,khd->qhd", a, vi.to(q.dtype))
    return out


def _ref_paged(q, k_cache, v_cache, cu_q, cu_k, block_tables, scale, causal):
    """Paged KV reference: reconstruct dense K/V per-sequence then attend."""
    S          = cu_q.numel() - 1
    block_size = k_cache.shape[1]
    ratio      = q.shape[1] // k_cache.shape[2]
    out        = torch.empty_like(q)
    for i in range(S):
        qs, qe = int(cu_q[i]), int(cu_q[i + 1])
        ks, ke = int(cu_k[i]), int(cu_k[i + 1])
        kl     = ke - ks
        npages = (kl + block_size - 1) // block_size
        pids   = block_tables[i, :npages]
        kd = k_cache[pids].reshape(-1, k_cache.shape[2], k_cache.shape[3])[:kl]
        vd = v_cache[pids].reshape(-1, v_cache.shape[2], v_cache.shape[3])[:kl]
        qi = q[qs:qe].float()
        ki = kd.float(); vi = vd.float()
        if ratio > 1:
            ki = ki.repeat_interleave(ratio, dim=1)
            vi = vi.repeat_interleave(ratio, dim=1)
        a = torch.einsum("qhd,khd->hqk", qi, ki) * scale
        if causal:
            ql = qe - qs
            m = torch.triu(torch.ones(ql, kl, device=q.device, dtype=torch.bool),
                           diagonal=kl - ql + 1)
            a.masked_fill_(m.unsqueeze(0), float("-inf"))
        a = torch.softmax(a, dim=-1).to(q.dtype)
        out[qs:qe] = torch.einsum("hqk,khd->qhd", a, vi.to(q.dtype))
    return out


def _cu(lens, dev):
    t = torch.zeros(len(lens) + 1, dtype=torch.int32, device=dev)
    torch.cumsum(torch.tensor(lens, dtype=torch.int32, device=dev), 0, out=t[1:])
    return t


def _block_tables(cu_k, block_size, num_blocks, dev):
    S       = cu_k.numel() - 1
    kv_lens = (cu_k[1:] - cu_k[:-1]).tolist()
    max_pgs = max((int(kl) + block_size - 1) // block_size for kl in kv_lens)
    return torch.randint(0, num_blocks, (S, max_pgs), dtype=torch.int32, device=dev)


# ---------------------------------------------------------------------------
# non-paged: prefill
# ---------------------------------------------------------------------------

@pytest.mark.flash_attn_varlen_func
@_SKIP_GPU
@_SKIP_VENDOR
@pytest.mark.parametrize("seq_lens", [
    [(512, 512), (256, 256), (128, 128)],
    [(1, 1328), (5, 18), (129, 463)],
])
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("causal", [True, False])
def test_gluon_varlen_non_paged(seq_lens, num_heads, head_size, dtype, causal):
    """non-paged: flat K/V + cu_seqlens_k -> Gluon non-paged fast-path."""
    with torch.device(device):
        utils.init_seed(1234567890)
        nq, nk  = num_heads
        scale   = head_size ** -0.5
        q_lens  = [s[0] for s in seq_lens]
        kv_lens = [s[1] for s in seq_lens]

        q    = torch.randn(sum(q_lens),  nq, head_size, dtype=dtype)
        k    = torch.randn(sum(kv_lens), nk, head_size, dtype=dtype)
        v    = torch.randn_like(k)
        cu_q = _cu(q_lens,  device)
        cu_k = _cu(kv_lens, device)

        out = flag_gems.ops.flash_attn_varlen_func(
            q=q, k=k, v=v,
            cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
            max_seqlen_q=max(q_lens), max_seqlen_k=max(kv_lens),
            softmax_scale=scale, causal=causal,
            window_size=(-1, -1), softcap=0, fa_version=3,
        )
        ref = _ref_varlen(q, k, v, cu_q, cu_k, scale, causal)
        msg = f"max_diff={torch.max(torch.abs(out - ref))}"
        torch.testing.assert_close(out, ref, atol=2e-2, rtol=1e-2, msg=msg)


# ---------------------------------------------------------------------------
# paged: prefill - mixed (ql, kl) sequences
# ---------------------------------------------------------------------------

@pytest.mark.flash_attn_varlen_func
@_SKIP_GPU
@_SKIP_VENDOR
@pytest.mark.parametrize("seq_lens", [
    [(1, 1328), (5, 18), (129, 463)],
    [(64, 64), (128, 128), (64, 256)],
])
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_gluon_paged_prefill(seq_lens, num_heads, head_size, block_size, dtype):
    """paged prefill: mixed (ql, kl), causal=True."""
    with torch.device(device):
        utils.init_seed(1234567890)
        nq, nk  = num_heads
        scale   = head_size ** -0.5
        q_lens  = [s[0] for s in seq_lens]
        kv_lens = [s[1] for s in seq_lens]
        num_blocks = 2048

        q       = torch.randn(sum(q_lens),  nq, head_size, dtype=dtype)
        k_cache = torch.randn(num_blocks, block_size, nk, head_size, dtype=dtype)
        v_cache = torch.randn_like(k_cache)
        cu_q    = _cu(q_lens,  device)
        cu_k    = _cu(kv_lens, device)
        bt      = _block_tables(cu_k, block_size, num_blocks, device)

        out = flag_gems.ops.flash_attn_varlen_func(
            q=q, k=k_cache, v=v_cache,
            cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
            max_seqlen_q=max(q_lens), max_seqlen_k=max(kv_lens),
            softmax_scale=scale, causal=True,
            window_size=(-1, -1), block_table=bt, softcap=0, fa_version=3,
        )
        ref = _ref_paged(q, k_cache, v_cache, cu_q, cu_k, bt, scale, causal=True)
        msg = f"max_diff={torch.max(torch.abs(out - ref))}"
        torch.testing.assert_close(out, ref, atol=2e-2, rtol=1e-2, msg=msg)


# ---------------------------------------------------------------------------
# paged: prefill non-causal
# ---------------------------------------------------------------------------

@pytest.mark.flash_attn_varlen_func
@_SKIP_GPU
@_SKIP_VENDOR
@pytest.mark.parametrize("seq_lens", [[(64, 128), (128, 64), (64, 64)]])
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_gluon_paged_non_causal(seq_lens, num_heads, head_size, block_size, dtype):
    """paged prefill non-causal: kl can differ from ql in either direction."""
    with torch.device(device):
        utils.init_seed(42)
        nq, nk  = num_heads
        scale   = head_size ** -0.5
        q_lens  = [s[0] for s in seq_lens]
        kv_lens = [s[1] for s in seq_lens]
        num_blocks = 2048

        q       = torch.randn(sum(q_lens),  nq, head_size, dtype=dtype)
        k_cache = torch.randn(num_blocks, block_size, nk, head_size, dtype=dtype)
        v_cache = torch.randn_like(k_cache)
        cu_q    = _cu(q_lens,  device)
        cu_k    = _cu(kv_lens, device)
        bt      = _block_tables(cu_k, block_size, num_blocks, device)

        out = flag_gems.ops.flash_attn_varlen_func(
            q=q, k=k_cache, v=v_cache,
            cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
            max_seqlen_q=max(q_lens), max_seqlen_k=max(kv_lens),
            softmax_scale=scale, causal=False,
            window_size=(-1, -1), block_table=bt, softcap=0, fa_version=3,
        )
        ref = _ref_paged(q, k_cache, v_cache, cu_q, cu_k, bt, scale, causal=False)
        msg = f"max_diff={torch.max(torch.abs(out - ref))}"
        torch.testing.assert_close(out, ref, atol=2e-2, rtol=1e-2, msg=msg)


# ---------------------------------------------------------------------------
# paged: decode (seqlen_q=1, large batch)
# ---------------------------------------------------------------------------

@pytest.mark.flash_attn_varlen_func
@_SKIP_GPU
@_SKIP_VENDOR
@pytest.mark.parametrize("seq_lens", [
    [(1, 2048)],
    [(1, 512)]  * 32,
    [(1, 2048)] * 16,
    [(1, 1024)] * 64,
])
@pytest.mark.parametrize("num_heads", [(16, 8), (8, 2)] if not QUICK_MODE else [(8, 2)])
@pytest.mark.parametrize("head_size", [128] if not QUICK_MODE else [128])
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_gluon_paged_decode(seq_lens, num_heads, head_size, block_size, dtype):
    """paged decode: seqlen_q=1, large batch, causal=True."""
    with torch.device(device):
        utils.init_seed(42)
        nq, nk  = num_heads
        scale   = head_size ** -0.5
        q_lens  = [s[0] for s in seq_lens]
        kv_lens = [s[1] for s in seq_lens]
        num_blocks = max(2048, sum((kl + block_size - 1) // block_size
                                   for kl in kv_lens) + 64)

        q       = torch.randn(sum(q_lens),  nq, head_size, dtype=dtype)
        k_cache = torch.randn(num_blocks, block_size, nk, head_size, dtype=dtype)
        v_cache = torch.randn_like(k_cache)
        cu_q    = _cu(q_lens,  device)
        cu_k    = _cu(kv_lens, device)
        bt      = _block_tables(cu_k, block_size, num_blocks, device)

        out = flag_gems.ops.flash_attn_varlen_func(
            q=q, k=k_cache, v=v_cache,
            cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
            max_seqlen_q=max(q_lens), max_seqlen_k=max(kv_lens),
            softmax_scale=scale, causal=True,
            window_size=(-1, -1), block_table=bt, softcap=0, fa_version=3,
        )
        ref = _ref_paged(q, k_cache, v_cache, cu_q, cu_k, bt, scale, causal=True)
        msg = f"max_diff={torch.max(torch.abs(out - ref))}"
        torch.testing.assert_close(out, ref, atol=2e-2, rtol=1e-2, msg=msg)


# ---------------------------------------------------------------------------
# paged: GQA variants
# ---------------------------------------------------------------------------

@pytest.mark.flash_attn_varlen_func
@_SKIP_GPU
@_SKIP_VENDOR
@pytest.mark.parametrize("num_heads", [
    (4, 4), (8, 2), (8, 1), (16, 2), (16, 4),
] if not QUICK_MODE else [(8, 2)])
@pytest.mark.parametrize("seq_lens", [[(1, 2048)] * 16])
@pytest.mark.parametrize("head_size", [128])
@pytest.mark.parametrize("block_size", [64])
@pytest.mark.parametrize("dtype", DTYPES)
def test_gluon_paged_gqa(num_heads, seq_lens, head_size, block_size, dtype):
    """paged decode GQA: various nq/nk ratios."""
    with torch.device(device):
        utils.init_seed(42)
        nq, nk  = num_heads
        scale   = head_size ** -0.5
        q_lens  = [s[0] for s in seq_lens]
        kv_lens = [s[1] for s in seq_lens]
        num_blocks = 4096

        q       = torch.randn(sum(q_lens),  nq, head_size, dtype=dtype)
        k_cache = torch.randn(num_blocks, block_size, nk, head_size, dtype=dtype)
        v_cache = torch.randn_like(k_cache)
        cu_q    = _cu(q_lens,  device)
        cu_k    = _cu(kv_lens, device)
        bt      = _block_tables(cu_k, block_size, num_blocks, device)

        out = flag_gems.ops.flash_attn_varlen_func(
            q=q, k=k_cache, v=v_cache,
            cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
            max_seqlen_q=max(q_lens), max_seqlen_k=max(kv_lens),
            softmax_scale=scale, causal=True,
            window_size=(-1, -1), block_table=bt, softcap=0, fa_version=3,
        )
        ref = _ref_paged(q, k_cache, v_cache, cu_q, cu_k, bt, scale, causal=True)
        msg = f"max_diff={torch.max(torch.abs(out - ref))}"
        torch.testing.assert_close(out, ref, atol=2e-2, rtol=1e-2, msg=msg)


# ---------------------------------------------------------------------------
# paged: head_size variants
# ---------------------------------------------------------------------------

@pytest.mark.flash_attn_varlen_func
@_SKIP_GPU
@_SKIP_VENDOR
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("seq_lens", [[(1, 2048)] * 16])
@pytest.mark.parametrize("num_heads", [(16, 8)])
@pytest.mark.parametrize("block_size", [64])
@pytest.mark.parametrize("dtype", DTYPES)
def test_gluon_paged_head_size(head_size, seq_lens, num_heads, block_size, dtype):
    """paged: different head_size values (64/128/256)."""
    with torch.device(device):
        utils.init_seed(42)
        nq, nk  = num_heads
        scale   = head_size ** -0.5
        q_lens  = [s[0] for s in seq_lens]
        kv_lens = [s[1] for s in seq_lens]
        num_blocks = 4096

        q       = torch.randn(sum(q_lens),  nq, head_size, dtype=dtype)
        k_cache = torch.randn(num_blocks, block_size, nk, head_size, dtype=dtype)
        v_cache = torch.randn_like(k_cache)
        cu_q    = _cu(q_lens,  device)
        cu_k    = _cu(kv_lens, device)
        bt      = _block_tables(cu_k, block_size, num_blocks, device)

        out = flag_gems.ops.flash_attn_varlen_func(
            q=q, k=k_cache, v=v_cache,
            cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
            max_seqlen_q=max(q_lens), max_seqlen_k=max(kv_lens),
            softmax_scale=scale, causal=True,
            window_size=(-1, -1), block_table=bt, softcap=0, fa_version=3,
        )
        ref = _ref_paged(q, k_cache, v_cache, cu_q, cu_k, bt, scale, causal=True)
        msg = f"max_diff={torch.max(torch.abs(out - ref))}"
        torch.testing.assert_close(out, ref, atol=2e-2, rtol=1e-2, msg=msg)


# ---------------------------------------------------------------------------
# paged: block_size variants
# ---------------------------------------------------------------------------

@pytest.mark.flash_attn_varlen_func
@_SKIP_GPU
@_SKIP_VENDOR
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("seq_lens", [[(1, 2048)] * 16, [(64, 512)] * 4])
@pytest.mark.parametrize("num_heads", [(16, 8)])
@pytest.mark.parametrize("head_size", [128])
@pytest.mark.parametrize("dtype", DTYPES)
def test_gluon_paged_block_size(block_size, seq_lens, num_heads, head_size, dtype):
    """paged: different block_size values (64/128)."""
    with torch.device(device):
        utils.init_seed(42)
        nq, nk  = num_heads
        scale   = head_size ** -0.5
        q_lens  = [s[0] for s in seq_lens]
        kv_lens = [s[1] for s in seq_lens]
        num_blocks = max(2048, sum((kl + block_size - 1) // block_size
                                   for kl in kv_lens) + 64)

        q       = torch.randn(sum(q_lens),  nq, head_size, dtype=dtype)
        k_cache = torch.randn(num_blocks, block_size, nk, head_size, dtype=dtype)
        v_cache = torch.randn_like(k_cache)
        cu_q    = _cu(q_lens,  device)
        cu_k    = _cu(kv_lens, device)
        bt      = _block_tables(cu_k, block_size, num_blocks, device)

        out = flag_gems.ops.flash_attn_varlen_func(
            q=q, k=k_cache, v=v_cache,
            cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
            max_seqlen_q=max(q_lens), max_seqlen_k=max(kv_lens),
            softmax_scale=scale, causal=True,
            window_size=(-1, -1), block_table=bt, softcap=0, fa_version=3,
        )
        ref = _ref_paged(q, k_cache, v_cache, cu_q, cu_k, bt, scale, causal=True)
        msg = f"max_diff={torch.max(torch.abs(out - ref))}"
        torch.testing.assert_close(out, ref, atol=2e-2, rtol=1e-2, msg=msg)
