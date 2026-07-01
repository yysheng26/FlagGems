"""FA3 Gluon via C++ extension (step 4).

Run:
  USE_C_EXTENSION=1 CUDA_VISIBLE_DEVICES=2 \\
  FLAGGEMS_SOURCE_DIR=/path/to/src/flag_gems \\
  PYTHONPATH=/path/to/src \\
  pytest tests/test_flash_attn_varlen_fa3_gluon_cext.py -v --quick
"""
import os

import pytest
import torch

import flag_gems
from flag_gems.config import has_c_extension, use_c_extension

from . import accuracy_utils as utils
from .conftest import QUICK_MODE

device = flag_gems.device

_SKIP_NO_CEXT = pytest.mark.skipif(not has_c_extension, reason="C extension not built")
_SKIP_NO_USE_CEXT = pytest.mark.skipif(not use_c_extension, reason="Set USE_C_EXTENSION=1")
_SKIP_GPU = pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9),
    reason="Gluon requires Hopper GPU (sm_90+)",
)


def _cu(lens, dev):
    t = torch.zeros(len(lens) + 1, dtype=torch.int32, device=dev)
    torch.cumsum(torch.tensor(lens, dtype=torch.int32, device=dev), 0, out=t[1:])
    return t


def _seqused(lens, dev):
    return torch.tensor(list(lens), dtype=torch.int32, device=dev)


def _block_tables(kv_lens, block_size, num_blocks, dev):
    max_pgs = max((int(kl) + block_size - 1) // block_size for kl in kv_lens)
    return torch.randint(0, num_blocks, (len(kv_lens), max_pgs), dtype=torch.int32, device=dev)


def _ref_varlen(q, k, v, cu_q, cu_k, scale, causal):
    s = cu_q.numel() - 1
    ratio = q.shape[1] // k.shape[1]
    out = torch.empty_like(q)
    for i in range(s):
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


def _ref_paged(q, k_cache, v_cache, cu_q, seqused_k, block_tables, scale, causal):
    s = cu_q.numel() - 1
    block_size = k_cache.shape[1]
    ratio = q.shape[1] // k_cache.shape[2]
    out = torch.empty_like(q)
    for i in range(s):
        qs, qe = int(cu_q[i]), int(cu_q[i + 1])
        kl = int(seqused_k[i])
        npages = (kl + block_size - 1) // block_size
        pids = block_tables[i, :npages]
        kd = k_cache[pids].reshape(-1, k_cache.shape[2], k_cache.shape[3])[:kl]
        vd = v_cache[pids].reshape(-1, v_cache.shape[2], v_cache.shape[3])[:kl]
        qi = q[qs:qe].float()
        ki, vi = kd.float(), vd.float()
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


@pytest.mark.flash_attn_varlen_func
@_SKIP_NO_CEXT
@_SKIP_NO_USE_CEXT
@_SKIP_GPU
def test_fa3_gluon_cext_non_paged():
    """use_c_extension + fa_version=3 non-paged."""
    with torch.device(device):
        utils.init_seed(20260630)
        head_size = 128
        scale = head_size ** -0.5
        q_lens, kv_lens = [64], [64]
        nq, nk = 16, 8
        q = torch.randn(sum(q_lens), nq, head_size, dtype=torch.float16)
        k = torch.randn(sum(kv_lens), nk, head_size, dtype=torch.float16)
        v = torch.randn_like(k)
        cu_q = _cu(q_lens, device)
        cu_k = _cu(kv_lens, device)
        out = flag_gems.ops.flash_attn_varlen_func(
            q=q, k=k, v=v,
            cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
            max_seqlen_q=max(q_lens), max_seqlen_k=max(kv_lens),
            softmax_scale=scale, causal=True,
            window_size=(-1, -1), softcap=0, fa_version=3,
        )
        ref = _ref_varlen(q, k, v, cu_q, cu_k, scale, causal=True)
        torch.testing.assert_close(out, ref, atol=2e-2, rtol=1e-2)


@pytest.mark.flash_attn_varlen_func
@_SKIP_NO_CEXT
@_SKIP_NO_USE_CEXT
@_SKIP_GPU
def test_fa3_gluon_cext_paged():
    """use_c_extension + fa_version=3 paged."""
    with torch.device(device):
        utils.init_seed(20260630)
        head_size = 128
        block_size = 64
        scale = head_size ** -0.5
        q_lens, kv_lens = [64, 32], [128, 96]
        nq, nk = 16, 8
        num_blocks = 512
        q = torch.randn(sum(q_lens), nq, head_size, dtype=torch.float16)
        k_cache = torch.randn(num_blocks, block_size, nk, head_size, dtype=torch.float16)
        v_cache = torch.randn_like(k_cache)
        cu_q = _cu(q_lens, device)
        sk = _seqused(kv_lens, device)
        bt = _block_tables(kv_lens, block_size, num_blocks, device)
        out = flag_gems.ops.flash_attn_varlen_func(
            q=q, k=k_cache, v=v_cache,
            cu_seqlens_q=cu_q, seqused_k=sk,
            max_seqlen_q=max(q_lens), max_seqlen_k=max(kv_lens),
            softmax_scale=scale, causal=True,
            window_size=(-1, -1), block_table=bt, softcap=0, fa_version=3,
        )
        ref = _ref_paged(q, k_cache, v_cache, cu_q, sk, bt, scale, causal=True)
        torch.testing.assert_close(out, ref, atol=2e-2, rtol=1e-2)


@pytest.mark.flash_attn_varlen_func
@_SKIP_NO_CEXT
@_SKIP_GPU
def test_fa3_gluon_torch_ops_direct():
    """torch.ops.flag_gems.flash_attn_varlen_func (no USE_C_EXTENSION gate)."""
    if not hasattr(torch.ops.flag_gems, "flash_attn_varlen_func"):
        pytest.skip("torch.ops.flag_gems.flash_attn_varlen_func not registered")
    with torch.device(device):
        utils.init_seed(42)
        head_size = 128
        scale = head_size ** -0.5
        q_lens, kv_lens = [1, 1], [64, 128]
        nq, nk = 16, 8
        q = torch.randn(sum(q_lens), nq, head_size, dtype=torch.float16)
        k = torch.randn(sum(kv_lens), nk, head_size, dtype=torch.float16)
        v = torch.randn_like(k)
        cu_q = _cu(q_lens, device)
        cu_k = _cu(kv_lens, device)
        out, _ = torch.ops.flag_gems.flash_attn_varlen_func(
            q, k, v,
            max(q_lens), cu_q, max(kv_lens),
            cu_k, None, None,
            0.0, scale, True,
            None, 0.0, None,
            False, False, None,
            False, None, None,
            None, None, None, None,
            0, 1, 0, None, 3,
        )
        ref = _ref_varlen(q, k, v, cu_q, cu_k, scale, causal=False)
        torch.testing.assert_close(out, ref, atol=2e-2, rtol=1e-2)