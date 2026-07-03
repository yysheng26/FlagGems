import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@gluon.constexpr_function
def _dot_layout(num_warps):
    return gl.BlockedLayout([1, 1], [1, 32], [num_warps, 1], [1, 0])


@triton.jit
def apply_mask(
    S, col_idx, row_idx, max_seqlen_q, max_seqlen_k,
    window_size_left, window_size_right,
    is_even_mn: tl.constexpr, is_causal: tl.constexpr, is_local: tl.constexpr,
):
    need_mask = is_causal | is_local | (not is_even_mn)
    if need_mask:
        col_rb = tl.minimum(
            max_seqlen_k - 1,
            row_idx + max_seqlen_k - max_seqlen_q + window_size_right,
        )
        if is_causal:
            S = tl.where(col_idx[None, :] > col_rb[:, None], float("-inf"), S)
    return S


@gluon.jit
def test_kernel(q_ptr, k_ptr, q_row_stride, k_row_stride, q_len, k_len,
                d: gl.constexpr, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,
                BLOCK_K: gl.constexpr, num_warps: gl.constexpr):
    dot_layout: gl.constexpr = _dot_layout(num_warps)
    row_layout: gl.constexpr = gl.SliceLayout(1, dot_layout)
    col_layout: gl.constexpr = gl.SliceLayout(0, dot_layout)
    m_block = 0
    n_block = 0
    gQ = tl.make_block_ptr(
        base=q_ptr, shape=(q_len, d), strides=(q_row_stride, 1),
        offsets=(0, 0), block_shape=(BLOCK_M, BLOCK_K), order=(1, 0),
    )
    gK = tl.make_block_ptr(
        base=k_ptr, shape=(k_len, d), strides=(k_row_stride, 1),
        offsets=(0, 0), block_shape=(BLOCK_N, BLOCK_K), order=(0, 1),
    )
    bQ = tl.load(gQ, boundary_check=(0, 1))
    bK = tl.trans(tl.load(gK, boundary_check=(0, 1)))
    S = tl.dot(bQ, bK, out_dtype=tl.float32)
    row_idx = m_block * BLOCK_M + gl.arange(0, BLOCK_M, row_layout)
    col_idx = n_block * BLOCK_N + gl.arange(0, BLOCK_N, col_layout)
    S = apply_mask(S, col_idx, row_idx, q_len, k_len, -1, 0, False, True, False)
    tl.device_assert(tl.sum(S) == tl.sum(S))


if __name__ == "__main__":
    q = torch.randn(64, 64, dtype=torch.float16, device="cuda")
    k = torch.randn(64, 64, dtype=torch.float16, device="cuda")
    test_kernel[(1,)](q, k, q.stride(0), k.stride(0), 64, 64, 64, 64, 64, 64, 4)
    print("ok")