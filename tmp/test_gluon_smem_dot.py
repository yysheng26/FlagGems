import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.hopper import tma, mbarrier, fence_async_shared


@gluon.constexpr_function
def _dot_layout(num_warps):
    return gl.BlockedLayout([1, 1], [1, 32], [num_warps, 1], [1, 0])


@gluon.constexpr_function
def _d_blk(d):
    return 64


@gluon.constexpr_function
def _kv_layout(block_n, d, dtype):
    return gl.NVMMASharedLayout.get_default_for([block_n, _d_blk(d)], dtype)


@gluon.jit
def _tma_load(desc, off0, off1, smem, bar, nbytes, dot_layout):
    mbarrier.init(bar, count=1)
    mbarrier.expect(bar, nbytes)
    tma.async_copy_global_to_shared(desc, [off0, off1], bar, smem)
    mbarrier.wait(bar, phase=0)
    mbarrier.invalidate(bar)
    fence_async_shared()
    return smem.load(dot_layout)


@gluon.jit
def test_kernel(q_ptr, k_ptr, q_row_stride, k_row_stride, q_len, k_len,
                d: gl.constexpr, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,
                num_warps: gl.constexpr):
    dtype: gl.constexpr = q_ptr.dtype.element_ty
    d_blk: gl.constexpr = _d_blk(d)
    dot_layout: gl.constexpr = _dot_layout(num_warps)
    kv_layout: gl.constexpr = _kv_layout(BLOCK_N, d, dtype)
    q_layout: gl.constexpr = _kv_layout(BLOCK_M, d, dtype)
    nbytes: gl.constexpr = BLOCK_M * d_blk * 2

    q_smem = gl.allocate_shared_memory(dtype, [BLOCK_M, d_blk], q_layout)
    k_smem = gl.allocate_shared_memory(dtype, [BLOCK_N, d_blk], kv_layout)
    bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())

    desc_q = tma.make_tensor_descriptor(
        q_ptr, shape=[q_len, d], strides=[q_row_stride, 1],
        block_shape=[BLOCK_M, d_blk], layout=q_layout,
    )
    desc_k = tma.make_tensor_descriptor(
        k_ptr, shape=[k_len, d], strides=[k_row_stride, 1],
        block_shape=[BLOCK_N, d_blk], layout=kv_layout,
    )
    bQ = _tma_load(desc_q, 0, 0, q_smem, bar, nbytes, dot_layout)
    bK_raw = _tma_load(desc_k, 0, 0, k_smem, bar, BLOCK_N * d_blk * 2, dot_layout)
    bK = tl.trans(bK_raw)
    S = tl.dot(bQ, bK, out_dtype=tl.float32)
    o_ptr = q_ptr
    tl.store(o_ptr, S.to(dtype))


if __name__ == "__main__":
    q = torch.randn(64, 64, dtype=torch.float16, device="cuda")
    k = torch.randn(64, 64, dtype=torch.float16, device="cuda")
    test_kernel[(1,)](q, k, q.stride(0), k.stride(0), 64, 64, 64, 64, 64, 4)
    print("ok")