import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.hopper import tma, mbarrier, fence_async_shared


@gluon.constexpr_function
def _dot_layout(num_warps):
    return gl.BlockedLayout([1, 1], [1, 32], [num_warps, 1], [1, 0])


@gluon.jit
def test_kernel(q_ptr, k_ptr, q_row_stride, k_row_stride, q_len, k_len,
                d: gl.constexpr, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,
                num_warps: gl.constexpr):
    dtype: gl.constexpr = q_ptr.dtype.element_ty
    d_blk: gl.constexpr = 64
    dot_layout: gl.constexpr = _dot_layout(num_warps)
    qk_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_N], gl.float32)
    kv_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for([BLOCK_N, d_blk], dtype)
    q_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for([BLOCK_M, d_blk], dtype)

    q_smem = gl.allocate_shared_memory(dtype, [BLOCK_M, d_blk], q_layout)
    k_smem = gl.allocate_shared_memory(dtype, [BLOCK_N, d_blk], kv_layout)
    s_smem = gl.allocate_shared_memory(gl.float32, [BLOCK_M, BLOCK_N], qk_layout)
    bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())

    desc_q = tma.make_tensor_descriptor(
        q_ptr, shape=[q_len, d], strides=[q_row_stride, 1],
        block_shape=[BLOCK_M, d_blk], layout=q_layout,
    )
    desc_k = tma.make_tensor_descriptor(
        k_ptr, shape=[k_len, d], strides=[k_row_stride, 1],
        block_shape=[BLOCK_N, d_blk], layout=kv_layout,
    )
    mbarrier.init(bar, count=1)
    mbarrier.expect(bar, BLOCK_M * d_blk * 2)
    tma.async_copy_global_to_shared(desc_q, [0, 0], bar, q_smem)
    mbarrier.wait(bar, phase=0)
    mbarrier.invalidate(bar)
    bQ = q_smem.load(dot_layout)

    mbarrier.init(bar, count=1)
    mbarrier.expect(bar, BLOCK_N * d_blk * 2)
    tma.async_copy_global_to_shared(desc_k, [0, 0], bar, k_smem)
    mbarrier.wait(bar, phase=0)
    mbarrier.invalidate(bar)
    bK = tl.trans(k_smem.load(dot_layout))

    S = tl.dot(bQ, bK, out_dtype=tl.float32)
    s_smem.store(S)
    fence_async_shared()
    S2 = s_smem.load(dot_layout)
    row_layout: gl.constexpr = gl.SliceLayout(1, dot_layout)
    m = gl.max(S2, axis=1)


if __name__ == "__main__":
    q = torch.randn(64, 64, dtype=torch.float16, device="cuda")
    k = torch.randn(64, 64, dtype=torch.float16, device="cuda")
    test_kernel[(1,)](q, k, q.stride(0), k.stride(0), 64, 64, 64, 64, 64, 4)
    print("ok")