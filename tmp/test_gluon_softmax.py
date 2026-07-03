import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.hopper import tma, mbarrier, fence_async_shared


@gluon.constexpr_function
def _dot_layout(num_warps):
    return gl.BlockedLayout([1, 1], [1, 32], [num_warps, 1], [1, 0])


@triton.jit
def softmax_rescale(O_acc, S, row_max, row_sum, softmax_scale_log2e: tl.constexpr, is_border: tl.constexpr):
    prev_max = row_max
    row_max = tl.maximum(row_max, tl.max(S, 1))
    cur_max = tl.where(row_max == float("-inf"), 0, row_max) if is_border else row_max
    p_scale = tl.math.exp2((prev_max - cur_max) * softmax_scale_log2e)
    row_sum *= p_scale
    O_acc *= p_scale[:, None]
    max_scaled = tl.where(row_max == float("-inf"), 0, row_max * softmax_scale_log2e)
    P = tl.math.exp2(S * softmax_scale_log2e - max_scaled[:, None])
    row_sum = row_sum + tl.sum(P, 1)
    return O_acc, P, row_max, row_sum


@gluon.jit
def test_kernel(q_ptr, k_ptr, q_row_stride, k_row_stride, q_len, k_len,
                d: gl.constexpr, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,
                num_warps: gl.constexpr):
    dtype: gl.constexpr = q_ptr.dtype.element_ty
    d_blk: gl.constexpr = 64
    dot_layout: gl.constexpr = _dot_layout(num_warps)
    row_layout: gl.constexpr = gl.SliceLayout(1, dot_layout)
    kv_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for([BLOCK_N, d_blk], dtype)
    q_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for([BLOCK_M, d_blk], dtype)

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
    acc_ = gl.zeros((BLOCK_M, d_blk), dtype=gl.float32, layout=dot_layout)
    rowmax_ = gl.full([BLOCK_M], float("-inf"), dtype=gl.float32, layout=row_layout)
    rowsum_ = gl.zeros([BLOCK_M], dtype=gl.float32, layout=row_layout)
    acc_, P, rowmax_, rowsum_ = softmax_rescale(acc_, S, rowmax_, rowsum_, 1.0, True)


if __name__ == "__main__":
    q = torch.randn(64, 64, dtype=torch.float16, device="cuda")
    k = torch.randn(64, 64, dtype=torch.float16, device="cuda")
    test_kernel[(1,)](q, k, q.stride(0), k.stride(0), 64, 64, 64, 64, 64, 4)
    print("ok")