import torch
import triton
import triton.language as tl
from triton.experimental.gluon.language.nvidia.hopper import tma, mbarrier, fence_async_shared


@triton.jit
def test_tma_desc_kernel(
    q_ptr,
    q_row_stride,
    q_len,
    d: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    m_block = tl.program_id(0)
    # runtime base like varlen
    q_base = q_ptr + m_block * BLOCK_M * q_row_stride
    desc_q = tma.make_tensor_descriptor(
        q_base,
        shape=[q_len, d],
        strides=[q_row_stride, 1],
        block_shape=[BLOCK_M, BLOCK_K],
    )
    tile = desc_q.load([m_block * BLOCK_M, 0])
    tl.device_assert(tl.sum(tile) == tl.sum(tile))


if __name__ == "__main__":
    q = torch.randn(128, 64, dtype=torch.float16, device="cuda")
    test_tma_desc_kernel[(1,)](q, q.stride(0), 128, 64, 64, 64)
    print("tma desc in triton jit ok")