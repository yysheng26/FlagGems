import torch
import triton
import triton.language as tl


@triton.jit
def test_kernel(q_ptr, k_ptr, cu_k_ptr, q_row_stride, k_row_stride, bid,
                d: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    k_bos = tl.load(cu_k_ptr + bid).to(tl.int32)
    k_eos = tl.load(cu_k_ptr + bid + 1).to(tl.int32)
    k_len = k_eos - k_bos
    q_len = k_len

    desc_q = tl.make_tensor_descriptor(
        q_ptr, shape=[q_len, d], strides=[q_row_stride, 1],
        block_shape=[BLOCK_M, BLOCK_K],
    )
    k_ptr_seq = k_ptr + k_bos * k_row_stride
    desc_k = tl.make_tensor_descriptor(
        k_ptr_seq, shape=[k_len, d], strides=[k_row_stride, 1],
        block_shape=[BLOCK_N, BLOCK_K],
    )
    bQ = desc_q.load([0, 0])
    bK = tl.trans(desc_k.load([0, 0]))
    S = tl.dot(bQ, bK, out_dtype=tl.float32)
    tl.store(q_ptr, S.to(q_ptr.dtype.element_ty))


if __name__ == "__main__":
    q = torch.randn(128, 64, dtype=torch.float16, device="cuda")
    k = torch.randn(256, 64, dtype=torch.float16, device="cuda")
    cu_k = torch.tensor([0, 128, 256], dtype=torch.int32, device="cuda")
    test_kernel[(1,)](q, k, cu_k, q.stride(0), k.stride(0), 0, 64, 64, 64, 64)
    print("ok")