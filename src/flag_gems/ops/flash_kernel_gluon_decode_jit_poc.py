"""Minimal decode kernel for libtriton_jit C++ POC (no torch import)."""
import triton
import triton.language as tl

@triton.jit
def flash_varlen_decode_gluon_kernel_jit(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    lse_ptr,
    cu_seqlens_q_ptr,
    cu_seqlens_k_ptr,
    batch_ids_ptr,
    q_row_stride,
    q_head_stride,
    k_row_stride,
    v_row_stride,
    o_row_stride,
    o_head_stride,
    k_head_stride,
    v_head_stride,
    total_q,
    softmax_scale,
    is_causal: tl.constexpr,
    PACK_GQA: tl.constexpr,
    H_HK_RATIO: tl.constexpr,
    MAX_Q_LEN: tl.constexpr,
    BLOCK_N: tl.constexpr,
    D: tl.constexpr,
    D_PAD: tl.constexpr,
):
    """Non-paged decode: one CTA per (decode_batch, kv_head or q_head), tl.dot over KV."""
    pid_db = tl.program_id(0)
    pid_h = tl.program_id(1)

    bid = tl.load(batch_ids_ptr + pid_db).to(tl.int32)
    q_bos = tl.load(cu_seqlens_q_ptr + bid).to(tl.int32)
    q_eos = tl.load(cu_seqlens_q_ptr + bid + 1).to(tl.int32)
    q_len = q_eos - q_bos
    k_bos = tl.load(cu_seqlens_k_ptr + bid).to(tl.int32)
    k_eos = tl.load(cu_seqlens_k_ptr + bid + 1).to(tl.int32)
    k_len = k_eos - k_bos

    offs_d = tl.arange(0, D_PAD)
    mask_d = offs_d < D

    if PACK_GQA:
        kv_hid = pid_h
    else:
        q_head_fixed = pid_h
        kv_hid = q_head_fixed // H_HK_RATIO

    for qi in tl.static_range(MAX_Q_LEN):
        q_idx = qi
        if q_idx < q_len:
            if PACK_GQA:
                for qh in tl.static_range(H_HK_RATIO):
                    q_head = kv_hid * H_HK_RATIO + qh

                    q_ptrs = (
                        q_ptr
                        + (q_bos + q_idx) * q_row_stride
                        + q_head * q_head_stride
                        + offs_d
                    )
                    q = tl.load(q_ptrs, mask=mask_d, other=0.0).to(tl.float32)

                    m_i = -float("inf")
                    l_i = 0.0
                    acc = tl.zeros([D_PAD], dtype=tl.float32)

                    for start_n in tl.range(0, k_len, BLOCK_N):
                        cols = start_n + tl.arange(0, BLOCK_N)
                        mask_k = cols < k_len

                        k_ptrs = (
                            k_ptr
                            + (k_bos + cols[:, None]) * k_row_stride
                            + kv_hid * k_head_stride
                            + offs_d[None, :]
                        )
                        k = tl.load(
                            k_ptrs,
                            mask=mask_k[:, None] & mask_d[None, :],
                            other=0.0,
                        ).to(tl.float32)

                        qk = tl.sum(q[None, :] * k, axis=1) * softmax_scale
                        if is_causal:
                            causal_hi = q_idx + (k_len - q_len)
                            qk = tl.where(cols <= causal_hi, qk, -float("inf"))
                        qk = tl.where(mask_k, qk, -float("inf"))

                        m_ij = tl.max(qk, axis=0)
                        p = tl.exp(qk - m_ij)
                        l_ij = tl.sum(p, axis=0)
                        alpha = tl.exp(m_i - m_ij)

                        v_ptrs = (
                            v_ptr
                            + (k_bos + cols[:, None]) * v_row_stride
                            + kv_hid * v_head_stride
                            + offs_d[None, :]
                        )
                        v = tl.load(
                            v_ptrs,
                            mask=mask_k[:, None] & mask_d[None, :],
                            other=0.0,
                        ).to(tl.float32)

                        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
                        l_i = l_i * alpha + l_ij
                        m_i = m_ij

                    inv_l = tl.where(l_i > 0, 1.0 / l_i, 0.0)
                    o = (acc * inv_l).to(o_ptr.dtype.element_ty)
                    lse_val = m_i + tl.log(tl.where(l_i > 0, l_i, 1.0))

                    o_ptrs = (
                        o_ptr
                        + (q_bos + q_idx) * o_row_stride
                        + q_head * o_head_stride
                        + offs_d
                    )
                    tl.store(o_ptrs, o, mask=mask_d)

                    lse_ptrs = lse_ptr + q_head * total_q + q_bos + q_idx
                    tl.store(lse_ptrs, lse_val)
            else:
                q_head = q_head_fixed

                q_ptrs = (
                    q_ptr
                    + (q_bos + q_idx) * q_row_stride
                    + q_head * q_head_stride
                    + offs_d
                )
                q = tl.load(q_ptrs, mask=mask_d, other=0.0).to(tl.float32)

                m_i = -float("inf")
                l_i = 0.0
                acc = tl.zeros([D_PAD], dtype=tl.float32)

                for start_n in tl.range(0, k_len, BLOCK_N):
                    cols = start_n + tl.arange(0, BLOCK_N)
                    mask_k = cols < k_len

                    k_ptrs = (
                        k_ptr
                        + (k_bos + cols[:, None]) * k_row_stride
                        + kv_hid * k_head_stride
                        + offs_d[None, :]
                    )
                    k = tl.load(
                        k_ptrs,
                        mask=mask_k[:, None] & mask_d[None, :],
                        other=0.0,
                    ).to(tl.float32)

                    qk = tl.sum(q[None, :] * k, axis=1) * softmax_scale
                    if is_causal:
                        causal_hi = q_idx + (k_len - q_len)
                        qk = tl.where(cols <= causal_hi, qk, -float("inf"))
                    qk = tl.where(mask_k, qk, -float("inf"))

                    m_ij = tl.max(qk, axis=0)
                    p = tl.exp(qk - m_ij)
                    l_ij = tl.sum(p, axis=0)
                    alpha = tl.exp(m_i - m_ij)

                    v_ptrs = (
                        v_ptr
                        + (k_bos + cols[:, None]) * v_row_stride
                        + kv_hid * v_head_stride
                        + offs_d[None, :]
                    )
                    v = tl.load(
                        v_ptrs,
                        mask=mask_k[:, None] & mask_d[None, :],
                        other=0.0,
                    ).to(tl.float32)

                    acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
                    l_i = l_i * alpha + l_ij
                    m_i = m_ij

                inv_l = tl.where(l_i > 0, 1.0 / l_i, 0.0)
                o = (acc * inv_l).to(o_ptr.dtype.element_ty)
                lse_val = m_i + tl.log(tl.where(l_i > 0, l_i, 1.0))

                o_ptrs = (
                    o_ptr
                    + (q_bos + q_idx) * o_row_stride
                    + q_head * o_head_stride
                    + offs_d
                )
                tl.store(o_ptrs, o, mask=mask_d)

                lse_ptrs = lse_ptr + q_head * total_q + q_bos + q_idx
                tl.store(lse_ptrs, lse_val)
