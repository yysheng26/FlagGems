"""
flash_kernel_gluon.py — FA3 varlen forward on Hopper via Triton Gluon.

Why Gluon over standard Triton for varlen?
------------------------------------------
Triton FA3 disables warp_specialize when cu_seqlens_k is present because the
K/V base pointer (k_ptr + k_bos * stride) is a runtime value derived from
cu_seqlens_k[bid].  Triton's TaskIdPropagation pass cannot track runtime
pointers across the warp-specialization boundary, so NVGPUWarpSpecialization
fails and the kernel falls back to a non-pipelined path.

Gluon's tma.make_tensor_descriptor runs on-device at runtime, accepting any
pointer or shape computed from register values.  This lets the load partition
build its TMA descriptor from k_bos (loaded from cu_seqlens_k inside the
kernel) and issue fully-pipelined TMA loads for every sequence in the batch.

Structure
---------
Phase 1  flash_varlen_fwd_p1_kernel
    Single entry point, no warp_specialize.  All warps cooperate:
    warp 0 issues TMA loads; all warps wait on mbarrier then run WGMMA.
    Used to validate correctness before adding the pipeline.

Phase 2  flash_varlen_fwd_p2_kernel
    gl.warp_specialize splits the CTA into:
      load_partition  (1 warp)  — builds desc_k/desc_v at runtime, issues TMA
      compute_partition (4 warps, 1 warp-group) — WGMMA QKᵀ, softmax, WGMMA PV
    num_warps for the entry kernel must be 5 (1 load + 4 compute).

Tensor layouts (varlen packed format)
--------------------------------------
  q, o   : [total_q, h,  d]
  k, v   : [total_k, hk, d]
  softmax_lse : [h, total_q]
  cu_seqlens_q/k : [batch+1]  int32, prefix-sum of per-sequence lengths
"""

import math
import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.hopper import (
    tma,
    mbarrier,
    fence_async_shared,
    warpgroup_mma,
    warpgroup_mma_wait,
)
from triton.language.core import _aggregate as aggregate


# ---------------------------------------------------------------------------
# Compile-time helpers
# ---------------------------------------------------------------------------

@gluon.constexpr_function
def _tile_nbytes(rows, cols, elem_bits):
    return rows * cols * (elem_bits // 8)


@gluon.constexpr_function
def _pick_warps_per_cta(BLOCK_M, BLOCK_N, num_warps):
    """Expand [4,1] atom along M then N until product == num_warps."""
    wpc = [4, 1]
    m = 16
    while wpc[0] * wpc[1] != num_warps:
        if BLOCK_M > m * wpc[0]:
            wpc[0] *= 2
        else:
            wpc[1] *= 2
    return wpc


@gluon.constexpr_function
def _pick_instr_n(BLOCK_M, BLOCK_N, num_warps):
    """Largest instr_shape[1] that divides BLOCK_N and fits in per-warp budget."""
    m = 16
    mReps = triton.cdiv(BLOCK_M, m)
    nReps = triton.cdiv(num_warps, mReps)
    maxN = max(BLOCK_N // nReps, 8)
    n = 256
    while n > maxN or BLOCK_N % n != 0:
        n -= 8
    return n


# ---------------------------------------------------------------------------
# Shared data structures for the warp_specialize pipeline (Phase 2)
# ---------------------------------------------------------------------------

@aggregate
class BarrierCounter:
    """Circular index + phase bit for multi-stage pipeline."""
    index: gl.tensor
    phase: gl.tensor
    num_barriers: gl.constexpr

    @gluon.constexpr_function
    def __init__(self, index, phase, num_barriers):
        self.index = index
        self.phase = phase
        self.num_barriers = gl.constexpr(num_barriers)

    @gluon.must_use_result
    @gluon.jit
    def increment(self):
        if self.num_barriers == 1:
            return BarrierCounter(gl.to_tensor(0), self.phase ^ 1, self.num_barriers)
        next_index = self.index + 1
        rollover = next_index == self.num_barriers
        new_index = gl.where(rollover, 0, next_index)
        new_phase = gl.where(rollover, self.phase ^ 1, self.phase)
        return BarrierCounter(new_index, new_phase, self.num_barriers)


@aggregate
class KVChannel:
    """Multi-stage double buffer for K and V tiles between producer and consumer."""
    k_smem: gl.shared_memory_descriptor
    v_smem: gl.shared_memory_descriptor
    ready_bars: gl.shared_memory_descriptor   # producer signals data ready
    empty_bars: gl.shared_memory_descriptor   # consumer signals slot free
    num_stages: gl.constexpr

    @gluon.constexpr_function
    def __init__(self, k_smem, v_smem, ready_bars, empty_bars, num_stages):
        self.k_smem = k_smem
        self.v_smem = v_smem
        self.ready_bars = ready_bars
        self.empty_bars = empty_bars
        self.num_stages = gl.constexpr(num_stages)

    @gluon.jit
    def alloc(
        BLOCK_N: gl.constexpr,
        d: gl.constexpr,
        dtype: gl.constexpr,
        kv_layout: gl.constexpr,
        num_stages: gl.constexpr,
    ):
        k_smem = gl.allocate_shared_memory(dtype, [num_stages, BLOCK_N, d], kv_layout)
        v_smem = gl.allocate_shared_memory(dtype, [num_stages, BLOCK_N, d], kv_layout)
        ready_bars = gl.allocate_shared_memory(
            gl.int64, [num_stages, 1], mbarrier.MBarrierLayout())
        empty_bars = gl.allocate_shared_memory(
            gl.int64, [num_stages, 1], mbarrier.MBarrierLayout())
        for i in gl.static_range(num_stages):
            mbarrier.init(ready_bars.index(i), count=1)
            mbarrier.init(empty_bars.index(i), count=1)
            # Pre-arrive empty barriers so producer can start immediately.
            mbarrier.arrive(empty_bars.index(i), count=1)
        return KVChannel(k_smem, v_smem, ready_bars, empty_bars, num_stages)

    @gluon.jit
    def release(self):
        self.k_smem._keep_alive()
        self.v_smem._keep_alive()
        for i in gl.static_range(self.num_stages):
            mbarrier.invalidate(self.ready_bars.index(i))
            mbarrier.invalidate(self.empty_bars.index(i))


# ---------------------------------------------------------------------------
# Online softmax helper (shared by Phase 1 and Phase 2)
# ---------------------------------------------------------------------------

@gluon.jit
def _online_softmax_step(
    acc, rowmax, rowsum, qk,
    scale_log2e,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
):
    """
    One iteration of the online softmax update (log2-based).

    Numerically stable: when all scores in a row are -inf (fully masked),
    rowmax stays -inf and we substitute 0 for -inf before exp2 so that
    exp2(-inf - 0) = 0 rather than exp2(nan).
    """
    new_max = gl.max(qk, axis=1)                        # [BLOCK_M]
    new_max = gl.maximum(rowmax, new_max)
    safe_max = gl.where(new_max == float("-inf"),
                        gl.zeros_like(new_max), new_max)

    # Rescale running accumulator and rowsum by exp2(old_max - new_max).
    alpha = gl.exp2((rowmax - safe_max) * scale_log2e)  # [BLOCK_M]
    acc_row_layout: gl.constexpr = gl.SliceLayout(1, acc.type.layout)
    alpha_acc = gl.convert_layout(alpha, acc_row_layout)
    acc    = acc * alpha_acc[:, None]
    rowsum = rowsum * alpha

    # Compute attention weights for this KV block.
    P      = gl.exp2((qk - safe_max[:, None]) * scale_log2e)   # [BLOCK_M, BLOCK_N]
    rowsum = rowsum + gl.sum(P, axis=1)
    rowmax = new_max
    return acc, rowmax, rowsum, P


# ---------------------------------------------------------------------------
# Phase 1: single entry point (no warp_specialize), for correctness testing
# ---------------------------------------------------------------------------

@gluon.jit
def _p1_main_loop(
    q_smem,                 # already loaded Q tile in smem
    softmax_lse_ptr,
    o_smem_out,
    desc_o,
    cu_seqlens_k_ptr,
    k_ptr, v_ptr,
    k_row_stride, v_row_stride,
    k_head_stride, v_head_stride,
    q_bos, q_len,
    h_hk_ratio,
    d:           gl.constexpr,
    scale_log2e: gl.constexpr,
    is_causal:   gl.constexpr,
    BLOCK_M:     gl.constexpr,
    BLOCK_N:     gl.constexpr,
    num_warps:   gl.constexpr,
    dtype:       gl.constexpr,
    qk_layout:   gl.constexpr,
    pv_layout:   gl.constexpr,
    kv_layout:   gl.constexpr,
    p_smem_layout: gl.constexpr,
    pv_a_layout: gl.constexpr,
    row_layout:  gl.constexpr,
    total_q,
    m_block, bid, hid,
):
    k_bos = gl.load(cu_seqlens_k_ptr + bid).to(gl.int32)
    k_eos = gl.load(cu_seqlens_k_ptr + bid + 1).to(gl.int32)
    k_len = k_eos - k_bos

    kv_hid = hid // h_hk_ratio
    k_seq = k_ptr + k_bos * k_row_stride + kv_hid * k_head_stride
    v_seq = v_ptr + k_bos * v_row_stride + kv_hid * v_head_stride

    desc_k = tma.make_tensor_descriptor(
        k_seq, shape=[k_len, d], strides=[k_row_stride, 1],
        block_shape=[BLOCK_N, d], layout=kv_layout)
    desc_v = tma.make_tensor_descriptor(
        v_seq, shape=[k_len, d], strides=[v_row_stride, 1],
        block_shape=[BLOCK_N, d], layout=kv_layout)

    k_smem = gl.allocate_shared_memory(dtype, [BLOCK_N, d], kv_layout)
    v_smem = gl.allocate_shared_memory(dtype, [BLOCK_N, d], kv_layout)
    bar_kv = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    p_smem = gl.allocate_shared_memory(dtype, [BLOCK_M, BLOCK_N], p_smem_layout)

    acc    = gl.zeros([BLOCK_M, d],   dtype=gl.float32, layout=pv_layout)
    rowmax = gl.full([BLOCK_M],  float("-inf"), dtype=gl.float32, layout=row_layout)
    rowsum = gl.zeros([BLOCK_M], dtype=gl.float32, layout=row_layout)

    n_block_max = gl.cdiv(k_len, BLOCK_N)
    if is_causal:
        causal_limit = gl.cdiv(
            (m_block + 1) * BLOCK_M + k_len - q_len, BLOCK_N)
        n_block_max = gl.minimum(n_block_max, causal_limit)

    # Iterate KV blocks in reverse so causal skipping naturally trims the tail.
    for n_block in range(n_block_max - 1, -1, -1):
        start_n = n_block * BLOCK_N

        # -- Load K and V tile via TMA (warp 0 issues, all warps wait) --------
        nbytes_kv: gl.constexpr = _tile_nbytes(BLOCK_N, d, dtype.primitive_bitwidth) * 2
        mbarrier.init(bar_kv, count=1)
        mbarrier.expect(bar_kv, nbytes_kv)
        if gl.warp_id() == 0:
            tma.async_copy_global_to_shared(desc_k, [start_n, 0], bar_kv, k_smem)
            tma.async_copy_global_to_shared(desc_v, [start_n, 0], bar_kv, v_smem)
        mbarrier.wait(bar_kv, phase=0)
        mbarrier.invalidate(bar_kv)

        # -- QKᵀ via WGMMA ----------------------------------------------------
        fence_async_shared()
        kt_smem = k_smem.permute((1, 0))   # [d, BLOCK_N] for the K^T operand
        qk = warpgroup_mma(
            q_smem, kt_smem,
            gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=qk_layout),
            is_async=True, use_acc=False)
        qk = warpgroup_mma_wait(0, (qk,))

        # -- Masking ----------------------------------------------------------
        col_idx  = start_n + gl.arange(0, BLOCK_N)
        row_idx  = m_block * BLOCK_M + gl.arange(0, BLOCK_M)
        # padding mask: valid columns only
        qk = gl.where(col_idx[None, :] < k_len,  qk, float("-inf"))
        # padding mask: valid rows only (last Q tile may be partial)
        qk = gl.where(row_idx[:, None] < q_len,   qk, float("-inf"))
        if is_causal:
            # right-aligned causal: col <= row + (k_len - q_len)
            row_limit = row_idx + (k_len - q_len)
            qk = gl.where(col_idx[None, :] <= row_limit[:, None], qk, float("-inf"))

        # -- Online softmax + acc rescale -------------------------------------
        acc, rowmax, rowsum, P = _online_softmax_step(
            acc, rowmax, rowsum, qk, scale_log2e, BLOCK_M, BLOCK_N)

        # -- P smem round-trip (required: qk_layout ≠ pv_layout parent) ------
        P_typed = P.to(dtype)
        p_smem.store(P_typed)
        fence_async_shared()
        P_dot = p_smem.load(pv_a_layout)

        # -- PV via WGMMA -----------------------------------------------------
        acc = warpgroup_mma(P_dot, v_smem, acc, is_async=True, use_acc=True)
        acc = warpgroup_mma_wait(0, (acc,))

    # -- Epilogue: normalize, store O, write softmax_lse ----------------------
    pv_row_layout: gl.constexpr = gl.SliceLayout(1, pv_layout)
    rowsum_pv = gl.convert_layout(rowsum, pv_row_layout)
    inv_sum = gl.where(
        (rowsum_pv == 0.0) | (rowsum_pv != rowsum_pv),
        gl.full([BLOCK_M], 1.0, dtype=gl.float32, layout=pv_row_layout),
        1.0 / rowsum_pv,
    )
    out = (acc * inv_sum[:, None]).to(dtype)
    o_smem_out.store(out)
    fence_async_shared()
    tma.async_copy_shared_to_global(desc_o, [m_block * BLOCK_M, 0], o_smem_out)
    tma.store_wait(pendings=0)

    # softmax_lse: [h, total_q] layout, log(sum(exp(...))) in natural log
    lse_val = gl.where(
        (rowsum == 0.0) | (rowsum != rowsum),
        gl.full([BLOCK_M], float("inf"), dtype=gl.float32, layout=row_layout),
        rowmax / scale_log2e + gl.log(rowsum) / scale_log2e,
    )
    lse_ptr  = softmax_lse_ptr + hid * total_q + q_bos + m_block * BLOCK_M
    lse_mask = gl.arange(0, BLOCK_M) < (q_len - m_block * BLOCK_M)
    gl.store(lse_ptr + gl.arange(0, BLOCK_M), lse_val, mask=lse_mask)


@gluon.jit
def flash_varlen_fwd_p1_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr,
    softmax_lse_ptr,
    cu_seqlens_q_ptr,
    cu_seqlens_k_ptr,
    q_row_stride, k_row_stride, v_row_stride, o_row_stride,
    q_head_stride, k_head_stride, v_head_stride, o_head_stride,
    total_q,
    h_hk_ratio:  gl.constexpr,
    d:           gl.constexpr,
    scale_log2e: gl.constexpr,
    is_causal:   gl.constexpr,
    BLOCK_M:     gl.constexpr,
    BLOCK_N:     gl.constexpr,
    num_warps:   gl.constexpr,
):
    """
    Phase 1 kernel — no warp_specialize, single warp-group (num_warps=4).

    Grid: (cdiv(max_seqlen_q, BLOCK_M), batch, num_heads)
    """
    m_block = gl.program_id(0)
    bid     = gl.program_id(1)
    hid     = gl.program_id(2)

    q_bos = gl.load(cu_seqlens_q_ptr + bid).to(gl.int32)
    q_eos = gl.load(cu_seqlens_q_ptr + bid + 1).to(gl.int32)
    q_len = q_eos - q_bos

    if m_block * BLOCK_M >= q_len:
        return

    q_seq  = q_ptr + q_bos * q_row_stride + hid * q_head_stride
    o_seq  = o_ptr + q_bos * o_row_stride + hid * o_head_stride

    dtype: gl.constexpr      = q_ptr.dtype.element_ty
    q_layout:  gl.constexpr  = gl.NVMMASharedLayout.get_default_for([BLOCK_M, d], dtype)
    kv_layout: gl.constexpr  = gl.NVMMASharedLayout.get_default_for([BLOCK_N, d], dtype)
    o_layout:  gl.constexpr  = gl.NVMMASharedLayout.get_default_for([BLOCK_M, d], dtype)

    # WGMMA accumulator layouts
    qk_wpc: gl.constexpr = _pick_warps_per_cta(BLOCK_M, BLOCK_N, num_warps)
    pv_wpc: gl.constexpr = _pick_warps_per_cta(BLOCK_M, d, num_warps)
    qk_instr_n: gl.constexpr = _pick_instr_n(BLOCK_M, BLOCK_N, num_warps)
    pv_instr_n: gl.constexpr = _pick_instr_n(BLOCK_M, d, num_warps)
    qk_layout: gl.constexpr = gl.NVMMADistributedLayout(
        version=[3, 0],
        warps_per_cta=qk_wpc,
        instr_shape=[16, qk_instr_n, 256 // dtype.primitive_bitwidth],
    )
    pv_layout: gl.constexpr = gl.NVMMADistributedLayout(
        version=[3, 0],
        warps_per_cta=pv_wpc,
        instr_shape=[16, pv_instr_n, 256 // dtype.primitive_bitwidth],
    )
    row_layout: gl.constexpr = gl.SliceLayout(1, qk_layout)
    p_smem_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for(
        [BLOCK_M, BLOCK_N], dtype)
    pv_a_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=pv_layout,
        k_width=32 // dtype.primitive_bitwidth)

    # Load Q tile
    desc_q = tma.make_tensor_descriptor(
        q_seq, shape=[q_len, d], strides=[q_row_stride, 1],
        block_shape=[BLOCK_M, d], layout=q_layout)
    desc_o = tma.make_tensor_descriptor(
        o_seq, shape=[q_len, d], strides=[o_row_stride, 1],
        block_shape=[BLOCK_M, d], layout=o_layout)

    q_smem    = gl.allocate_shared_memory(dtype, [BLOCK_M, d], q_layout)
    o_smem    = gl.allocate_shared_memory(dtype, [BLOCK_M, d], o_layout)
    bar_q     = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())

    mbarrier.init(bar_q, count=1)
    mbarrier.expect(bar_q, _tile_nbytes(BLOCK_M, d, dtype.primitive_bitwidth))
    if gl.warp_id() == 0:
        tma.async_copy_global_to_shared(desc_q, [m_block * BLOCK_M, 0], bar_q, q_smem)
    mbarrier.wait(bar_q, phase=0)
    mbarrier.invalidate(bar_q)

    _p1_main_loop(
        q_smem, softmax_lse_ptr, o_smem, desc_o,
        cu_seqlens_k_ptr, k_ptr, v_ptr,
        k_row_stride, v_row_stride,
        k_head_stride, v_head_stride,
        q_bos, q_len,
        h_hk_ratio, d, scale_log2e, is_causal,
        BLOCK_M, BLOCK_N, num_warps, dtype,
        qk_layout, pv_layout, kv_layout, p_smem_layout, pv_a_layout, row_layout,
        total_q, m_block, bid, hid,
    )


# ---------------------------------------------------------------------------
# Phase 2: warp_specialize producer-consumer pipeline
# ---------------------------------------------------------------------------

@gluon.jit
def _p2_load_partition(
    channel,
    k_ptr, v_ptr,
    cu_seqlens_k_ptr,
    k_row_stride, v_row_stride,
    k_head_stride, v_head_stride,
    h_hk_ratio,
    d:        gl.constexpr,
    is_causal: gl.constexpr,
    BLOCK_M:  gl.constexpr,
    BLOCK_N:  gl.constexpr,
    kv_layout: gl.constexpr,
    dtype:    gl.constexpr,
    m_block, bid, hid, q_len,
):
    """Producer partition (1 warp): loads K and V tiles into the channel."""
    k_bos = gl.load(cu_seqlens_k_ptr + bid).to(gl.int32)
    k_eos = gl.load(cu_seqlens_k_ptr + bid + 1).to(gl.int32)
    k_len = k_eos - k_bos

    kv_hid = hid // h_hk_ratio
    k_seq  = k_ptr + k_bos * k_row_stride + kv_hid * k_head_stride
    v_seq  = v_ptr + k_bos * v_row_stride + kv_hid * v_head_stride

    # Build TMA descriptors from runtime base/shape — the whole point of Gluon.
    desc_k = tma.make_tensor_descriptor(
        k_seq, shape=[k_len, d], strides=[k_row_stride, 1],
        block_shape=[BLOCK_N, d], layout=kv_layout)
    desc_v = tma.make_tensor_descriptor(
        v_seq, shape=[k_len, d], strides=[v_row_stride, 1],
        block_shape=[BLOCK_N, d], layout=kv_layout)

    nbytes_kv: gl.constexpr = _tile_nbytes(BLOCK_N, d, dtype.primitive_bitwidth) * 2

    n_block_max = gl.cdiv(k_len, BLOCK_N)
    if is_causal:
        causal_limit = gl.cdiv(
            (m_block + 1) * BLOCK_M + k_len - q_len, BLOCK_N)
        n_block_max = gl.minimum(n_block_max, causal_limit)

    counter = BarrierCounter(gl.to_tensor(0), gl.to_tensor(0), channel.num_stages)

    for n_block in range(n_block_max - 1, -1, -1):
        start_n = n_block * BLOCK_N
        idx   = counter.index
        phase = counter.phase
        k_slot     = channel.k_smem.index(idx)
        v_slot     = channel.v_smem.index(idx)
        ready_bar  = channel.ready_bars.index(idx)
        empty_bar  = channel.empty_bars.index(idx)

        # Wait until consumer has finished with this slot.
        mbarrier.wait(empty_bar, phase)
        mbarrier.expect(ready_bar, nbytes_kv)
        tma.async_copy_global_to_shared(desc_k, [start_n, 0], ready_bar, k_slot)
        tma.async_copy_global_to_shared(desc_v, [start_n, 0], ready_bar, v_slot)

        counter = counter.increment()


@gluon.jit
def _p2_compute_partition(
    channel,
    q_smem, p_smem, o_smem, desc_o,
    softmax_lse_ptr,
    cu_seqlens_k_ptr,
    q_bos, q_len,
    d:           gl.constexpr,
    scale_log2e: gl.constexpr,
    is_causal:   gl.constexpr,
    BLOCK_M:     gl.constexpr,
    BLOCK_N:     gl.constexpr,
    num_warps:   gl.constexpr,   # warps in this compute partition (4)
    dtype:       gl.constexpr,
    qk_layout:   gl.constexpr,
    pv_layout:   gl.constexpr,
    pv_a_layout: gl.constexpr,
    row_layout:  gl.constexpr,
    total_q,
    m_block, bid, hid,
):
    """Consumer partition (4 warps = 1 warp-group): WGMMA QKᵀ, softmax, WGMMA PV."""
    k_bos = gl.load(cu_seqlens_k_ptr + bid).to(gl.int32)
    k_eos = gl.load(cu_seqlens_k_ptr + bid + 1).to(gl.int32)
    k_len = k_eos - k_bos

    acc    = gl.zeros([BLOCK_M, d],   dtype=gl.float32, layout=pv_layout)
    rowmax = gl.full([BLOCK_M],  float("-inf"), dtype=gl.float32, layout=row_layout)
    rowsum = gl.zeros([BLOCK_M], dtype=gl.float32, layout=row_layout)

    n_block_max = gl.cdiv(k_len, BLOCK_N)
    if is_causal:
        causal_limit = gl.cdiv(
            (m_block + 1) * BLOCK_M + k_len - q_len, BLOCK_N)
        n_block_max = gl.minimum(n_block_max, causal_limit)

    counter = BarrierCounter(gl.to_tensor(0), gl.to_tensor(0), channel.num_stages)

    for n_block in range(n_block_max - 1, -1, -1):
        start_n = n_block * BLOCK_N
        idx   = counter.index
        phase = counter.phase
        k_slot    = channel.k_smem.index(idx)
        v_slot    = channel.v_smem.index(idx)
        ready_bar = channel.ready_bars.index(idx)
        empty_bar = channel.empty_bars.index(idx)

        # Wait for producer to finish loading this slot.
        mbarrier.wait(ready_bar, phase)
        fence_async_shared()

        # QKᵀ — K is [BLOCK_N, d], we need [d, BLOCK_N] as WGMMA B operand.
        kt_smem = k_slot.permute((1, 0))
        qk_layout_local: gl.constexpr = qk_layout
        qk = warpgroup_mma(
            q_smem, kt_smem,
            gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=qk_layout_local),
            is_async=True, use_acc=False)
        qk = warpgroup_mma_wait(0, (qk,))

        # Masking
        col_idx = start_n + gl.arange(0, BLOCK_N)
        row_idx = m_block * BLOCK_M + gl.arange(0, BLOCK_M)
        qk = gl.where(col_idx[None, :] < k_len,  qk, float("-inf"))
        qk = gl.where(row_idx[:, None] < q_len,   qk, float("-inf"))
        if is_causal:
            row_limit = row_idx + (k_len - q_len)
            qk = gl.where(col_idx[None, :] <= row_limit[:, None], qk, float("-inf"))

        # Online softmax
        acc, rowmax, rowsum, P = _online_softmax_step(
            acc, rowmax, rowsum, qk, scale_log2e, BLOCK_M, BLOCK_N)

        # P smem round-trip to get DotOperandLayout for WGMMA A.
        P_typed = P.to(dtype)
        p_smem.store(P_typed)
        fence_async_shared()
        P_dot = p_smem.load(pv_a_layout)

        # PV
        acc = warpgroup_mma(P_dot, v_slot, acc, is_async=True, use_acc=True)
        acc = warpgroup_mma_wait(0, (acc,))

        # Signal producer that this slot is now free.
        mbarrier.arrive(empty_bar)
        counter = counter.increment()

    # Epilogue
    pv_row_layout: gl.constexpr = gl.SliceLayout(1, pv_layout)
    rowsum_pv = gl.convert_layout(rowsum, pv_row_layout)
    inv_sum = gl.where(
        (rowsum_pv == 0.0) | (rowsum_pv != rowsum_pv),
        gl.full([BLOCK_M], 1.0, dtype=gl.float32, layout=pv_row_layout),
        1.0 / rowsum_pv,
    )
    out = (acc * inv_sum[:, None]).to(dtype)
    o_smem.store(out)
    fence_async_shared()
    tma.async_copy_shared_to_global(desc_o, [m_block * BLOCK_M, 0], o_smem)
    tma.store_wait(pendings=0)

    lse_val = gl.where(
        (rowsum == 0.0) | (rowsum != rowsum),
        gl.full([BLOCK_M], float("inf"), dtype=gl.float32, layout=row_layout),
        rowmax / scale_log2e + gl.log(rowsum) / scale_log2e,
    )
    lse_ptr  = softmax_lse_ptr + hid * total_q + q_bos + m_block * BLOCK_M
    lse_mask = gl.arange(0, BLOCK_M) < (q_len - m_block * BLOCK_M)
    gl.store(lse_ptr + gl.arange(0, BLOCK_M), lse_val, mask=lse_mask)


@gluon.jit
def flash_varlen_fwd_p2_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr,
    softmax_lse_ptr,
    cu_seqlens_q_ptr,
    cu_seqlens_k_ptr,
    q_row_stride, k_row_stride, v_row_stride, o_row_stride,
    q_head_stride, k_head_stride, v_head_stride, o_head_stride,
    total_q,
    h_hk_ratio:  gl.constexpr,
    d:           gl.constexpr,
    scale_log2e: gl.constexpr,
    is_causal:   gl.constexpr,
    BLOCK_M:     gl.constexpr,
    BLOCK_N:     gl.constexpr,
    NUM_STAGES:  gl.constexpr,
    # num_warps for this kernel must be 5 (1 load + 4 compute).
    # Passed as constexpr so partitions can use it for layout helpers.
    COMPUTE_WARPS: gl.constexpr,  # = 4
):
    """
    Phase 2 kernel — warp_specialize producer-consumer pipeline.

    num_warps=5: 1 producer warp (TMA load) + 4 consumer warps (WGMMA).
    Grid: (cdiv(max_seqlen_q, BLOCK_M), batch, num_heads)
    """
    m_block = gl.program_id(0)
    bid     = gl.program_id(1)
    hid     = gl.program_id(2)

    q_bos = gl.load(cu_seqlens_q_ptr + bid).to(gl.int32)
    q_eos = gl.load(cu_seqlens_q_ptr + bid + 1).to(gl.int32)
    q_len = q_eos - q_bos

    if m_block * BLOCK_M >= q_len:
        return

    q_seq = q_ptr + q_bos * q_row_stride + hid * q_head_stride
    o_seq = o_ptr + q_bos * o_row_stride + hid * o_head_stride

    dtype: gl.constexpr      = q_ptr.dtype.element_ty
    q_layout:  gl.constexpr  = gl.NVMMASharedLayout.get_default_for([BLOCK_M, d], dtype)
    kv_layout: gl.constexpr  = gl.NVMMASharedLayout.get_default_for([BLOCK_N, d], dtype)
    o_layout:  gl.constexpr  = gl.NVMMASharedLayout.get_default_for([BLOCK_M, d], dtype)

    # Layout for the compute partition (uses COMPUTE_WARPS, not total num_warps).
    qk_wpc: gl.constexpr     = _pick_warps_per_cta(BLOCK_M, BLOCK_N, COMPUTE_WARPS)
    pv_wpc: gl.constexpr     = _pick_warps_per_cta(BLOCK_M, d, COMPUTE_WARPS)
    qk_instr_n: gl.constexpr = _pick_instr_n(BLOCK_M, BLOCK_N, COMPUTE_WARPS)
    pv_instr_n: gl.constexpr = _pick_instr_n(BLOCK_M, d, COMPUTE_WARPS)
    qk_layout: gl.constexpr  = gl.NVMMADistributedLayout(
        version=[3, 0], warps_per_cta=qk_wpc,
        instr_shape=[16, qk_instr_n, 256 // dtype.primitive_bitwidth])
    pv_layout: gl.constexpr  = gl.NVMMADistributedLayout(
        version=[3, 0], warps_per_cta=pv_wpc,
        instr_shape=[16, pv_instr_n, 256 // dtype.primitive_bitwidth])
    row_layout: gl.constexpr = gl.SliceLayout(1, qk_layout)
    p_smem_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for(
        [BLOCK_M, BLOCK_N], dtype)
    pv_a_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=pv_layout,
        k_width=32 // dtype.primitive_bitwidth)

    # Shared memory allocations (visible to both partitions via capture).
    q_smem    = gl.allocate_shared_memory(dtype, [BLOCK_M, d], q_layout)
    p_smem    = gl.allocate_shared_memory(dtype, [BLOCK_M, BLOCK_N], p_smem_layout)
    o_smem    = gl.allocate_shared_memory(dtype, [BLOCK_M, d], o_layout)
    bar_q     = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())

    # KV pipeline channel (NUM_STAGES slots, each with ready+empty barrier).
    channel = KVChannel.alloc(BLOCK_N, d, dtype, kv_layout, NUM_STAGES)

    # TMA descriptors for Q and O (static per CTA, built in entry).
    desc_q = tma.make_tensor_descriptor(
        q_seq, shape=[q_len, d], strides=[q_row_stride, 1],
        block_shape=[BLOCK_M, d], layout=q_layout)
    desc_o = tma.make_tensor_descriptor(
        o_seq, shape=[q_len, d], strides=[o_row_stride, 1],
        block_shape=[BLOCK_M, d], layout=o_layout)

    # Load Q before splitting into partitions (all warps participate in the wait,
    # but only one warp issues the TMA; after warp_specialize the compute partition
    # owns q_smem as a read-only input for the whole CTA lifetime).
    mbarrier.init(bar_q, count=1)
    mbarrier.expect(bar_q, _tile_nbytes(BLOCK_M, d, dtype.primitive_bitwidth))
    if gl.warp_id() == 0:
        tma.async_copy_global_to_shared(desc_q, [m_block * BLOCK_M, 0], bar_q, q_smem)
    mbarrier.wait(bar_q, phase=0)
    mbarrier.invalidate(bar_q)

    gl.warp_specialize(
        [
            (_p2_compute_partition, (
                channel,
                q_smem, p_smem, o_smem, desc_o,
                softmax_lse_ptr,
                cu_seqlens_k_ptr,
                q_bos, q_len,
                d, scale_log2e, is_causal,
                BLOCK_M, BLOCK_N, COMPUTE_WARPS, dtype,
                qk_layout, pv_layout, pv_a_layout, row_layout,
                total_q, m_block, bid, hid,
            )),
            (_p2_load_partition, (
                channel,
                k_ptr, v_ptr,
                cu_seqlens_k_ptr,
                k_row_stride, v_row_stride,
                k_head_stride, v_head_stride,
                h_hk_ratio, d, is_causal,
                BLOCK_M, BLOCK_N, kv_layout, dtype,
                m_block, bid, hid, q_len,
            )),
        ],
        [COMPUTE_WARPS],   # warps per partition
        [232],             # maxnreg for compute partition (leave headroom for load)
    )

    channel.release()


# ---------------------------------------------------------------------------
# Shared-memory budget helpers (Python, host-side only)
# ---------------------------------------------------------------------------

_SMEM_LIMIT = 232448   # 227 KB, conservative for H100 with maxnreg=256


def _smem_nonpaged(BLOCK_M, BLOCK_N, d, elem_bytes, num_stages):
    """Bytes used by flash_varlen_fwd_p2_kernel smem allocations."""
    q_tile  = BLOCK_M * d * elem_bytes
    o_tile  = BLOCK_M * d * elem_bytes
    p_tile  = BLOCK_M * BLOCK_N * elem_bytes
    kv_pipe = num_stages * 2 * BLOCK_N * d * elem_bytes
    return q_tile + o_tile + p_tile + kv_pipe + 256   # +256 for barriers/misc


def _pick_block_m(d, elem_bytes, num_stages, BLOCK_N=64, smem_limit=_SMEM_LIMIT):
    for bm in (128, 64, 32):
        if _smem_nonpaged(bm, BLOCK_N, d, elem_bytes, num_stages) <= smem_limit:
            return bm
    raise RuntimeError(
        f"No valid BLOCK_M for d={d}, BLOCK_N={BLOCK_N}, "
        f"num_stages={num_stages}: even BLOCK_M=32 exceeds smem limit")


# ---------------------------------------------------------------------------
# Python launchers
# ---------------------------------------------------------------------------

_allocator_registered = False


def _ensure_allocator():
    global _allocator_registered
    if not _allocator_registered:
        def _alloc(size, align, stream):
            return torch.empty(size, device="cuda", dtype=torch.uint8)
        triton.set_allocator(_alloc)
        _allocator_registered = True


def flash_varlen_fwd_p1(
    q, k, v, o,
    softmax_lse,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q: int,
    softmax_scale: float,
    is_causal: bool,
    BLOCK_M: int = 0,
    BLOCK_N: int = 64,
    num_warps: int = 4,
):
    """
    Phase 1 launcher — single warp-group, no pipeline.  Use for correctness checks.

    Inputs
    ------
    q, k, v  : [total_q/k, h/hk, d]  contiguous, fp16 or bf16
    o        : [total_q, h, d]        pre-allocated output
    softmax_lse : [h, total_q]        pre-allocated, fp32
    cu_seqlens_q/k : [batch+1] int32
    max_seqlen_q : upper bound on sequence length (used only for grid sizing)
    """
    _ensure_allocator()
    total_q, h, d = q.shape
    hk = k.shape[1]
    batch = cu_seqlens_q.shape[0] - 1
    h_hk_ratio = h // hk
    scale_log2e = softmax_scale * math.log2(math.e)

    if BLOCK_N == 0:
        BLOCK_N = 64 if d <= 128 else 32
    if BLOCK_M == 0:
        BLOCK_M = _pick_block_m(d, q.element_size(), num_stages=1, BLOCK_N=BLOCK_N)
    BLOCK_M = max(BLOCK_M, 64)   # WGMMA requires BLOCK_M >= 64

    grid = (triton.cdiv(max_seqlen_q, BLOCK_M), batch, h)

    flash_varlen_fwd_p1_kernel[grid](
        q, k, v, o,
        softmax_lse,
        cu_seqlens_q, cu_seqlens_k,
        q.stride(0), k.stride(0), v.stride(0), o.stride(0),
        q.stride(1), k.stride(1), v.stride(1), o.stride(1),
        total_q,
        h_hk_ratio, d, scale_log2e, is_causal,
        BLOCK_M, BLOCK_N,
        num_warps=num_warps,
        maxnreg=256,
    )
    return o, softmax_lse


def flash_varlen_fwd(
    q, k, v, o,
    softmax_lse,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float,
    is_causal: bool,
    BLOCK_M: int = 0,
    BLOCK_N: int = 64,
    num_stages: int = 2,
    maxnreg: int = 256,
):
    """
    Phase 2 launcher — warp_specialize producer-consumer, 5 warps per CTA.

    Inputs
    ------
    q, k, v  : [total_q/k, h/hk, d]  contiguous, fp16 or bf16
    o        : [total_q, h, d]        pre-allocated output
    softmax_lse : [h, total_q]        pre-allocated, fp32
    cu_seqlens_q/k : [batch+1] int32
    max_seqlen_q/k : upper bounds on sequence length
    num_stages : pipeline depth (2 or 3; auto-bumped to 3 for long sequences)
    """
    _ensure_allocator()
    total_q, h, d = q.shape
    hk = k.shape[1]
    batch = cu_seqlens_q.shape[0] - 1
    h_hk_ratio = h // hk
    scale_log2e = softmax_scale * math.log2(math.e)

    if BLOCK_N == 0:
        BLOCK_N = 64 if d <= 128 else 32
    # Heuristic: bump pipeline depth for long KV sequences with small head dim.
    if num_stages == 2 and d <= 128 and max_seqlen_k >= 2048:
        num_stages = 3
    if BLOCK_M == 0:
        try:
            BLOCK_M = _pick_block_m(d, q.element_size(), num_stages, BLOCK_N)
        except RuntimeError:
            num_stages = max(num_stages - 1, 1)
            BLOCK_M = _pick_block_m(d, q.element_size(), num_stages, BLOCK_N)
    BLOCK_M = max(BLOCK_M, 64)   # WGMMA minimum

    # Cap user-provided BLOCK_M to what fits in smem.
    safe_bm = _pick_block_m(d, q.element_size(), num_stages, BLOCK_N)
    BLOCK_M = min(BLOCK_M, safe_bm)

    if maxnreg < 256:
        maxnreg = 256

    COMPUTE_WARPS = 4   # always one full warp-group for WGMMA

    grid = (triton.cdiv(max_seqlen_q, BLOCK_M), batch, h)

    flash_varlen_fwd_p2_kernel[grid](
        q, k, v, o,
        softmax_lse,
        cu_seqlens_q, cu_seqlens_k,
        q.stride(0), k.stride(0), v.stride(0), o.stride(0),
        q.stride(1), k.stride(1), v.stride(1), o.stride(1),
        total_q,
        h_hk_ratio, d, scale_log2e, is_causal,
        BLOCK_M, BLOCK_N, num_stages, COMPUTE_WARPS,
        num_warps=COMPUTE_WARPS + 1,   # 4 compute + 1 load
        maxnreg=maxnreg,
    )
    return o, softmax_lse
