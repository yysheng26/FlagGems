"""
flash_kernel_gluon.py 鈥?Gluon flash attention forward for Hopper varlen/paged paths.

Why Gluon?
----------
The Triton FA3 kernel disables warp_specialize when is_cu_seqlens_k=True because
k_bos (loaded from cu_seqlens_k) is a runtime value. tl.make_tensor_descriptor
with a runtime-derived base pointer cannot be labeled by the Triton
TaskIdPropagation dataflow pass, so the NVGPUWarpSpecialization MLIR pass fails.

For paged KV, the K/V base pointer for each page must be looked up from a
page_table at runtime, making TaskIdPropagation fail for the same reason.

Gluon's tma.make_tensor_descriptor accepts runtime shape/strides/base on the
device side, bypassing TaskIdPropagation entirely. This file provides a
correctness-first Gluon kernel that uses TMA loads + gl.dot for the attention
computation on Hopper GPUs.

Phase 1 (this file): TMA loads + WGMMA (no warp_specialize yet).
Phase 2: gl.warp_specialize producer-consumer pipeline (non-paged and paged).
Phase 2b: decode kernels (max_seqlen_q <= 8) use TMA + dot_fma, BLOCK_M=min(16, max_seqlen_q).
"""
import math
import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.hopper import tma, mbarrier, fence_async_shared
from flag_gems.utils import libentry
from triton.experimental.gluon.language.nvidia.hopper import (
    tma, mbarrier, fence_async_shared,
    warpgroup_mma, warpgroup_mma_wait,
)
from triton.language.core import _aggregate as aggregate


# 鈹€鈹€ constexpr byte-size helper 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

@gluon.constexpr_function
def _tile_nbytes(shape0, shape1, elem_bits):
    """Byte size of a 2D tile with given element bitwidth."""
    return shape0 * shape1 * (elem_bits // 8)


_DECODE_MAX_SEQLEN_Q = 8
_DECODE_BLOCK_M = 16


def _is_decode_mode(max_seqlen_q: int) -> bool:
    return max_seqlen_q <= _DECODE_MAX_SEQLEN_Q


def _decode_block_n(head_size: int) -> int:
    return 32 if head_size >= 256 else 64


@gluon.constexpr_function
def _decode_blocked_layout(num_warps, rows, d):
    """BlockedLayout for decode dot_fma [rows, d] tiles (Q/acc/O or tall K/V)."""
    d_pt = d // (2 * num_warps)
    if rows <= 16:
        return gl.BlockedLayout(
            size_per_thread=[1, d_pt],
            threads_per_warp=[rows, 2],
            warps_per_cta=[1, num_warps],
            order=[0, 1],
        )
    rows_pw = rows // num_warps
    return gl.BlockedLayout(
        size_per_thread=[2, d // num_warps],
        threads_per_warp=[rows_pw // 2, 4],
        warps_per_cta=[num_warps, 1],
        order=[0, 1],
    )


@gluon.constexpr_function
def _decode_qk_blocked_layout(num_warps, BLOCK_M, BLOCK_N):
    """BlockedLayout for qk [BLOCK_M, BLOCK_N] tiles in decode dot_fma."""
    n_pt = BLOCK_N // (2 * num_warps)
    return gl.BlockedLayout(
        size_per_thread=[1, n_pt],
        threads_per_warp=[BLOCK_M, 2],
        warps_per_cta=[1, num_warps],
        order=[1, 0],
    )


# --- WGMMA layout helpers (mirrors 05-wgmma.py reference) ---

@gluon.constexpr_function
def _pick_warps_per_cta(BLOCK_M, BLOCK_N, num_warps):
    """Expand [4,1] atom until warps_per_cta covers num_warps.
    Tile along M only when BLOCK_M > 16 * current_m_warps, else along N.
    """
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
    """Largest valid instr_shape[1] that divides BLOCK_N and fits in maxN."""
    m = 16
    mReps = triton.cdiv(BLOCK_M, m)
    nReps = triton.cdiv(num_warps, mReps)
    maxN = max(BLOCK_N // nReps, 8)
    n = 256
    while n > maxN or BLOCK_N % n != 0:
        n -= 8
    return n


# 鈹€鈹€ BarrierCounter and Channel for warp_specialize pipeline (adapted from w8a8_block_fp8_bmm.py) 鈹€鈹€
@aggregate
class BarrierCounter:
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
        index = gl.where(rollover, 0, next_index)
        phase = gl.where(rollover, self.phase ^ 1, self.phase)
        return BarrierCounter(index, phase, self.num_barriers)


@aggregate
class Channel:
    k_smem: gl.shared_memory_descriptor
    v_smem: gl.shared_memory_descriptor
    ready_bars: gl.shared_memory_descriptor
    empty_bars: gl.shared_memory_descriptor
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
        BLOCK_M: gl.constexpr,
        BLOCK_N: gl.constexpr,
        d: gl.constexpr,
        dtype: gl.constexpr,
        kv_layout: gl.constexpr,
        num_stages: gl.constexpr,
        num_warps: gl.constexpr,
    ):
        k_smem = gl.allocate_shared_memory(
            dtype, [num_stages, BLOCK_N, d], kv_layout
        )
        v_smem = gl.allocate_shared_memory(
            dtype, [num_stages, BLOCK_N, d], kv_layout
        )
        ready_bars = gl.allocate_shared_memory(
            gl.int64, [num_stages, 1], mbarrier.MBarrierLayout()
        )
        empty_bars = gl.allocate_shared_memory(
            gl.int64, [num_stages, 1], mbarrier.MBarrierLayout()
        )
        for i in gl.static_range(num_stages):
            mbarrier.init(ready_bars.index(i), count=1)
            mbarrier.init(empty_bars.index(i), count=1)
            mbarrier.arrive(empty_bars.index(i), count=1)
        return Channel(k_smem, v_smem, ready_bars, empty_bars, num_stages)

    @gluon.jit
    def release(self):
        self.k_smem._keep_alive()
        self.v_smem._keep_alive()
        for i in gl.static_range(self.num_stages):
            mbarrier.invalidate(self.ready_bars.index(i))
            mbarrier.invalidate(self.empty_bars.index(i))


# 鈹€鈹€ online softmax step 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

@gluon.jit
def _softmax_step(
    acc, rowmax, rowsum,
    qk,                    # [BLOCK_M, BLOCK_N] fp32锛宭ayout = qk_layout
    scale_log2e,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
):
    """Online softmax step (log2-based, numerically stable).

    Mirrors Triton flash_kernel.py softmax_rescale: when row_max is -inf
    (all scores masked), substitute 0 for -inf before exponentiation so
    exp2(-inf - 0) = 0 rather than exp2(-inf - (-inf)) = exp2(nan).
    """
    new_max = gl.max(qk, axis=1)
    new_max = gl.maximum(rowmax, new_max)
    # Substitute 0 for -inf exactly as Triton does in is_border branches.
    safe_max = gl.where(new_max == float("-inf"),
                        gl.zeros_like(new_max), new_max)
    alpha = gl.exp2((rowmax - safe_max) * scale_log2e)
    # Broadcast alpha to pv_layout for acc rescaling
    acc_row_layout: gl.constexpr = gl.SliceLayout(1, acc.type.layout)
    alpha_for_acc = gl.convert_layout(alpha, acc_row_layout)
    acc    = acc * alpha_for_acc[:, None]
    rowsum = rowsum * alpha
    # Use safe_max so exp2(qk*s - (-inf)) doesn't produce nan
    P      = gl.exp2((qk - safe_max[:, None]) * scale_log2e)
    rowsum = rowsum + gl.sum(P, axis=1)
    rowmax = new_max
    return acc, rowmax, rowsum, P


@gluon.jit
def _softmax_step_decode(acc, rowmax, rowsum, qk, scale_log2e,
                         qk_row_layout: gl.constexpr,
                         acc_row_layout: gl.constexpr):
    """Online softmax for decode dot_fma (mirrors _softmax_step layout rules)."""
    new_max = gl.max(qk, axis=1)
    new_max = gl.maximum(rowmax, new_max)
    safe_max = gl.where(new_max == float("-inf"),
                        gl.zeros_like(new_max), new_max)
    alpha = gl.exp2((rowmax - safe_max) * scale_log2e)
    alpha_for_acc = gl.convert_layout(alpha, acc_row_layout)
    acc = acc * alpha_for_acc[:, None]
    rowsum = rowsum * alpha
    P = gl.exp2((qk - safe_max[:, None]) * scale_log2e)
    rowsum = rowsum + gl.sum(P, axis=1)
    rowmax = gl.convert_layout(new_max, qk_row_layout)
    return acc, rowmax, rowsum, P


# --- compute and load partitions for warp_specialize (Phase 2) ---

@gluon.jit
def compute_partition(channel, q_smem, p_smem, o_smem_out, desc_o, o_ptr, softmax_lse_ptr, q_ptr, desc_q, bar_q, cu_seqlens_q_ptr, cu_seqlens_k_ptr, q_row_stride, k_row_stride, v_row_stride, o_row_stride, q_head_stride, k_head_stride, v_head_stride, o_head_stride, total_q, h, hk, h_hk_ratio, d, scale_softmax_log2, is_causal, BLOCK_M, BLOCK_N, q_len, k_len, hid, num_warps):
    dtype: gl.constexpr = q_ptr.dtype.element_ty
    q_layout:  gl.constexpr = gl.NVMMASharedLayout.get_default_for([BLOCK_M, d], dtype)
    kv_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for([BLOCK_N, d], dtype)
    o_layout:  gl.constexpr = gl.NVMMASharedLayout.get_default_for([BLOCK_M, d], dtype)
    qk_warps_per_cta: gl.constexpr = _pick_warps_per_cta(BLOCK_M, BLOCK_N, num_warps)
    pv_warps_per_cta: gl.constexpr = _pick_warps_per_cta(BLOCK_M, d, num_warps)
    qk_instr_n: gl.constexpr = _pick_instr_n(BLOCK_M, BLOCK_N, num_warps)
    pv_instr_n: gl.constexpr = _pick_instr_n(BLOCK_M, d, num_warps)
    qk_layout: gl.constexpr = gl.NVMMADistributedLayout(
        version=[3, 0],
        warps_per_cta=qk_warps_per_cta,
        instr_shape=[16, qk_instr_n, 256 // dtype.primitive_bitwidth],
    )
    pv_layout: gl.constexpr = gl.NVMMADistributedLayout(
        version=[3, 0],
        warps_per_cta=pv_warps_per_cta,
        instr_shape=[16, pv_instr_n, 256 // dtype.primitive_bitwidth],
    )
    row_layout: gl.constexpr = gl.SliceLayout(1, qk_layout)
    pv_a_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0,
        parent=pv_layout,
        k_width=32 // dtype.primitive_bitwidth,
    )
    p_smem_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for(
        [BLOCK_M, BLOCK_N], dtype)

    m_block = gl.program_id(0)
    bid = gl.program_id(1)
    hid = gl.program_id(2)

    q_bos = gl.load(cu_seqlens_q_ptr + bid).to(gl.int32)
    q_eos = gl.load(cu_seqlens_q_ptr + bid + 1).to(gl.int32)
    q_len = q_eos - q_bos

    k_bos = gl.load(cu_seqlens_k_ptr + bid).to(gl.int32)
    k_eos = gl.load(cu_seqlens_k_ptr + bid + 1).to(gl.int32)
    k_len = k_eos - k_bos

    # load Q
    mbarrier.expect(bar_q, _tile_nbytes(BLOCK_M, d, dtype.primitive_bitwidth))
    tma.async_copy_global_to_shared(desc_q, [m_block * BLOCK_M, 0], bar_q, q_smem)
    mbarrier.wait(bar_q, phase=0)
    mbarrier.invalidate(bar_q)

    acc    = gl.zeros((BLOCK_M, d), dtype=gl.float32, layout=pv_layout)
    rowmax = gl.full([BLOCK_M], float("-inf"), dtype=gl.float32, layout=row_layout)
    rowsum = gl.zeros([BLOCK_M], dtype=gl.float32, layout=row_layout)

    counter = BarrierCounter(gl.to_tensor(0), gl.to_tensor(0), channel.num_stages)

    n_block_max = gl.cdiv(k_len, BLOCK_N)
    if is_causal:
        causal_limit = gl.cdiv((m_block + 1) * BLOCK_M + k_len - q_len, BLOCK_N)
        n_block_max = gl.minimum(n_block_max, causal_limit)

    phase = 0
    for n_block in range(n_block_max - 1, -1, -1):
        start_n = n_block * BLOCK_N
        index, phase = counter.index, counter.phase
        k_slot = channel.k_smem.index(index)
        v_slot = channel.v_smem.index(index)
        ready_bar = channel.ready_bars.index(index)
        empty_bar = channel.empty_bars.index(index)
        mbarrier.wait(ready_bar, phase)
        kt_smem = k_slot.permute((1, 0))
        qk = warpgroup_mma(q_smem, kt_smem,
                           gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=qk_layout),
                           is_async=True, use_acc=False)
        qk = warpgroup_mma_wait(0, (qk,))
        col_idx   = start_n + gl.arange(0, BLOCK_N)
        col_mask  = col_idx[None, :] < k_len
        qk = gl.where(col_mask, qk, float("-inf"))
        if is_causal:
            row_limit = m_block * BLOCK_M + gl.arange(0, BLOCK_M) + (k_len - q_len)
            causal_mask = col_idx[None, :] <= row_limit[:, None]
            qk = gl.where(causal_mask, qk, float("-inf"))
        acc, rowmax, rowsum, P = _softmax_step(
            acc, rowmax, rowsum, qk, scale_softmax_log2, BLOCK_M, BLOCK_N)
        P_typed = P.to(dtype)
        p_smem.store(P_typed)
        fence_async_shared()
        P_dot = p_smem.load(pv_a_layout)
        acc = warpgroup_mma(P_dot, v_slot, acc, is_async=True, use_acc=True)
        acc = warpgroup_mma_wait(0, (acc,))
        mbarrier.arrive(empty_bar)
        counter = counter.increment()

    # epilogue
    pv_row_layout: gl.constexpr = gl.SliceLayout(1, pv_layout)
    rowsum_pv = gl.convert_layout(rowsum, pv_row_layout)
    rowmax_pv = gl.convert_layout(rowmax, pv_row_layout)
    inv_sum = gl.where(
        (rowsum_pv == 0) | (rowsum_pv != rowsum_pv),
        gl.full([BLOCK_M], 1.0, dtype=gl.float32, layout=pv_row_layout),
        1.0 / rowsum_pv,
    )
    out = (acc * inv_sum[:, None]).to(dtype)
    o_smem_out.store(out)
    fence_async_shared()
    tma.async_copy_shared_to_global(desc_o, [m_block * BLOCK_M, 0], o_smem_out)
    tma.store_wait(pendings=0)
    row_std_layout: gl.constexpr = gl.SliceLayout(1, qk_layout)
    lse_val = gl.where(
        (rowsum == 0) | (rowsum != rowsum),
        gl.full([BLOCK_M], float("inf"), dtype=gl.float32, layout=row_std_layout),
        rowmax / scale_softmax_log2 + gl.log(rowsum) / scale_softmax_log2,
    )
    lse_ptr = softmax_lse_ptr + hid * total_q + q_bos + m_block * BLOCK_M
    lse_mask = gl.arange(0, BLOCK_M) < (q_len - m_block * BLOCK_M)
    gl.store(lse_ptr + gl.arange(0, BLOCK_M), lse_val, mask=lse_mask)


@gluon.jit
def load_partition(channel, k_ptr, v_ptr, desc_k, desc_v, cu_seqlens_q_ptr, cu_seqlens_k_ptr, q_row_stride, k_row_stride, v_row_stride, q_head_stride, k_head_stride, v_head_stride, h, hk, h_hk_ratio, d, is_causal, BLOCK_M, BLOCK_N, q_len, k_len, num_warps):
    m_block = gl.program_id(0)
    bid = gl.program_id(1)
    hid = gl.program_id(2)
    q_bos = gl.load(cu_seqlens_q_ptr + bid).to(gl.int32)
    q_eos = gl.load(cu_seqlens_q_ptr + bid + 1).to(gl.int32)
    q_len = q_eos - q_bos
    k_bos = gl.load(cu_seqlens_k_ptr + bid).to(gl.int32)
    k_eos = gl.load(cu_seqlens_k_ptr + bid + 1).to(gl.int32)
    k_len = k_eos - k_bos
    k_seq = k_ptr + k_bos * k_row_stride + (hid // h_hk_ratio) * k_head_stride
    v_seq = v_ptr + k_bos * v_row_stride + (hid // h_hk_ratio) * v_head_stride
    counter = BarrierCounter(gl.to_tensor(0), gl.to_tensor(0), channel.num_stages)
    n_block_max = gl.cdiv(k_len, BLOCK_N)
    if is_causal:
        causal_limit = gl.cdiv((m_block + 1) * BLOCK_M + k_len - q_len, BLOCK_N)
        n_block_max = gl.minimum(n_block_max, causal_limit)
    for n_block in range(n_block_max - 1, -1, -1):
        start_n = n_block * BLOCK_N
        index, phase = counter.index, counter.phase
        k_slot = channel.k_smem.index(index)
        v_slot = channel.v_smem.index(index)
        ready_bar = channel.ready_bars.index(index)
        empty_bar = channel.empty_bars.index(index)
        mbarrier.wait(empty_bar, phase)
        mbarrier.expect(ready_bar, _tile_nbytes(BLOCK_N, d, k_ptr.dtype.element_ty.primitive_bitwidth) * 2)
        tma.async_copy_global_to_shared(desc_k, [start_n, 0], ready_bar, k_slot)
        tma.async_copy_global_to_shared(desc_v, [start_n, 0], ready_bar, v_slot)
        counter = counter.increment()


@gluon.jit
def compute_paged_partition(channel, q_smem, p_smem, o_smem_out, desc_o, o_ptr, softmax_lse_ptr, q_ptr, desc_q, bar_q, cu_seqlens_q_ptr, cu_seqlens_k_ptr, page_table_ptr, q_row_stride, q_head_stride, o_row_stride, o_head_stride, k_page_stride, k_row_stride, k_head_stride, pt_batch_stride, total_q, h, hk, h_hk_ratio, d, block_size, scale_softmax_log2, is_causal, BLOCK_M, BLOCK_N, q_len, k_len, hid, num_warps):
    dtype: gl.constexpr = q_ptr.dtype.element_ty
    q_layout:  gl.constexpr = gl.NVMMASharedLayout.get_default_for([BLOCK_M, d], dtype)
    kv_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for([BLOCK_N, d], dtype)
    o_layout:  gl.constexpr = gl.NVMMASharedLayout.get_default_for([BLOCK_M, d], dtype)
    qk_warps_per_cta: gl.constexpr = _pick_warps_per_cta(BLOCK_M, BLOCK_N, num_warps)
    pv_warps_per_cta: gl.constexpr = _pick_warps_per_cta(BLOCK_M, d, num_warps)
    qk_instr_n: gl.constexpr = _pick_instr_n(BLOCK_M, BLOCK_N, num_warps)
    pv_instr_n: gl.constexpr = _pick_instr_n(BLOCK_M, d, num_warps)
    qk_layout: gl.constexpr = gl.NVMMADistributedLayout(
        version=[3, 0],
        warps_per_cta=qk_warps_per_cta,
        instr_shape=[16, qk_instr_n, 256 // dtype.primitive_bitwidth],
    )
    pv_layout: gl.constexpr = gl.NVMMADistributedLayout(
        version=[3, 0],
        warps_per_cta=pv_warps_per_cta,
        instr_shape=[16, pv_instr_n, 256 // dtype.primitive_bitwidth],
    )
    row_layout: gl.constexpr = gl.SliceLayout(1, qk_layout)
    pv_a_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0,
        parent=pv_layout,
        k_width=32 // dtype.primitive_bitwidth,
    )
    p_smem_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for(
        [BLOCK_M, BLOCK_N], dtype)

    m_block = gl.program_id(0)
    bid = gl.program_id(1)
    hid = gl.program_id(2)

    q_bos = gl.load(cu_seqlens_q_ptr + bid).to(gl.int32)
    q_eos = gl.load(cu_seqlens_q_ptr + bid + 1).to(gl.int32)
    q_len = q_eos - q_bos

    k_bos = gl.load(cu_seqlens_k_ptr + bid).to(gl.int32)
    k_eos = gl.load(cu_seqlens_k_ptr + bid + 1).to(gl.int32)
    k_len = k_eos - k_bos

    mbarrier.expect(bar_q, _tile_nbytes(BLOCK_M, d, dtype.primitive_bitwidth))
    tma.async_copy_global_to_shared(desc_q, [m_block * BLOCK_M, 0], bar_q, q_smem)
    mbarrier.wait(bar_q, phase=0)
    mbarrier.invalidate(bar_q)

    acc    = gl.zeros((BLOCK_M, d), dtype=gl.float32, layout=pv_layout)
    rowmax = gl.full([BLOCK_M], float("-inf"), dtype=gl.float32, layout=row_layout)
    rowsum = gl.zeros([BLOCK_M], dtype=gl.float32, layout=row_layout)

    counter = BarrierCounter(gl.to_tensor(0), gl.to_tensor(0), channel.num_stages)

    n_block_max = gl.cdiv(k_len, BLOCK_N)
    if is_causal:
        causal_limit = gl.cdiv((m_block + 1) * BLOCK_M + k_len - q_len, BLOCK_N)
        n_block_max = gl.minimum(n_block_max, causal_limit)

    for n_block in range(n_block_max - 1, -1, -1):
        tok_start = n_block * BLOCK_N
        tile_tokens = gl.minimum(k_len - tok_start, BLOCK_N)
        index, phase = counter.index, counter.phase
        k_slot = channel.k_smem.index(index)
        v_slot = channel.v_smem.index(index)
        ready_bar = channel.ready_bars.index(index)
        empty_bar = channel.empty_bars.index(index)
        mbarrier.wait(ready_bar, phase)
        kt_smem = k_slot.permute((1, 0))
        qk = warpgroup_mma(q_smem, kt_smem,
                           gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=qk_layout),
                           is_async=True, use_acc=False)
        qk = warpgroup_mma_wait(0, (qk,))
        col_idx   = gl.arange(0, BLOCK_N)
        col_mask  = col_idx[None, :] < tile_tokens
        qk = gl.where(col_mask, qk, float("-inf"))
        if is_causal:
            row_limit = m_block * BLOCK_M + gl.arange(0, BLOCK_M) + (k_len - q_len)
            col_global = tok_start + col_idx
            causal_mask = col_global[None, :] <= row_limit[:, None]
            qk = gl.where(causal_mask, qk, float("-inf"))
        acc, rowmax, rowsum, P = _softmax_step(
            acc, rowmax, rowsum, qk, scale_softmax_log2, BLOCK_M, BLOCK_N)
        P_typed = P.to(dtype)
        p_smem.store(P_typed)
        fence_async_shared()
        P_dot = p_smem.load(pv_a_layout)
        acc = warpgroup_mma(P_dot, v_slot, acc, is_async=True, use_acc=True)
        acc = warpgroup_mma_wait(0, (acc,))
        mbarrier.arrive(empty_bar)
        counter = counter.increment()

    pv_row_layout: gl.constexpr = gl.SliceLayout(1, pv_layout)
    rowsum_pv = gl.convert_layout(rowsum, pv_row_layout)
    inv_sum = gl.where(
        (rowsum_pv == 0) | (rowsum_pv != rowsum_pv),
        gl.full([BLOCK_M], 1.0, dtype=gl.float32, layout=pv_row_layout),
        1.0 / rowsum_pv,
    )
    out = (acc * inv_sum[:, None]).to(dtype)
    o_smem_out.store(out)
    fence_async_shared()
    tma.async_copy_shared_to_global(desc_o, [m_block * BLOCK_M, 0], o_smem_out)
    tma.store_wait(pendings=0)
    row_std_layout: gl.constexpr = gl.SliceLayout(1, qk_layout)
    lse_val = gl.where(
        (rowsum == 0) | (rowsum != rowsum),
        gl.full([BLOCK_M], float("inf"), dtype=gl.float32, layout=row_std_layout),
        rowmax / scale_softmax_log2 + gl.log(rowsum) / scale_softmax_log2,
    )
    lse_ptr = softmax_lse_ptr + hid * total_q + q_bos + m_block * BLOCK_M
    lse_mask = gl.arange(0, BLOCK_M) < (q_len - m_block * BLOCK_M)
    gl.store(lse_ptr + gl.arange(0, BLOCK_M), lse_val, mask=lse_mask)


@gluon.jit
def load_paged_partition(channel, k_ptr, v_ptr, page_table_ptr, cu_seqlens_q_ptr, cu_seqlens_k_ptr, q_row_stride, k_row_stride, v_row_stride, q_head_stride, k_head_stride_gen, v_head_stride, pt_batch_stride, k_page_stride, k_head_stride, h, hk, h_hk_ratio, d, block_size, is_causal, BLOCK_M, BLOCK_N, q_len, k_len, num_warps):
    m_block = gl.program_id(0)
    bid = gl.program_id(1)
    hid = gl.program_id(2)
    q_bos = gl.load(cu_seqlens_q_ptr + bid).to(gl.int32)
    q_eos = gl.load(cu_seqlens_q_ptr + bid + 1).to(gl.int32)
    q_len = q_eos - q_bos
    k_bos = gl.load(cu_seqlens_k_ptr + bid).to(gl.int32)
    k_eos = gl.load(cu_seqlens_k_ptr + bid + 1).to(gl.int32)
    k_len = k_eos - k_bos
    kv_hid = hid // h_hk_ratio
    dtype: gl.constexpr = k_ptr.dtype.element_ty
    kv_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for([BLOCK_N, d], dtype)
    nbytes_per_stage: gl.constexpr = _tile_nbytes(BLOCK_N, d, dtype.primitive_bitwidth) * 2
    counter = BarrierCounter(gl.to_tensor(0), gl.to_tensor(0), channel.num_stages)
    n_block_max = gl.cdiv(k_len, BLOCK_N)
    if is_causal:
        causal_limit = gl.cdiv((m_block + 1) * BLOCK_M + k_len - q_len, BLOCK_N)
        n_block_max = gl.minimum(n_block_max, causal_limit)
    for n_block in range(n_block_max - 1, -1, -1):
        pt_offset = bid * pt_batch_stride + n_block
        page_idx = gl.load(page_table_ptr + pt_offset).to(gl.int32)
        k_page_base = k_ptr + page_idx * k_page_stride + kv_hid * k_head_stride
        v_page_base = v_ptr + page_idx * k_page_stride + kv_hid * k_head_stride
        index, phase = counter.index, counter.phase
        k_slot = channel.k_smem.index(index)
        v_slot = channel.v_smem.index(index)
        ready_bar = channel.ready_bars.index(index)
        empty_bar = channel.empty_bars.index(index)
        mbarrier.wait(empty_bar, phase)
        mbarrier.expect(ready_bar, nbytes_per_stage)
        desc_k = tma.make_tensor_descriptor(
            k_page_base, shape=[block_size, d], strides=[k_row_stride, 1],
            block_shape=[BLOCK_N, d], layout=kv_layout)
        desc_v = tma.make_tensor_descriptor(
            v_page_base, shape=[block_size, d], strides=[k_row_stride, 1],
            block_shape=[BLOCK_N, d], layout=kv_layout)
        tma.async_copy_global_to_shared(desc_k, [0, 0], ready_bar, k_slot)
        tma.async_copy_global_to_shared(desc_v, [0, 0], ready_bar, v_slot)
        counter = counter.increment()


# --- decode kernels: TMA + dot_fma (BLOCK_M=min(16,max_q), no warp_specialize) ---

@gluon.jit
def _sync_load_kv_tile(desc_k, desc_v, start_n, bar_kv, k_smem, v_smem, nbytes):
    mbarrier.init(bar_kv, count=1)
    mbarrier.expect(bar_kv, nbytes)
    tma.async_copy_global_to_shared(desc_k, [start_n, 0], bar_kv, k_smem)
    tma.async_copy_global_to_shared(desc_v, [start_n, 0], bar_kv, v_smem)
    mbarrier.wait(bar_kv, phase=0)
    mbarrier.invalidate(bar_kv)


@gluon.jit
def _sync_load_paged_kv_tile(
    k_page_base, v_page_base, block_size, d, k_row_stride, kv_layout,
    bar_kv, k_smem, v_smem, nbytes, BLOCK_N: gl.constexpr,
):
    mbarrier.init(bar_kv, count=1)
    mbarrier.expect(bar_kv, nbytes)
    desc_k = tma.make_tensor_descriptor(
        k_page_base, shape=[block_size, d], strides=[k_row_stride, 1],
        block_shape=[BLOCK_N, d], layout=kv_layout)
    desc_v = tma.make_tensor_descriptor(
        v_page_base, shape=[block_size, d], strides=[k_row_stride, 1],
        block_shape=[BLOCK_N, d], layout=kv_layout)
    tma.async_copy_global_to_shared(desc_k, [0, 0], bar_kv, k_smem)
    tma.async_copy_global_to_shared(desc_v, [0, 0], bar_kv, v_smem)
    mbarrier.wait(bar_kv, phase=0)
    mbarrier.invalidate(bar_kv)


@gluon.jit
def _decode_epilogue(
    acc, rowmax, rowsum, o_smem_out, desc_o, softmax_lse_ptr,
    m_block, q_bos, q_len, hid, total_q, scale_softmax_log2,
    BLOCK_M: gl.constexpr, md_layout: gl.constexpr,
    qk_row_layout: gl.constexpr, dtype: gl.constexpr,
):
    acc_row_layout: gl.constexpr = gl.SliceLayout(1, md_layout)
    rowsum_acc = gl.convert_layout(rowsum, acc_row_layout)
    inv_sum = gl.where(
        (rowsum_acc == 0) | (rowsum_acc != rowsum_acc),
        gl.full([BLOCK_M], 1.0, dtype=gl.float32, layout=acc_row_layout),
        1.0 / rowsum_acc,
    )
    out = (acc * inv_sum[:, None]).to(dtype)
    o_smem_out.store(out)
    fence_async_shared()
    tma.async_copy_shared_to_global(desc_o, [m_block * BLOCK_M, 0], o_smem_out)
    tma.store_wait(pendings=0)
    lse_val = gl.where(
        (rowsum == 0) | (rowsum != rowsum),
        gl.full([BLOCK_M], float("inf"), dtype=gl.float32, layout=qk_row_layout),
        rowmax / scale_softmax_log2 + gl.log(rowsum) / scale_softmax_log2,
    )
    lse_ptr = softmax_lse_ptr + hid * total_q + q_bos + m_block * BLOCK_M
    lse_mask = gl.arange(0, BLOCK_M) < (q_len - m_block * BLOCK_M)
    gl.store(lse_ptr + gl.arange(0, BLOCK_M), lse_val, mask=lse_mask)


@libentry()
@gluon.jit
def flash_varlen_fwd_gluon_decode_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr,
    softmax_lse_ptr,
    cu_seqlens_q_ptr,
    cu_seqlens_k_ptr,
    q_row_stride, k_row_stride, v_row_stride, o_row_stride,
    q_head_stride, k_head_stride, v_head_stride, o_head_stride,
    total_q,
    h: gl.constexpr, hk: gl.constexpr, h_hk_ratio: gl.constexpr,
    d: gl.constexpr, scale_softmax_log2: gl.constexpr, is_causal: gl.constexpr,
    BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, num_warps: gl.constexpr,
):
    m_block = gl.program_id(0)
    bid = gl.program_id(1)
    hid = gl.program_id(2)

    q_bos = gl.load(cu_seqlens_q_ptr + bid).to(gl.int32)
    q_eos = gl.load(cu_seqlens_q_ptr + bid + 1).to(gl.int32)
    q_len = q_eos - q_bos
    k_bos = gl.load(cu_seqlens_k_ptr + bid).to(gl.int32)
    k_eos = gl.load(cu_seqlens_k_ptr + bid + 1).to(gl.int32)
    k_len = k_eos - k_bos
    if m_block * BLOCK_M >= q_len:
        return

    q_seq = q_ptr + q_bos * q_row_stride + hid * q_head_stride
    k_seq = k_ptr + k_bos * k_row_stride + (hid // h_hk_ratio) * k_head_stride
    v_seq = v_ptr + k_bos * v_row_stride + (hid // h_hk_ratio) * v_head_stride
    o_seq = o_ptr + q_bos * o_row_stride + hid * o_head_stride

    dtype: gl.constexpr = q_ptr.dtype.element_ty
    q_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for([BLOCK_M, d], dtype)
    kv_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for([BLOCK_N, d], dtype)
    o_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for([BLOCK_M, d], dtype)
    md_layout: gl.constexpr = _decode_blocked_layout(num_warps, BLOCK_M, d)
    nd_layout: gl.constexpr = _decode_blocked_layout(num_warps, BLOCK_N, d)
    qk_mn_layout: gl.constexpr = _decode_qk_blocked_layout(num_warps, BLOCK_M, BLOCK_N)
    qk_lhs: gl.constexpr = gl.DotOperandLayout(parent=qk_mn_layout, operand_index=0, k_width=0)
    qk_rhs: gl.constexpr = gl.DotOperandLayout(parent=qk_mn_layout, operand_index=1, k_width=0)
    pv_lhs: gl.constexpr = gl.DotOperandLayout(parent=md_layout, operand_index=0, k_width=0)
    pv_rhs: gl.constexpr = gl.DotOperandLayout(parent=md_layout, operand_index=1, k_width=0)
    qk_row_layout: gl.constexpr = gl.SliceLayout(1, qk_mn_layout)
    acc_row_layout: gl.constexpr = gl.SliceLayout(1, md_layout)

    desc_q = tma.make_tensor_descriptor(
        q_seq, shape=[q_len, d], strides=[q_row_stride, 1],
        block_shape=[BLOCK_M, d], layout=q_layout)
    desc_k = tma.make_tensor_descriptor(
        k_seq, shape=[k_len, d], strides=[k_row_stride, 1],
        block_shape=[BLOCK_N, d], layout=kv_layout)
    desc_v = tma.make_tensor_descriptor(
        v_seq, shape=[k_len, d], strides=[v_row_stride, 1],
        block_shape=[BLOCK_N, d], layout=kv_layout)
    desc_o = tma.make_tensor_descriptor(
        o_seq, shape=[q_len, d], strides=[o_row_stride, 1],
        block_shape=[BLOCK_M, d], layout=o_layout)

    q_smem = gl.allocate_shared_memory(dtype, [BLOCK_M, d], q_layout)
    k_smem = gl.allocate_shared_memory(dtype, [BLOCK_N, d], kv_layout)
    v_smem = gl.allocate_shared_memory(dtype, [BLOCK_N, d], kv_layout)
    o_smem_out = gl.allocate_shared_memory(dtype, [BLOCK_M, d], o_layout)
    bar_q = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    bar_kv = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    nbytes_kv: gl.constexpr = _tile_nbytes(BLOCK_N, d, dtype.primitive_bitwidth) * 2

    mbarrier.init(bar_q, count=1)
    mbarrier.expect(bar_q, _tile_nbytes(BLOCK_M, d, dtype.primitive_bitwidth))
    tma.async_copy_global_to_shared(desc_q, [m_block * BLOCK_M, 0], bar_q, q_smem)
    mbarrier.wait(bar_q, phase=0)
    mbarrier.invalidate(bar_q)

    acc = gl.zeros((BLOCK_M, d), dtype=gl.float32, layout=md_layout)
    rowmax = gl.full([BLOCK_M], float("-inf"), dtype=gl.float32, layout=qk_row_layout)
    rowsum = gl.zeros([BLOCK_M], dtype=gl.float32, layout=qk_row_layout)

    n_block_max = gl.cdiv(k_len, BLOCK_N)
    if is_causal:
        causal_limit = gl.cdiv((m_block + 1) * BLOCK_M + k_len - q_len, BLOCK_N)
        n_block_max = gl.minimum(n_block_max, causal_limit)

    q_tile = q_smem.load(md_layout)
    valid_row = m_block * BLOCK_M + gl.arange(0, BLOCK_M, layout=qk_row_layout) < q_len
    for n_block in range(n_block_max - 1, -1, -1):
        start_n = n_block * BLOCK_N
        _sync_load_kv_tile(desc_k, desc_v, start_n, bar_kv, k_smem, v_smem, nbytes_kv)
        k_tile = k_smem.load(nd_layout)
        qk_acc = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=qk_mn_layout)
        qk = gl.dot_fma(
            gl.convert_layout(q_tile.to(gl.float32), qk_lhs),
            gl.convert_layout(k_tile.permute(1, 0).to(gl.float32), qk_rhs),
            qk_acc,
        )
        col_idx = start_n + gl.arange(0, BLOCK_N)
        qk = gl.where(col_idx[None, :] < k_len, qk, float("-inf"))
        qk = gl.where(valid_row[:, None], qk, float("-inf"))
        if is_causal:
            row_limit = m_block * BLOCK_M + gl.arange(0, BLOCK_M, layout=qk_row_layout) + (k_len - q_len)
            qk = gl.where(col_idx[None, :] <= row_limit[:, None], qk, float("-inf"))
        acc, rowmax, rowsum, P = _softmax_step_decode(
            acc, rowmax, rowsum, qk, scale_softmax_log2,
            qk_row_layout, acc_row_layout)
        v_tile = v_smem.load(nd_layout)
        acc = gl.dot_fma(
            gl.convert_layout(P, pv_lhs),
            gl.convert_layout(v_tile.to(gl.float32), pv_rhs),
            acc,
        )

    _decode_epilogue(
        acc, rowmax, rowsum, o_smem_out, desc_o, softmax_lse_ptr,
        m_block, q_bos, q_len, hid, total_q, scale_softmax_log2,
        BLOCK_M, md_layout, qk_row_layout, dtype)


@libentry()
@gluon.jit
def flash_paged_fwd_gluon_decode_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr,
    softmax_lse_ptr,
    cu_seqlens_q_ptr,
    cu_seqlens_k_ptr,
    page_table_ptr,
    q_row_stride, q_head_stride,
    o_row_stride, o_head_stride,
    k_page_stride, k_row_stride, k_head_stride,
    pt_batch_stride, total_q,
    h: gl.constexpr, hk: gl.constexpr, h_hk_ratio: gl.constexpr,
    d: gl.constexpr, block_size: gl.constexpr,
    scale_softmax_log2: gl.constexpr, is_causal: gl.constexpr,
    BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, num_warps: gl.constexpr,
):
    m_block = gl.program_id(0)
    bid = gl.program_id(1)
    hid = gl.program_id(2)

    q_bos = gl.load(cu_seqlens_q_ptr + bid).to(gl.int32)
    q_eos = gl.load(cu_seqlens_q_ptr + bid + 1).to(gl.int32)
    q_len = q_eos - q_bos
    k_bos = gl.load(cu_seqlens_k_ptr + bid).to(gl.int32)
    k_eos = gl.load(cu_seqlens_k_ptr + bid + 1).to(gl.int32)
    k_len = k_eos - k_bos
    if m_block * BLOCK_M >= q_len:
        return

    kv_hid = hid // h_hk_ratio
    q_seq = q_ptr + q_bos * q_row_stride + hid * q_head_stride
    o_seq = o_ptr + q_bos * o_row_stride + hid * o_head_stride

    dtype: gl.constexpr = q_ptr.dtype.element_ty
    q_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for([BLOCK_M, d], dtype)
    kv_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for([BLOCK_N, d], dtype)
    o_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for([BLOCK_M, d], dtype)
    md_layout: gl.constexpr = _decode_blocked_layout(num_warps, BLOCK_M, d)
    nd_layout: gl.constexpr = _decode_blocked_layout(num_warps, BLOCK_N, d)
    qk_mn_layout: gl.constexpr = _decode_qk_blocked_layout(num_warps, BLOCK_M, BLOCK_N)
    qk_lhs: gl.constexpr = gl.DotOperandLayout(parent=qk_mn_layout, operand_index=0, k_width=0)
    qk_rhs: gl.constexpr = gl.DotOperandLayout(parent=qk_mn_layout, operand_index=1, k_width=0)
    pv_lhs: gl.constexpr = gl.DotOperandLayout(parent=md_layout, operand_index=0, k_width=0)
    pv_rhs: gl.constexpr = gl.DotOperandLayout(parent=md_layout, operand_index=1, k_width=0)
    qk_row_layout: gl.constexpr = gl.SliceLayout(1, qk_mn_layout)
    acc_row_layout: gl.constexpr = gl.SliceLayout(1, md_layout)

    desc_q = tma.make_tensor_descriptor(
        q_seq, shape=[q_len, d], strides=[q_row_stride, 1],
        block_shape=[BLOCK_M, d], layout=q_layout)
    desc_o = tma.make_tensor_descriptor(
        o_seq, shape=[q_len, d], strides=[o_row_stride, 1],
        block_shape=[BLOCK_M, d], layout=o_layout)

    q_smem = gl.allocate_shared_memory(dtype, [BLOCK_M, d], q_layout)
    k_smem = gl.allocate_shared_memory(dtype, [BLOCK_N, d], kv_layout)
    v_smem = gl.allocate_shared_memory(dtype, [BLOCK_N, d], kv_layout)
    o_smem_out = gl.allocate_shared_memory(dtype, [BLOCK_M, d], o_layout)
    bar_q = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    bar_kv = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    nbytes_kv: gl.constexpr = _tile_nbytes(BLOCK_N, d, dtype.primitive_bitwidth) * 2

    mbarrier.init(bar_q, count=1)
    mbarrier.expect(bar_q, _tile_nbytes(BLOCK_M, d, dtype.primitive_bitwidth))
    tma.async_copy_global_to_shared(desc_q, [m_block * BLOCK_M, 0], bar_q, q_smem)
    mbarrier.wait(bar_q, phase=0)
    mbarrier.invalidate(bar_q)

    acc = gl.zeros((BLOCK_M, d), dtype=gl.float32, layout=md_layout)
    rowmax = gl.full([BLOCK_M], float("-inf"), dtype=gl.float32, layout=qk_row_layout)
    rowsum = gl.zeros([BLOCK_M], dtype=gl.float32, layout=qk_row_layout)

    n_block_max = gl.cdiv(k_len, BLOCK_N)
    if is_causal:
        causal_limit = gl.cdiv((m_block + 1) * BLOCK_M + k_len - q_len, BLOCK_N)
        n_block_max = gl.minimum(n_block_max, causal_limit)

    q_tile = q_smem.load(md_layout)
    valid_row = m_block * BLOCK_M + gl.arange(0, BLOCK_M, layout=qk_row_layout) < q_len
    for n_block in range(n_block_max - 1, -1, -1):
        tok_start = n_block * BLOCK_N
        tile_tokens = gl.minimum(k_len - tok_start, BLOCK_N)
        pt_offset = bid * pt_batch_stride + n_block
        page_idx = gl.load(page_table_ptr + pt_offset).to(gl.int32)
        k_page_base = k_ptr + page_idx * k_page_stride + kv_hid * k_head_stride
        v_page_base = v_ptr + page_idx * k_page_stride + kv_hid * k_head_stride
        _sync_load_paged_kv_tile(
            k_page_base, v_page_base, block_size, d, k_row_stride, kv_layout,
            bar_kv, k_smem, v_smem, nbytes_kv, BLOCK_N,
        )
        k_tile = k_smem.load(nd_layout)
        qk_acc = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=qk_mn_layout)
        qk = gl.dot_fma(
            gl.convert_layout(q_tile.to(gl.float32), qk_lhs),
            gl.convert_layout(k_tile.permute(1, 0).to(gl.float32), qk_rhs),
            qk_acc,
        )
        col_idx = gl.arange(0, BLOCK_N)
        qk = gl.where(col_idx[None, :] < tile_tokens, qk, float("-inf"))
        qk = gl.where(valid_row[:, None], qk, float("-inf"))
        if is_causal:
            row_limit = m_block * BLOCK_M + gl.arange(0, BLOCK_M, layout=qk_row_layout) + (k_len - q_len)
            col_global = tok_start + col_idx
            qk = gl.where(col_global[None, :] <= row_limit[:, None], qk, float("-inf"))
        acc, rowmax, rowsum, P = _softmax_step_decode(
            acc, rowmax, rowsum, qk, scale_softmax_log2,
            qk_row_layout, acc_row_layout)
        v_tile = v_smem.load(nd_layout)
        acc = gl.dot_fma(
            gl.convert_layout(P, pv_lhs),
            gl.convert_layout(v_tile.to(gl.float32), pv_rhs),
            acc,
        )

    _decode_epilogue(
        acc, rowmax, rowsum, o_smem_out, desc_o, softmax_lse_ptr,
        m_block, q_bos, q_len, hid, total_q, scale_softmax_log2,
        BLOCK_M, md_layout, qk_row_layout, dtype)


# --- main kernel ---

@libentry()
@gluon.jit
def flash_varlen_fwd_gluon_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr,
    softmax_lse_ptr,
    cu_seqlens_q_ptr,
    cu_seqlens_k_ptr,
    q_row_stride, k_row_stride, v_row_stride, o_row_stride,
    q_head_stride, k_head_stride, v_head_stride, o_head_stride,
    total_q,
    h:           gl.constexpr,
    hk:          gl.constexpr,
    h_hk_ratio:  gl.constexpr,
    d:           gl.constexpr,
    scale_softmax_log2: gl.constexpr,
    is_causal:   gl.constexpr,
    BLOCK_M:     gl.constexpr,
    BLOCK_N:     gl.constexpr,
    num_warps:   gl.constexpr,
    num_stages:  gl.constexpr,
):
    """
    Flash attention forward pass for varlen (cu_seqlens) inputs on Hopper.

    Grid: (cdiv(max_seqlen_q, BLOCK_M), batch, num_heads)

    Uses TMA for loads (runtime shape/strides OK 鈥?no TaskIdPropagation issue)
    and gl.dot for the attention matmuls. Phase 2: warp_specialize producer-consumer added.
    """
    m_block = gl.program_id(0)
    bid     = gl.program_id(1)
    hid     = gl.program_id(2)

    # 鈹€鈹€ sequence lengths 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    q_bos = gl.load(cu_seqlens_q_ptr + bid).to(gl.int32)
    q_eos = gl.load(cu_seqlens_q_ptr + bid + 1).to(gl.int32)
    q_len = q_eos - q_bos

    k_bos = gl.load(cu_seqlens_k_ptr + bid).to(gl.int32)
    k_eos = gl.load(cu_seqlens_k_ptr + bid + 1).to(gl.int32)
    k_len = k_eos - k_bos

    # noop CTA
    if m_block * BLOCK_M >= q_len:
        return

    # 鈹€鈹€ head base pointers 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    q_seq = q_ptr + q_bos * q_row_stride + hid * q_head_stride
    k_seq = k_ptr + k_bos * k_row_stride + (hid // h_hk_ratio) * k_head_stride
    v_seq = v_ptr + k_bos * v_row_stride + (hid // h_hk_ratio) * v_head_stride
    o_seq = o_ptr + q_bos * o_row_stride + hid * o_head_stride

    # 鈹€鈹€ shared memory layouts 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    dtype: gl.constexpr = q_ptr.dtype.element_ty
    q_layout:  gl.constexpr = gl.NVMMASharedLayout.get_default_for([BLOCK_M, d], dtype)
    kv_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for([BLOCK_N, d], dtype)
    o_layout:  gl.constexpr = gl.NVMMASharedLayout.get_default_for([BLOCK_M, d], dtype)

    # 鈹€鈹€ WGMMA accumulator layouts (only pv needed in entry for pv_a_layout) 鈹€鈹€鈹€
    # warps_per_cta and instr chosen to match 05-wgmma.py reference.
    pv_warps_per_cta: gl.constexpr = _pick_warps_per_cta(BLOCK_M, d, num_warps)
    pv_instr_n: gl.constexpr = _pick_instr_n(BLOCK_M, d, num_warps)
    pv_layout: gl.constexpr = gl.NVMMADistributedLayout(
        version=[3, 0],
        warps_per_cta=pv_warps_per_cta,
        instr_shape=[16, pv_instr_n, 256 // dtype.primitive_bitwidth],
    )

    # 鈹€鈹€ TMA descriptors 鈥?runtime shape / ptr is fine in Gluon 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    desc_q = tma.make_tensor_descriptor(
        q_seq, shape=[q_len, d], strides=[q_row_stride, 1],
        block_shape=[BLOCK_M, d], layout=q_layout)
    desc_k = tma.make_tensor_descriptor(
        k_seq, shape=[k_len, d], strides=[k_row_stride, 1],
        block_shape=[BLOCK_N, d], layout=kv_layout)
    desc_v = tma.make_tensor_descriptor(
        v_seq, shape=[k_len, d], strides=[v_row_stride, 1],
        block_shape=[BLOCK_N, d], layout=kv_layout)
    desc_o = tma.make_tensor_descriptor(
        o_seq, shape=[q_len, d], strides=[o_row_stride, 1],
        block_shape=[BLOCK_M, d], layout=o_layout)

    # 鈹€鈹€ shared memory 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    q_smem = gl.allocate_shared_memory(dtype, [BLOCK_M, d], q_layout)
    bar_q  = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar_q,  count=1)

    # P tile in shared memory + DotOperandLayout for PV WGMMA A operand.
    # P comes out of softmax with qk_layout (NVMMADistributedLayout).  To feed
    # it as the LHS of warpgroup_mma(P, v_smem, acc) we must convert it to
    # DotOperandLayout(parent=pv_layout).  The only safe path when qk_layout
    # may differ from pv_layout is the smem round-trip recommended in
    # 05-wgmma.py:  store P 鈫?p_smem  鈫? p_smem.load(pv_a_layout).
    # This avoids a convert_layout across mismatched parent layouts.
    p_smem_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for(
        [BLOCK_M, BLOCK_N], dtype)
    p_smem = gl.allocate_shared_memory(dtype, [BLOCK_M, BLOCK_N], p_smem_layout)
    o_smem_out = gl.allocate_shared_memory(dtype, [BLOCK_M, d], o_layout)
    pv_a_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0,
        parent=pv_layout,
        k_width=32 // dtype.primitive_bitwidth,
    )

    # channel for KV pipeline (Phase 2)
    channel = Channel.alloc(BLOCK_M, BLOCK_N, d, dtype, kv_layout, num_stages, num_warps)

    gl.warp_specialize(
        [
            (compute_partition, (channel, q_smem, p_smem, o_smem_out, desc_o, o_ptr, softmax_lse_ptr, q_ptr, desc_q, bar_q, cu_seqlens_q_ptr, cu_seqlens_k_ptr, q_row_stride, k_row_stride, v_row_stride, o_row_stride, q_head_stride, k_head_stride, v_head_stride, o_head_stride, total_q, h, hk, h_hk_ratio, d, scale_softmax_log2, is_causal, BLOCK_M, BLOCK_N, q_len, k_len, hid, num_warps)),
            (load_partition, (channel, k_ptr, v_ptr, desc_k, desc_v, cu_seqlens_q_ptr, cu_seqlens_k_ptr, q_row_stride, k_row_stride, v_row_stride, q_head_stride, k_head_stride, v_head_stride, h, hk, h_hk_ratio, d, is_causal, BLOCK_M, BLOCK_N, q_len, k_len, num_warps)),
        ],
        [1],
        [24],
    )

    channel.release()


# 鈹€鈹€ Python launcher 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

_allocator_registered = False


def _ensure_allocator():
    global _allocator_registered
    if not _allocator_registered:
        def _alloc(size, align, stream):
            return torch.empty(size, device="cuda", dtype=torch.uint8)
        triton.set_allocator(_alloc)
        _allocator_registered = True


def flash_varlen_fwd_gluon(
    q, k, v, o,
    softmax_lse,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float,
    is_causal: bool,
    BLOCK_M: int = 0,  # 0 = auto-select based on smem budget for d
    BLOCK_N: int = 64,
    num_warps: int = 4,
    num_stages: int = 2,
    maxnreg: int = 256,
):
    """
    Flash attention forward for packed/varlen inputs on Hopper GPUs.

    q:            [total_q, h,  d]  contiguous
    k:            [total_k, hk, d]  contiguous
    v:            [total_k, hk, d]  contiguous
    o:            [total_q, h,  d]  pre-allocated output
    softmax_lse:  [h, total_q]      pre-allocated
    cu_seqlens_q: [batch+1] int32
    cu_seqlens_k: [batch+1] int32
    num_stages:   number of pipeline stages for warp_specialize (producer-consumer)
    """
    _ensure_allocator()

    total_q, h,  d = q.shape
    total_k, hk, _ = k.shape
    batch = cu_seqlens_q.shape[0] - 1
    h_hk_ratio = h // hk

    scale_log2 = softmax_scale * math.log2(math.e)

    # Decode path: dot_fma kernel, max_seqlen_q <= 8.
    if _is_decode_mode(max_seqlen_q):
        BLOCK_M = min(_DECODE_BLOCK_M, max_seqlen_q)
        BLOCK_N = _decode_block_n(d)
        grid = (triton.cdiv(max_seqlen_q, BLOCK_M), batch, h)
        flash_varlen_fwd_gluon_decode_kernel[grid](
            q, k, v, o,
            softmax_lse,
            cu_seqlens_q,
            cu_seqlens_k,
            q.stride(0), k.stride(0), v.stride(0), o.stride(0),
            q.stride(1), k.stride(1), v.stride(1), o.stride(1),
            total_q,
            h, hk, h_hk_ratio, d,
            scale_log2,
            is_causal,
            BLOCK_M, BLOCK_N,
            num_warps=4,
            maxnreg=128,
        )
        return o, softmax_lse

    if BLOCK_N == 0:
        # auto BLOCK_N: smaller for large d to keep tile compute balanced
        BLOCK_N = 32 if d >= 256 else (128 if d <= 64 else 64)

    elem_bytes = q.element_size()
    # heuristic for num_stages: larger pipeline helps when d small and KV long
    if num_stages == 2 and d <= 128 and max_seqlen_k >= 2048:
        num_stages = 3
    if max_seqlen_q <= 1:
        num_stages = 1  # reduce pipeline cost for pure decode
    if BLOCK_M == 0:
        try:
            picked = _pick_nonpaged_block_m(d, elem_bytes, num_stages, block_n=BLOCK_N)
            BLOCK_M = picked
        except RuntimeError:
            if num_stages > 1:
                num_stages = 1
                BLOCK_M = _pick_nonpaged_block_m(d, elem_bytes, num_stages, block_n=BLOCK_N)
            else:
                raise
    else:
        picked = _pick_nonpaged_block_m(d, elem_bytes, num_stages, block_n=BLOCK_N)
        if BLOCK_M > picked:
            BLOCK_M = picked  # cap to avoid smem OOM for large d

    # decode / small max q specialization: smaller BLOCK_M reduces tile overhang,
    # garbage row compute in Q tile, and Q/O smem for cases where q_len=1 or small
    if max_seqlen_q <= 64:
        BLOCK_M = min(BLOCK_M, 64)
    if max_seqlen_q <= 32:
        BLOCK_M = 32

    if maxnreg < 256:
        maxnreg = 256

    scale_log2 = softmax_scale * math.log2(math.e)

    grid = (triton.cdiv(max_seqlen_q, BLOCK_M), batch, h)

    flash_varlen_fwd_gluon_kernel[grid](
        q, k, v, o,
        softmax_lse,
        cu_seqlens_q,
        cu_seqlens_k,
        q.stride(0), k.stride(0), v.stride(0), o.stride(0),
        q.stride(1), k.stride(1), v.stride(1), o.stride(1),
        total_q,
        h, hk, h_hk_ratio, d,
        scale_log2,
        is_causal,
        BLOCK_M, BLOCK_N,
        num_warps=num_warps,
        num_stages=num_stages,
        maxnreg=maxnreg,
    )
    return o, softmax_lse


# 鈹€鈹€ Paged KV kernel 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
#
# paged KV layout:
#   k / v : [num_pages, block_size, num_heads_k, head_size]
#   page_table : [batch, max_pages_per_seq]  (int32)
#
# For each KV block of BLOCK_N tokens, we may span across page boundaries.
# We process BLOCK_N tokens by iterating one page-block at a time inside a
# sub-loop, building a runtime TMA descriptor per physical page.  This avoids
# the need for a contiguous K/V layout while still using TMA.
#
# To keep complexity manageable (Phase 1), BLOCK_N == block_size is required
# (one BLOCK_N tile == one physical page). This is the common deployment
# case (block_size = 16/32/64).  When BLOCK_N != block_size, the launcher
# falls back to the FA3 paged kernel.

@libentry()
@gluon.jit
def flash_paged_fwd_gluon_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr,
    softmax_lse_ptr,
    cu_seqlens_q_ptr,
    cu_seqlens_k_ptr,
    page_table_ptr,
    # strides
    q_row_stride, q_head_stride,
    o_row_stride, o_head_stride,
    # k/v paged strides: [num_pages, block_size, hk, d]
    k_page_stride,    # stride(0): between pages
    k_row_stride,     # stride(1): between rows within a page
    k_head_stride,    # stride(2): between heads
    # page_table strides
    pt_batch_stride,  # stride(0): between batches
    # sizes
    total_q,
    h:            gl.constexpr,
    hk:           gl.constexpr,
    h_hk_ratio:   gl.constexpr,
    d:            gl.constexpr,
    block_size:   gl.constexpr,   # physical page block size (== BLOCK_N)
    scale_softmax_log2: gl.constexpr,
    is_causal:    gl.constexpr,
    BLOCK_M:      gl.constexpr,
    BLOCK_N:      gl.constexpr,   # must equal block_size
    num_warps:    gl.constexpr,
    num_stages:   gl.constexpr,
):
    """
    Flash attention forward for paged KV on Hopper.

    Grid: (cdiv(max_seqlen_q, BLOCK_M), batch, num_heads)

    BLOCK_N == block_size: each KV tile maps to exactly one physical page.
    TMA descriptor for K/V is constructed per tile using the page_table lookup.
    num_stages for warp_specialize pipeline.
    """
    m_block = gl.program_id(0)
    bid     = gl.program_id(1)
    hid     = gl.program_id(2)

    q_bos = gl.load(cu_seqlens_q_ptr + bid).to(gl.int32)
    q_eos = gl.load(cu_seqlens_q_ptr + bid + 1).to(gl.int32)
    q_len = q_eos - q_bos

    k_bos = gl.load(cu_seqlens_k_ptr + bid).to(gl.int32)
    k_eos = gl.load(cu_seqlens_k_ptr + bid + 1).to(gl.int32)
    k_len = k_eos - k_bos

    if m_block * BLOCK_M >= q_len:
        return

    q_seq = q_ptr + q_bos * q_row_stride + hid * q_head_stride
    o_seq = o_ptr + q_bos * o_row_stride + hid * o_head_stride

    dtype: gl.constexpr = q_ptr.dtype.element_ty
    q_layout:  gl.constexpr = gl.NVMMASharedLayout.get_default_for([BLOCK_M, d], dtype)
    kv_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for([BLOCK_N, d], dtype)
    o_layout:  gl.constexpr = gl.NVMMASharedLayout.get_default_for([BLOCK_M, d], dtype)

    pv_warps_per_cta: gl.constexpr = _pick_warps_per_cta(BLOCK_M, d, num_warps)
    pv_instr_n:       gl.constexpr = _pick_instr_n(BLOCK_M, d, num_warps)
    pv_layout: gl.constexpr = gl.NVMMADistributedLayout(
        version=[3, 0],
        warps_per_cta=pv_warps_per_cta,
        instr_shape=[16, pv_instr_n, 256 // dtype.primitive_bitwidth],
    )

    desc_q = tma.make_tensor_descriptor(
        q_seq, shape=[q_len, d], strides=[q_row_stride, 1],
        block_shape=[BLOCK_M, d], layout=q_layout)
    desc_o = tma.make_tensor_descriptor(
        o_seq, shape=[q_len, d], strides=[o_row_stride, 1],
        block_shape=[BLOCK_M, d], layout=o_layout)

    q_smem = gl.allocate_shared_memory(dtype, [BLOCK_M, d], q_layout)
    p_smem_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for(
        [BLOCK_M, BLOCK_N], dtype)
    p_smem = gl.allocate_shared_memory(dtype, [BLOCK_M, BLOCK_N], p_smem_layout)
    o_smem_out = gl.allocate_shared_memory(dtype, [BLOCK_M, d], o_layout)

    bar_q  = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar_q,  count=1)

    pv_a_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=pv_layout,
        k_width=32 // dtype.primitive_bitwidth)

    channel = Channel.alloc(BLOCK_M, BLOCK_N, d, dtype, kv_layout, num_stages, num_warps)

    gl.warp_specialize(
        [
            (compute_paged_partition, (channel, q_smem, p_smem, o_smem_out, desc_o, o_ptr, softmax_lse_ptr, q_ptr, desc_q, bar_q, cu_seqlens_q_ptr, cu_seqlens_k_ptr, page_table_ptr, q_row_stride, q_head_stride, o_row_stride, o_head_stride, k_page_stride, k_row_stride, k_head_stride, pt_batch_stride, total_q, h, hk, h_hk_ratio, d, block_size, scale_softmax_log2, is_causal, BLOCK_M, BLOCK_N, q_len, k_len, hid, num_warps)),
            (load_paged_partition, (channel, k_ptr, v_ptr, page_table_ptr, cu_seqlens_q_ptr, cu_seqlens_k_ptr, q_row_stride, k_row_stride, k_row_stride, q_head_stride, k_head_stride, k_head_stride, pt_batch_stride, k_page_stride, k_head_stride, h, hk, h_hk_ratio, d, block_size, is_causal, BLOCK_M, BLOCK_N, q_len, k_len, num_warps)),
        ],
        [1],
        [24],
    )

    channel.release()


def _pick_paged_block_m(block_size: int, head_size: int,
                        elem_bytes: int = 2,
                        num_stages: int = 2,
                        smem_limit: int = 232448) -> int:
    """Choose the largest BLOCK_M (power-of-two 鈮?32) such that the paged
    gluon kernel's shared-memory footprint with pipeline stays within *smem_limit* bytes.

    Actual:
      Q tile : BLOCK_M 脳 head 脳 elem
      O tile : BLOCK_M 脳 head 脳 elem
      P tile : BLOCK_M 脳 block_size 脳 elem   (fp16)
      KV staged : num_stages 脳 2 脳 block_size 脳 head 脳 elem
    """
    for bm in (128, 64, 32):
        qo = 2 * bm * head_size * elem_bytes
        pp = bm * block_size * elem_bytes
        kv = num_stages * 2 * block_size * head_size * elem_bytes
        smem = qo + pp + kv + 128
        if smem <= smem_limit:
            return bm
    raise RuntimeError(
        f"No valid BLOCK_M found for block_size={block_size}, "
        f"head_size={head_size}, num_stages={num_stages}: even BLOCK_M=32 exceeds smem limit {smem_limit}"
    )


def _pick_nonpaged_block_m(head_size: int, elem_bytes: int = 2, num_stages: int = 2, smem_limit: int = 232448, block_n: int = 64) -> int:
    """Choose the largest BLOCK_M (power-of-two 鈮?32) such that the non-paged
    gluon kernel's shared-memory footprint with warp-specialize pipeline stays
    within *smem_limit* bytes.

    Accounts for:
      Q tile : BLOCK_M 脳 head_size 脳 elem_bytes
      O tile : BLOCK_M 脳 head_size 脳 elem_bytes
      P tile : BLOCK_M 脳 block_n 脳 elem_bytes   (fp16)
      KV pipeline : num_stages 脳 2 脳 block_n 脳 head_size 脳 elem_bytes
    """
    for bm in (128, 64, 32):
        qo = 2 * bm * head_size * elem_bytes
        pp = bm * block_n * elem_bytes
        kv = num_stages * 2 * block_n * head_size * elem_bytes
        smem = qo + pp + kv + 128  # bars, descriptors, misc
        if smem <= smem_limit:
            return bm
    raise RuntimeError(
        f"No valid BLOCK_M found for head_size={head_size}, block_n={block_n}, "
        f"num_stages={num_stages}: even BLOCK_M=32 exceeds smem limit {smem_limit}"
    )


def flash_paged_fwd_gluon(
    q, k, v, o,
    softmax_lse,
    cu_seqlens_q,
    cu_seqlens_k,
    page_table,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float,
    is_causal: bool,
    BLOCK_M: int = 0,       # 0 = auto-select based on smem budget
    num_warps: int = 4,
    num_stages: int = 2,
    maxnreg: int = 256,
):
    """
    Flash attention forward for paged KV on Hopper.

    q:            [total_q, h,  d]          contiguous
    k:            [num_pages, block_size, hk, d]  paged KV
    v:            [num_pages, block_size, hk, d]
    o:            [total_q, h,  d]          pre-allocated
    softmax_lse:  [h, total_q]              pre-allocated
    cu_seqlens_q: [batch+1] int32
    cu_seqlens_k: [batch+1] int32
    page_table:   [batch, max_pages_per_seq] int32

    Constraint: BLOCK_N is set to k.shape[1] (block_size). The kernel loads
    exactly one physical page per KV tile.  BLOCK_M is chosen automatically
    when left at 0 to keep the shared-memory footprint within hardware limits.

    num_stages: pipeline stages for warp_specialize.
    """
    _ensure_allocator()

    total_q, h, d = q.shape
    num_pages, block_size, hk, _ = k.shape
    batch     = cu_seqlens_q.shape[0] - 1
    h_hk_ratio = h // hk
    BLOCK_N   = block_size   # one tile == one page

    scale_log2 = softmax_scale * math.log2(math.e)

    # Decode path: dot_fma kernel, max_seqlen_q <= 8.
    if _is_decode_mode(max_seqlen_q):
        BLOCK_M = min(_DECODE_BLOCK_M, max_seqlen_q)
        grid = (triton.cdiv(max_seqlen_q, BLOCK_M), batch, h)
        flash_paged_fwd_gluon_decode_kernel[grid](
            q, k, v, o,
            softmax_lse,
            cu_seqlens_q,
            cu_seqlens_k,
            page_table,
            q.stride(0), q.stride(1),
            o.stride(0), o.stride(1),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            page_table.stride(0),
            total_q,
            h, hk, h_hk_ratio, d, block_size,
            scale_log2,
            is_causal,
            BLOCK_M, BLOCK_N,
            num_warps=4,
            maxnreg=128,
        )
        return o, softmax_lse

    elem_bytes = q.element_size()
    # heuristic for num_stages: larger pipeline helps when d small and KV long
    if num_stages == 2 and d <= 128 and max_seqlen_k >= 2048:
        num_stages = 3
    if max_seqlen_q <= 1:
        num_stages = 1  # reduce pipeline cost for pure decode
    if BLOCK_M == 0:
        try:
            BLOCK_M = _pick_paged_block_m(block_size, d, elem_bytes, num_stages)
        except RuntimeError:
            if num_stages > 1:
                num_stages = 1
                BLOCK_M = _pick_paged_block_m(block_size, d, elem_bytes, num_stages)
            else:
                raise

    # cap if user-provided BLOCK_M exceeds safe for (possibly reduced) num_stages
    max_safe = _pick_paged_block_m(block_size, d, elem_bytes, num_stages)
    if BLOCK_M > max_safe:
        BLOCK_M = max_safe

    # decode / small max q specialization: smaller BLOCK_M reduces tile overhang,
    # garbage row compute in Q tile, and Q/O smem for cases where q_len=1 or small
    if max_seqlen_q <= 64:
        BLOCK_M = min(BLOCK_M, 64)
    if max_seqlen_q <= 32:
        BLOCK_M = 32
    # re-cap after possible reduction from small-q rule
    max_safe = _pick_paged_block_m(block_size, d, elem_bytes, num_stages)
    if BLOCK_M > max_safe:
        BLOCK_M = max_safe

    if maxnreg < 256:
        maxnreg = 256

    scale_log2 = softmax_scale * math.log2(math.e)
    grid = (triton.cdiv(max_seqlen_q, BLOCK_M), batch, h)

    flash_paged_fwd_gluon_kernel[grid](
        q, k, v, o,
        softmax_lse,
        cu_seqlens_q,
        cu_seqlens_k,
        page_table,
        # Q/O strides
        q.stride(0), q.stride(1),
        o.stride(0), o.stride(1),
        # K/V paged strides
        k.stride(0),  # k_page_stride
        k.stride(1),  # k_row_stride  (within page)
        k.stride(2),  # k_head_stride
        # page_table strides
        page_table.stride(0),
        # sizes
        total_q,
        h, hk, h_hk_ratio, d, block_size,
        scale_log2,
        is_causal,
        BLOCK_M, BLOCK_N,
        num_warps=num_warps,
        num_stages=num_stages,
        maxnreg=maxnreg,
    )
    return o, softmax_lse
