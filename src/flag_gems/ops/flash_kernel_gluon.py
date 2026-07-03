"""
flash_kernel_gluon.py — Gluon FA3 varlen forward for Hopper.

Why Gluon?
----------
Triton FA3 disables warp_specialize when cu_seqlens_k / seqused_k are present because
runtime-derived TMA descriptor bases break TaskIdPropagation.  Gluon's device-side
tma.make_tensor_descriptor accepts runtime base/shape/strides.

File layout (see flash_kernel_gluon_文件结构.md):
  §0  imports / constants
  §1  Host utilities (tile selection, layout factories)
  §2  Device utilities (seqlen, softmax, sync helpers)
  §3  Warp-spec partitions (load / compute)
  §4  Kernel entry points (@gluon.jit)
  §5  Public launcher (flash_attn_varlen_gluon_fwd)

Implementation status:
  - flash_varlen_fwd_gluon_kernel: non-paged TMA + warp_specialize + packGQA (MVP grid)
  - flash_varlen_fwd_gluon_persistent_kernel: static persistent (grid=num_sm) when tiles >> SM
  - flash_varlen_decode_gluon_kernel: non-paged decode (q_len <= thresh) Triton dot path
  - flash_paged_fwd_gluon_kernel: paged cp.async (+ optional TMA) + warp_specialize + packGQA
  - split-KV (num_splits>1, combine, partial epilogue): commented out [splitkv-disabled]
  - persistent scheduler kernels/helpers: commented out (MVP-only rollback)
"""
from __future__ import annotations

import math

import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.ampere import async_copy as cp
from triton.experimental.gluon.language.nvidia.hopper import (
    fence_async_shared,
    mbarrier,
    tma,
    warpgroup_mma,
    warpgroup_mma_wait,
)
from triton.language.core import _aggregate as aggregate

from flag_gems.ops.get_scheduler_metadata import (
    # [splitkv-disabled] _vllm_num_splits_heuristic,
    get_pagedkv_tma,
    round_up_headdim,
    round_up_headdimv,
    tile_size_fwd_sm90,
    use_one_mma_wg,
)
from flag_gems.utils import libentry

# =============================================================================
# §0  imports / constants
# =============================================================================

_SMEM_LIMIT = 232_448  # Hopper per-CTA shared memory budget (~228 KiB)
# Target consumer warps (= BLOCK_M // 64 * 4). 8 => BLOCK_M 128 if smem allows.
_GLUON_VARLEN_TARGET_CONSUMER_WARPS = 8
_GLUON_DECODE_Q_THRESH = 4
# decode dot 路径仅用于短 KV；长 KV 即 q_len 小也走 WGMMA（TMA 扫 N 维）
_GLUON_DECODE_KV_THRESH = 512
# 临时实验：non-paged 全部走 prefill WGMMA，decode 路径保留不调用
_GLUON_PREFILL_ONLY = True
# 静态 persistent：total_tiles > num_sm * ratio 时启用，grid=num_sm stride 领活
_GLUON_STATIC_PERSISTENT_ENABLED = False
_GLUON_STATIC_PERSISTENT_TILE_RATIO = 2
_M_LOG2E = math.log2(math.e)

_allocator_registered = False


# =============================================================================
# §1  Host utilities
# =============================================================================

@gluon.constexpr_function
def _tile_nbytes(rows, cols, elem_bits):
    return rows * cols * (elem_bits // 8)


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def _decode_block_n(head_size: int) -> int:
    return 32 if head_size >= 256 else 64


def _pick_block_n(d: int, *, is_decode: bool) -> int:
    if is_decode:
        return _decode_block_n(d)
    if d >= 256:
        return 32
    if d <= 64:
        return 128
    return 64


def _pick_block_m(
    d: int,
    *,
    is_paged: bool,
    block_size: int = 1,
    block_n: int = 64,
    elem_bytes: int = 2,
    num_stages: int = 2,
    smem_limit: int = _SMEM_LIMIT,
) -> int:
    """Largest power-of-two BLOCK_M (32/64/128) that fits smem budget."""
    if is_paged:
        for bm in (128, 64, 32):
            qo = 2 * bm * d * elem_bytes
            pp = bm * block_size * elem_bytes
            kv = num_stages * 2 * block_size * d * elem_bytes
            if qo + pp + kv + 128 <= smem_limit:
                return bm
    else:
        for bm in (128, 64, 32):
            qo = 2 * bm * d * elem_bytes
            pp = bm * block_n * elem_bytes
            kv = num_stages * 2 * block_n * d * elem_bytes
            if qo + pp + kv + 128 <= smem_limit:
                return bm
    raise RuntimeError(
        f"No valid BLOCK_M for d={d}, is_paged={is_paged}, block_size={block_size}"
    )


def _round_up_pow2(n: int) -> int:
    """Round up to power of 2 (Triton launch num_warps + Gluon NVMMA smem tiles)."""
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def _round_down_pow2(n: int) -> int:
    """Round down to power of 2."""
    if n <= 1:
        return 1
    return 1 << (n.bit_length() - 1)


def _pick_sm90_varlen_tile_config(
    d: int,
    max_seqlen_q: int,
    is_causal: bool,
    *,
    is_paged: bool,
    use_kv_tma: bool,
    pack_gqa: bool,
    num_heads: int,
    num_heads_k: int,
    dv: int | None = None,
    is_local: bool = False,
) -> tuple[int, int, int, int, int]:
    """
    Mirror vLLM SM90 varlen tile + warpgroup layout.

    Consumer warpgroups = BLOCK_M // 64 (AtomLayoutQK in mainloop_fwd_sm90).
    Returns (block_m, block_n, num_warps, producer_warps, consumer_warps).
    Triton launch num_warps = consumer warps (default partition); worker
    producer_warps are extra (ttg.total-num-warps).  Layout/MMA use consumer only.
    block_m/block_n are rounded up to power-of-2 for Gluon NVMMA smem.
    """
    dv = d if dv is None else dv
    d_rounded = round_up_headdim(d)
    dv_rounded = round_up_headdimv(dv)
    paged_kv_non_tma = is_paged and not use_kv_tma
    uomw = use_one_mma_wg(
        90, d_rounded, max_seqlen_q, pack_gqa, num_heads, num_heads_k,
    )
    block_m, block_n = tile_size_fwd_sm90(
        d_rounded,
        dv_rounded,
        is_causal,
        is_local,
        element_size=2,
        paged_kv_non_TMA=paged_kv_non_tma,
        softcap=False,
        use_one_mma_wg=uomw,
    )
    # Gluon NVMMA shared memory requires power-of-2 tile dims (vLLM may use 176, etc.).
    block_m = _round_up_pow2(block_m)
    block_n = _round_up_pow2(block_n)
    num_consumer_wgs = block_m // 64
    consumer_warps = num_consumer_wgs * 4
    # PagedKVNonTMA / packGQA: producer runs as a full warpgroup (cp.async/TMA KV).
    producer_warps = 4 if (paged_kv_non_tma or pack_gqa) else 1
    num_warps = producer_warps + consumer_warps
    return block_m, block_n, num_warps, producer_warps, consumer_warps


def _pick_maxnreg_sm90(num_consumer_wgs: int, producer_warps: int) -> int | None:
    """
    Optional module-level maxnreg for Gluon launch.

    vLLM uses per-warpgroup setmaxnreg (producer ~24-40, consumer 232/256).
    A global ttg.maxnreg=232 with 8+4=12 physical warps exceeds the 64K/SM
    budget and CUDA reports MAX_THREADS_PER_BLOCK=256.  Let AllocateWarpGroups
    derive per-partition limits when running 2 consumer warpgroups.
    """
    total_warps = num_consumer_wgs * 4 + producer_warps
    if total_warps > 8:
        return None
    if num_consumer_wgs <= 1:
        return 256
    return 232


def _resolve_varlen_launch_params(
    *,
    d: int,
    max_seqlen_q: int,
    max_seqlen_k: int,
    batch: int,
    num_heads: int,
    num_heads_k: int,
    h_hk_ratio: int,
    is_causal: bool,
    is_paged: bool,
    use_kv_tma: bool,
    device: torch.device,
    elem_bytes: int,
    num_stages: int = 2,
    block_size: int | None = None,
) -> dict:
    """Host prep shared by non-paged / paged MVP launch."""
    pack_gqa = _get_pack_gqa(
        num_heads, num_heads_k, max_seqlen_q, 128,
        has_page_table=is_paged, use_kv_tma=use_kv_tma, varlen_q=True,
    )
    block_m, block_n, num_warps, producer_warps, consumer_warps = (
        _pick_sm90_varlen_tile_config(
            d, max_seqlen_q, is_causal,
            is_paged=is_paged, use_kv_tma=use_kv_tma, pack_gqa=pack_gqa,
            num_heads=num_heads, num_heads_k=num_heads_k,
        )
    )
    pack_gqa = _get_pack_gqa(
        num_heads, num_heads_k, max_seqlen_q, block_m,
        has_page_table=is_paged, use_kv_tma=use_kv_tma, varlen_q=True,
    )
    num_sm = _get_num_sm(device)
    num_splits = 1
    # [splitkv-disabled] num_splits = _get_num_splits_gluon(
    # [splitkv-disabled]     batch, num_heads_k, max_seqlen_q, max_seqlen_k,
    # [splitkv-disabled]     block_m, block_n, pack_gqa, h_hk_ratio, num_sm, is_causal, d, elem_bytes,
    # [splitkv-disabled] )
    # [splitkv-disabled] if num_splits > 1:
    # [splitkv-disabled]     pack_gqa = True
    block_m, block_n, num_warps, producer_warps, consumer_warps = (
        _pick_sm90_varlen_tile_config(
            d, max_seqlen_q, is_causal,
            is_paged=is_paged, use_kv_tma=use_kv_tma, pack_gqa=pack_gqa,
            num_heads=num_heads, num_heads_k=num_heads_k,
        )
    )
    paged_kv_non_tma = is_paged and not use_kv_tma
    producer_warps = 4 if (paged_kv_non_tma or pack_gqa) else 1
    d_nvmma = _round_up_pow2(d)
    d_rounded = round_up_headdim(d)
    uomw = use_one_mma_wg(
        90, d_rounded, max_seqlen_q, pack_gqa, num_heads, num_heads_k,
    )
    # Decode / short-q: keep tile_size_fwd_sm90 uomw BLOCK_M=64; do not boost to 128.
    if not uomw:
        target_bm = (
            64 if d_nvmma > d else (_GLUON_VARLEN_TARGET_CONSUMER_WARPS // 4) * 64
        )
        block_m = max(block_m, target_bm)
    if d_nvmma > d:
        block_m = min(block_m, 64)
    if is_paged and block_size is not None:
        block_n = min(block_n, _round_down_pow2(block_size))
    block_m, block_n, num_stages = _fit_gluon_varlen_tiles(
        block_m, block_n, d_nvmma, elem_bytes, num_stages,
    )
    launch_num_warps, consumer_warps = _gluon_varlen_consumer_warps(block_m)
    num_consumer_wgs = block_m // 64
    n_heads_grid = num_heads_k if pack_gqa else num_heads
    return {
        "block_m": block_m,
        "block_n": block_n,
        "num_warps": launch_num_warps,
        "launch_num_warps": launch_num_warps,
        "producer_warps": producer_warps,
        "consumer_warps": consumer_warps,
        "pack_gqa": pack_gqa,
        "num_splits": num_splits,
        "num_stages": num_stages,
        "num_sm": num_sm,
        "n_heads_grid": n_heads_grid,
        "maxnreg": _pick_maxnreg_sm90(num_consumer_wgs, producer_warps),
    }


@gluon.constexpr_function
def _pick_warps_per_cta(block_m, block_n, num_warps):
    wpc = [4, 1]
    m = 16
    while wpc[0] * wpc[1] != num_warps:
        if block_m > m * wpc[0]:
            wpc[0] *= 2
        else:
            wpc[1] *= 2
    return wpc


@gluon.constexpr_function
def _pick_instr_n(block_m, block_n, num_warps):
    m = 16
    m_reps = triton.cdiv(block_m, m)
    n_reps = triton.cdiv(num_warps, m_reps)
    max_n = max(block_n // n_reps, 8)
    n = 256
    while n > max_n or block_n % n != 0:
        n -= 8
    return n


def _should_pack_gqa(
    varlen_q: bool,
    seqlen_q: int,
    qhead_per_khead: int,
    block_m: int,
) -> bool:
    """Mirror hopper/heuristics.h::should_pack_gqa."""
    if varlen_q:
        return True

    def _round_up(a: int, b: int) -> int:
        return (a + b - 1) // b * b

    nopack_eff = float(seqlen_q) / float(_round_up(seqlen_q, block_m))
    pack_eff = float(seqlen_q * qhead_per_khead) / float(
        _round_up(seqlen_q * qhead_per_khead, block_m)
    )
    return nopack_eff < 0.9 * pack_eff


def _get_pack_gqa(
    num_heads: int,
    num_heads_k: int,
    max_seqlen_q: int,
    block_m: int,
    *,
    has_page_table: bool = False,
    use_kv_tma: bool = False,
    varlen_q: bool = True,
) -> bool:
    """Mirror hopper/flash_api.cpp::get_pack_gqa (SM90 Gluon subset)."""
    if num_heads == num_heads_k:
        return False
    # PagedKVNonTMA always packs (vLLM compiles fewer templates).
    if has_page_table and not use_kv_tma:
        return True
    qhead_per_khead = num_heads // num_heads_k
    return _should_pack_gqa(varlen_q, max_seqlen_q, qhead_per_khead, block_m)


def _should_use_paged_tma(
    block_size: int,
    max_seqlen_q: int,
    num_heads: int,
    num_heads_k: int,
    d: int,
    *,
    block_n: int | None = None,
) -> bool:
    """Mirror vLLM get_pagedkv_tma() heuristic (simplified)."""
    block_n = block_n or block_size
    total_q_rows = max_seqlen_q * (num_heads // num_heads_k)
    return block_size % block_n == 0 and total_q_rows > block_n


def _smem_bytes(
    block_m: int,
    block_n: int,
    d: int,
    *,
    elem_bytes: int = 2,
    num_stages: int = 2,
) -> int:
    qo = 2 * block_m * d * elem_bytes
    pp = block_m * block_n * elem_bytes
    kv = num_stages * 2 * block_n * d * elem_bytes
    return qo + pp + kv + 128


def _gluon_varlen_smem_bytes(
    block_m: int,
    block_n: int,
    d: int,
    *,
    elem_bytes: int = 2,
    num_stages: int = 2,
) -> int:
    """MVP Gluon varlen smem: Q/O + P + f32 QK staging + pipelined KV + barriers."""
    qo = 2 * block_m * d * elem_bytes
    pp = block_m * block_n * elem_bytes
    qk = block_m * block_n * 4
    kv = num_stages * 2 * block_n * d * elem_bytes
    return qo + pp + qk + kv + 2048


def _gluon_varlen_consumer_warps(block_m: int) -> tuple[int, int]:
    """
    Map BLOCK_M to consumer (default-partition) warp count for warp_specialize.

    launch num_warps must be a power-of-2 multiple of 4 and match the consumer
    warpgroups (BLOCK_M // 64, 4 warps each).  Worker producer_warps are extra.
    """
    bm = _round_up_pow2(block_m)
    if bm < 64:
        raise RuntimeError(f"BLOCK_M must be >= 64 for WGMMA, got {block_m}")
    consumer_warps = (bm // 64) * 4
    launch_warps = _round_up_pow2(consumer_warps)
    if launch_warps != consumer_warps:
        raise RuntimeError(
            f"Consumer warps {consumer_warps} rounds to launch {launch_warps}; "
            f"BLOCK_M={block_m} unsupported for Gluon layout"
        )
    return launch_warps, consumer_warps


def _fit_gluon_varlen_tiles(
    block_m: int,
    block_n: int,
    d: int,
    elem_bytes: int,
    num_stages: int,
    *,
    smem_limit: int = _SMEM_LIMIT,
) -> tuple[int, int, int]:
    """
    Shrink vLLM tile sizes to fit Hopper smem budget.

    Gluon NVMMA needs pow2 tiles; vLLM may pick block_n=176 which we must not
    round up to 256 blindly.  Prefer lowering num_stages, then block_n/block_m.
    """
    bm = _round_up_pow2(block_m)
    bn_up = _round_up_pow2(block_n)
    bn_opts: list[int] = []
    n = bn_up
    # paged block_size can be 16; bn must go down to 16 not 32 only.
    while n >= 16:
        bn_opts.append(n)
        n //= 2

    bm_opts = []
    for candidate in (128, 64, 32):
        if candidate <= bm:
            bm_opts.append(candidate)
    if not bm_opts:
        bm_opts = [32]
    ns_opts = [1]
    if num_stages > 1:
        ns_opts.append(num_stages)
    for bm_try in bm_opts:
        for ns in ns_opts:
            for bn_try in bn_opts:
                if (
                    _gluon_varlen_smem_bytes(
                        bm_try, bn_try, d, elem_bytes=elem_bytes, num_stages=ns,
                    )
                    <= smem_limit
                ):
                    return bm_try, bn_try, ns
    raise RuntimeError(
        f"No Gluon varlen tile fits smem: bm={block_m} bn={block_n} d={d} "
        f"elem_bytes={elem_bytes} stages={num_stages} limit={smem_limit}"
    )


@gluon.constexpr_function
def _nvmma_d(d: gl.constexpr) -> gl.constexpr:
    """NVMMA/TMA block tiles require power-of-2 head dims (e.g. 192 -> 256)."""
    if d <= 64:
        return 64
    if d <= 128:
        return 128
    return 256


@gluon.constexpr_function
def _nvmma_kv_layout(block_n, d, dtype):
    return gl.NVMMASharedLayout.get_default_for([block_n, _nvmma_d(d)], dtype)


@gluon.constexpr_function
def _nvmma_qo_layout(block_m, d, dtype):
    return gl.NVMMASharedLayout.get_default_for([block_m, _nvmma_d(d)], dtype)


@gluon.constexpr_function
def _scalar_i32_smem_layout():
    return gl.NVMMASharedLayout.get_default_for([1], gl.int32)


@gluon.constexpr_function
def _blocked_mn_layout(num_warps):
    """Blocked [M, N] layout for softmax / reductions on consumer warps."""
    return gl.BlockedLayout([1, 1], [1, 32], [num_warps, 1], [1, 0])


@gluon.constexpr_function
def _cpasync_blocked_layout(num_warps, elem_bits):
    """Blocked layout for cp.async: each thread moves >= 4 bytes."""
    elem_bytes = elem_bits // 8
    cols_per_transfer = max(1, 4 // elem_bytes)
    return gl.BlockedLayout(
        [1, cols_per_transfer], [1, 32], [num_warps, 1], [1, 0],
    )


@gluon.constexpr_function
def _qk_smem_layout(block_m, block_n):
    return gl.NVMMASharedLayout.get_default_for([block_m, block_n], gl.float32)


@gluon.jit
def _qk_mma_to_blocked(qk_mma, qk_smem, blocked_layout):
    """Round-trip WGMMA QK through smem for row/column reductions."""
    qk_smem.store(qk_mma)
    fence_async_shared()
    return qk_smem.load(blocked_layout)


@gluon.constexpr_function
def _decode_blocked_layout(num_warps, rows, d):
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


def _ensure_allocator() -> None:
    global _allocator_registered
    if _allocator_registered:
        return

    def _alloc(size, align, stream):
        return torch.empty(size, device="cuda", dtype=torch.uint8)

    triton.set_allocator(_alloc)
    _allocator_registered = True


def _scale_softmax_log2(softmax_scale: float) -> float:
    return softmax_scale * _M_LOG2E


def _split_varlen_batch_ids(
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    q_threshold: int = _GLUON_DECODE_Q_THRESH,
    kv_threshold: int = _GLUON_DECODE_KV_THRESH,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (decode_batch_ids, prefill_batch_ids). Decode only if q and kv are both short."""
    cu_q = cu_seqlens_q.detach().cpu().tolist()
    cu_k = cu_seqlens_k.detach().cpu().tolist()
    decode_list = []
    prefill_list = []
    for b in range(len(cu_q) - 1):
        q_len = int(cu_q[b + 1] - cu_q[b])
        k_len = int(cu_k[b + 1] - cu_k[b])
        if q_len <= q_threshold and k_len <= kv_threshold:
            decode_list.append(b)
        else:
            prefill_list.append(b)
    device = cu_seqlens_q.device
    decode_ids = torch.tensor(decode_list, dtype=torch.int32, device=device)
    prefill_ids = torch.tensor(prefill_list, dtype=torch.int32, device=device)
    return decode_ids, prefill_ids


def _max_q_len_for_batches(cu_seqlens_q: torch.Tensor, batch_ids: torch.Tensor) -> int:
    if batch_ids.numel() == 0:
        return 0
    cu = cu_seqlens_q.detach().cpu().tolist()
    return max(int(cu[int(b) + 1] - cu[int(b)]) for b in batch_ids.detach().cpu().tolist())


def _subset_cu_seqlens(cu_seqlens: torch.Tensor, batch_ids: torch.Tensor) -> torch.Tensor:
    """Compact cu_seqlens for a batch subset; entries keep global token offsets."""
    cu = cu_seqlens.detach().cpu().tolist()
    ids = batch_ids.detach().cpu().tolist()
    sub = [int(cu[int(b)]) for b in ids] + [int(cu[int(ids[-1]) + 1])]
    return torch.tensor(sub, dtype=torch.int32, device=cu_seqlens.device)


def _get_num_sm(device: torch.device) -> int:
    props = torch.cuda.get_device_properties(device)
    return int(getattr(props, "multi_processor_count", 132))


def _should_use_static_persistent(total_tiles: int, num_sm: int) -> bool:
    if not _GLUON_STATIC_PERSISTENT_ENABLED:
        return False
    return total_tiles > num_sm * _GLUON_STATIC_PERSISTENT_TILE_RATIO


def _static_persistent_tiles_per_cta(total_tiles: int, num_sm: int) -> int:
    return _cdiv(total_tiles, num_sm)


# [splitkv-disabled] def _get_num_splits_gluon(
# [splitkv-disabled]     batch: int,
# [splitkv-disabled]     num_heads_k: int,
# [splitkv-disabled]     max_seqlen_q: int,
# [splitkv-disabled]     max_seqlen_k: int,
# [splitkv-disabled]     block_m: int,
# [splitkv-disabled]     block_n: int,
# [splitkv-disabled]     pack_gqa: bool,
# [splitkv-disabled]     h_hk_ratio: int,
# [splitkv-disabled]     num_sm: int,
# [splitkv-disabled]     is_causal: bool,
# [splitkv-disabled]     d: int,
# [splitkv-disabled]     elem_bytes: int = 2,
# [splitkv-disabled] ) -> int:
# [splitkv-disabled]     """Mirror vLLM get_num_splits heuristic for varlen Gluon."""
# [splitkv-disabled]     seqlen_q_packgqa = max_seqlen_q * (h_hk_ratio if pack_gqa else 1)
# [splitkv-disabled]     num_n_blocks = _cdiv(max_seqlen_k, block_n)
# [splitkv-disabled]     num_m_blocks = _cdiv(seqlen_q_packgqa, block_m)
# [splitkv-disabled]     total_mblocks = batch * num_heads_k * num_m_blocks
# [splitkv-disabled]     size_one_kv_head = max_seqlen_k * d * 2 * elem_bytes
# [splitkv-disabled]     return _vllm_num_splits_heuristic(
# [splitkv-disabled]         total_mblocks, num_sm, num_n_blocks, num_m_blocks,
# [splitkv-disabled]         size_one_kv_head, is_causal, 128,
# [splitkv-disabled]     )


def _round_up_batch(batch: int, align: int = 4) -> int:
    return _cdiv(batch, align) * align


def _max_kvblocks_in_l2(
    block_n: int,
    d: int,
    elem_bytes: int,
    qhead_per_khead: int,
) -> int:
    """Mirror hopper/flash_prepare_scheduler.cu L2 capacity estimate."""
    if qhead_per_khead == 1:
        divisor = 1
    elif qhead_per_khead <= 2:
        divisor = 2
    elif qhead_per_khead <= 4:
        divisor = 4
    elif qhead_per_khead <= 8:
        divisor = 8
    else:
        divisor = 16
    size_l2 = (32 * 1024 * 1024) // divisor
    size_one_kvblock = block_n * d * 2 * elem_bytes
    return max(size_l2 // size_one_kvblock, 1)


# =============================================================================
# §2  Device utilities
# =============================================================================

class _BarrierCounter:
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


# aggregate() captures cls.__annotations__ at call time; __future__.annotations
# would leave strings if we used @aggregate on the class directly.
_BarrierCounter.__annotations__ = {
    "index": gl.tensor,
    "phase": gl.tensor,
    "num_barriers": gl.constexpr,
}
BarrierCounter = aggregate(_BarrierCounter)


class _Channel:
    @gluon.constexpr_function
    def __init__(self, k_smem, v_smem, k_ready_bars, v_ready_bars, k_empty_bars, v_empty_bars, num_stages):
        self.k_smem = k_smem
        self.v_smem = v_smem
        self.k_ready_bars = k_ready_bars
        self.v_ready_bars = v_ready_bars
        self.k_empty_bars = k_empty_bars
        self.v_empty_bars = v_empty_bars
        self.num_stages = gl.constexpr(num_stages)

    @gluon.jit
    def alloc(
        BLOCK_M: gl.constexpr,
        BLOCK_N: gl.constexpr,
        d: gl.constexpr,
        dtype: gl.constexpr,
        kv_layout: gl.constexpr,
        num_stages: gl.constexpr,
    ):
        k_smem = gl.allocate_shared_memory(dtype, [num_stages, BLOCK_N, d], kv_layout)
        v_smem = gl.allocate_shared_memory(dtype, [num_stages, BLOCK_N, d], kv_layout)
        k_ready_bars = gl.allocate_shared_memory(
            gl.int64, [num_stages, 1], mbarrier.MBarrierLayout()
        )
        v_ready_bars = gl.allocate_shared_memory(
            gl.int64, [num_stages, 1], mbarrier.MBarrierLayout()
        )
        k_empty_bars = gl.allocate_shared_memory(
            gl.int64, [num_stages, 1], mbarrier.MBarrierLayout()
        )
        v_empty_bars = gl.allocate_shared_memory(
            gl.int64, [num_stages, 1], mbarrier.MBarrierLayout()
        )
        for i in gl.static_range(num_stages):
            mbarrier.init(k_ready_bars.index(i), count=1)
            mbarrier.init(v_ready_bars.index(i), count=1)
            mbarrier.init(k_empty_bars.index(i), count=1)
            mbarrier.init(v_empty_bars.index(i), count=1)
            mbarrier.arrive(k_empty_bars.index(i), count=1)
            mbarrier.arrive(v_empty_bars.index(i), count=1)
        return Channel(k_smem, v_smem, k_ready_bars, v_ready_bars, k_empty_bars, v_empty_bars, num_stages)

    @gluon.jit
    def reinit(self):
        """Reset pipeline barriers between persistent tiles (same smem, fresh phases)."""
        self.reset_phases()

    @gluon.jit
    def reset_phases(self):
        """Flip pipeline to empty-ready without reallocating smem (persistent tile advance)."""
        for i in gl.static_range(self.num_stages):
            mbarrier.init(self.k_ready_bars.index(i), count=1)
            mbarrier.init(self.v_ready_bars.index(i), count=1)
            mbarrier.init(self.k_empty_bars.index(i), count=1)
            mbarrier.init(self.v_empty_bars.index(i), count=1)
            mbarrier.arrive(self.k_empty_bars.index(i), count=1)
            mbarrier.arrive(self.v_empty_bars.index(i), count=1)

    @gluon.jit
    def release(self):
        self.k_smem._keep_alive()
        self.v_smem._keep_alive()
        for i in gl.static_range(self.num_stages):
            mbarrier.invalidate(self.k_ready_bars.index(i))
            mbarrier.invalidate(self.v_ready_bars.index(i))
            mbarrier.invalidate(self.k_empty_bars.index(i))
            mbarrier.invalidate(self.v_empty_bars.index(i))


_Channel.__annotations__ = {
    "k_smem": gl.shared_memory_descriptor,
    "v_smem": gl.shared_memory_descriptor,
    "k_ready_bars": gl.shared_memory_descriptor,
    "v_ready_bars": gl.shared_memory_descriptor,
    "k_empty_bars": gl.shared_memory_descriptor,
    "v_empty_bars": gl.shared_memory_descriptor,
    "num_stages": gl.constexpr,
}
Channel = aggregate(_Channel)


@gluon.jit
def _read_seqlen_info(cu_seqlens_q_ptr, cu_seqlens_k_ptr, seqused_k_ptr, bid, IS_PAGED: gl.constexpr):
    """Return (q_bos, q_len, k_bos, k_len) for one batch index."""
    q_bos = gl.load(cu_seqlens_q_ptr + bid).to(gl.int32)
    q_eos = gl.load(cu_seqlens_q_ptr + bid + 1).to(gl.int32)
    q_len = q_eos - q_bos
    if IS_PAGED:
        k_bos = gl.to_tensor(0)
        k_len = gl.load(seqused_k_ptr + bid).to(gl.int32)
    else:
        k_bos = gl.load(cu_seqlens_k_ptr + bid).to(gl.int32)
        k_eos = gl.load(cu_seqlens_k_ptr + bid + 1).to(gl.int32)
        k_len = k_eos - k_bos
    return q_bos, q_len, k_bos, k_len


@gluon.jit
def _online_softmax_step(
    acc,
    rowmax,
    rowsum,
    qk,
    scale_log2e,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    qk_layout: gl.constexpr,
):
    row_layout: gl.constexpr = gl.SliceLayout(1, qk_layout)
    new_max = gl.max(qk, axis=1)
    new_max = gl.maximum(rowmax, new_max)
    safe_max = gl.where(new_max == float("-inf"), gl.zeros_like(new_max), new_max)
    alpha = gl.exp2((rowmax - safe_max) * scale_log2e)
    acc_row_layout: gl.constexpr = gl.SliceLayout(1, acc.type.layout)
    alpha_for_acc = gl.convert_layout(alpha, acc_row_layout)
    acc = acc * alpha_for_acc[:, None]
    rowsum = rowsum * alpha
    p = gl.exp2((qk - safe_max[:, None]) * scale_log2e)
    rowsum = rowsum + gl.sum(p, axis=1)
    rowmax = new_max
    return acc, rowmax, rowsum, p


@gluon.jit
def _mask_qk_tile(
    qk,
    start_n,
    k_len,
    q_len,
    m_block,
    is_causal,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    qk_blocked_layout: gl.constexpr,
    PACK_GQA: gl.constexpr,
    h_hk_ratio: gl.constexpr,
):
    col_idx = start_n + gl.arange(0, BLOCK_N, gl.SliceLayout(0, qk_blocked_layout))
    row_idx = gl.arange(0, BLOCK_M, gl.SliceLayout(1, qk_blocked_layout))
    qk = gl.where(col_idx[None, :] < k_len, qk, float("-inf"))
    if is_causal:
        qk = _apply_causal_mask(
            qk, col_idx, row_idx, q_len, k_len, m_block, BLOCK_M,
            PACK_GQA, h_hk_ratio,
        )
    return qk


@gluon.jit
def _packed_row_to_token(packed_row, h_hk_ratio):
    return packed_row // h_hk_ratio


@gluon.jit
def _apply_causal_mask(
    qk,
    col_idx,
    row_idx,
    q_len,
    k_len,
    m_block,
    BLOCK_M: gl.constexpr,
    PACK_GQA: gl.constexpr = False,
    h_hk_ratio: gl.constexpr = 1,
):
    if PACK_GQA:
        packed_row = m_block * BLOCK_M + row_idx
        row_limit = _packed_row_to_token(packed_row, h_hk_ratio) + (k_len - q_len)
    else:
        row_limit = m_block * BLOCK_M + row_idx + (k_len - q_len)
    causal_mask = col_idx[None, :] <= row_limit[:, None]
    return gl.where(causal_mask, qk, float("-inf"))


@gluon.jit
def _sync_load_kv_tile(desc_k, desc_v, start_n, bar_kv, k_smem, v_smem, nbytes):
    mbarrier.init(bar_kv, count=1)
    mbarrier.expect(bar_kv, nbytes)
    tma.async_copy_global_to_shared(desc_k, [start_n, 0], bar_kv, k_smem)
    tma.async_copy_global_to_shared(desc_v, [start_n, 0], bar_kv, v_smem)
    mbarrier.wait(bar_kv, phase=0)
    mbarrier.invalidate(bar_kv)


@gluon.jit
def _epilogue_store_o_lse(
    acc,
    rowmax,
    rowsum,
    o_smem_out,
    desc_o,
    softmax_lse_ptr,
    m_block,
    q_bos,
    q_len,
    hid,
    total_q,
    scale_softmax_log2,
    BLOCK_M: gl.constexpr,
    pv_layout: gl.constexpr,
    qk_layout: gl.constexpr,
    dtype: gl.constexpr,
):
    """Normalize acc, TMA-store O, write LSE. Phase 2: wire into compute_partition."""
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
def _cpasync_load_kv_tile(
    k_smem,
    v_smem,
    k_base,
    v_base,
    k_row_stride,
    tile_rows,
    d: gl.constexpr,
    BLOCK_N: gl.constexpr,
):
    """Load one [BLOCK_N, d] K/V tile from contiguous memory via cp.async."""
    d_pad: gl.constexpr = _nvmma_d(d)
    dtype: gl.constexpr = k_base.dtype.element_ty
    layout: gl.constexpr = _cpasync_blocked_layout(4, dtype.primitive_bitwidth)
    row_offs = gl.arange(0, BLOCK_N, gl.SliceLayout(1, layout))
    col_offs = gl.arange(0, d_pad, gl.SliceLayout(0, layout))
    mask = (row_offs[:, None] < tile_rows) & (col_offs[None, :] < d)
    k_ptrs = k_base + row_offs[:, None] * k_row_stride + col_offs[None, :]
    v_ptrs = v_base + row_offs[:, None] * k_row_stride + col_offs[None, :]
    cp.async_copy_global_to_shared(k_smem, k_ptrs, mask=mask)
    cp.async_copy_global_to_shared(v_smem, v_ptrs, mask=mask)
    cp.commit_group()
    cp.wait_group(0)
    fence_async_shared()


@gluon.jit
def _cpasync_load_paged_kv_tile(
    k_smem,
    v_smem,
    k_pool_ptr,
    v_pool_ptr,
    page_table_ptr,
    pt_batch_stride,
    bid,
    kv_hid,
    tok_start,
    k_len,
    k_page_stride,
    k_head_stride,
    k_row_stride,
    block_size,
    d: gl.constexpr,
    BLOCK_N: gl.constexpr,
):
    """
    Load one [BLOCK_N, d] paged K/V tile via cp.async.

    Each row uses global token index -> (logical_page, page_offset) -> physical page,
    mirroring vLLM PagedKVManager::load_page_table row indexing.
    """
    d_pad: gl.constexpr = _nvmma_d(d)
    dtype: gl.constexpr = k_pool_ptr.dtype.element_ty
    layout: gl.constexpr = _cpasync_blocked_layout(4, dtype.primitive_bitwidth)
    row_offs = gl.arange(0, BLOCK_N, gl.SliceLayout(1, layout))
    col_offs = gl.arange(0, d_pad, gl.SliceLayout(0, layout))
    row_idx = tok_start + row_offs
    row_mask = row_idx < k_len
    logical_page = row_idx // block_size
    page_offset = row_idx % block_size
    pt_ptrs = page_table_ptr + bid * pt_batch_stride + logical_page
    phys_page = gl.load(pt_ptrs, mask=row_mask, other=0).to(gl.int32)
    k_bases = (
        k_pool_ptr + phys_page * k_page_stride + kv_hid * k_head_stride
        + page_offset * k_row_stride
    )
    v_bases = (
        v_pool_ptr + phys_page * k_page_stride + kv_hid * k_head_stride
        + page_offset * k_row_stride
    )
    elem_mask = row_mask[:, None] & (col_offs[None, :] < d)
    k_ptrs = k_bases[:, None] + col_offs[None, :]
    v_ptrs = v_bases[:, None] + col_offs[None, :]
    cp.async_copy_global_to_shared(k_smem, k_ptrs, mask=elem_mask)
    cp.async_copy_global_to_shared(v_smem, v_ptrs, mask=elem_mask)
    cp.commit_group()
    cp.wait_group(0)
    fence_async_shared()


@gluon.jit
def _cpasync_load_paged_k_tile(
    k_smem,
    k_pool_ptr,
    page_table_ptr,
    pt_batch_stride,
    bid,
    kv_hid,
    tok_start,
    k_len,
    k_page_stride,
    k_head_stride,
    k_row_stride,
    block_size,
    d: gl.constexpr,
    BLOCK_N: gl.constexpr,
):
    d_pad: gl.constexpr = _nvmma_d(d)
    dtype: gl.constexpr = k_pool_ptr.dtype.element_ty
    layout: gl.constexpr = _cpasync_blocked_layout(4, dtype.primitive_bitwidth)
    row_offs = gl.arange(0, BLOCK_N, gl.SliceLayout(1, layout))
    col_offs = gl.arange(0, d_pad, gl.SliceLayout(0, layout))
    row_idx = tok_start + row_offs
    row_mask = row_idx < k_len
    logical_page = row_idx // block_size
    page_offset = row_idx % block_size
    pt_ptrs = page_table_ptr + bid * pt_batch_stride + logical_page
    phys_page = gl.load(pt_ptrs, mask=row_mask, other=0).to(gl.int32)
    k_bases = (
        k_pool_ptr + phys_page * k_page_stride + kv_hid * k_head_stride
        + page_offset * k_row_stride
    )
    elem_mask = row_mask[:, None] & (col_offs[None, :] < d)
    k_ptrs = k_bases[:, None] + col_offs[None, :]
    cp.async_copy_global_to_shared(k_smem, k_ptrs, mask=elem_mask)
    cp.commit_group()
    cp.wait_group(0)
    fence_async_shared()


@gluon.jit
def _cpasync_load_paged_v_tile(
    v_smem,
    v_pool_ptr,
    page_table_ptr,
    pt_batch_stride,
    bid,
    kv_hid,
    tok_start,
    k_len,
    k_page_stride,
    k_head_stride,
    k_row_stride,
    block_size,
    d: gl.constexpr,
    BLOCK_N: gl.constexpr,
):
    d_pad: gl.constexpr = _nvmma_d(d)
    dtype: gl.constexpr = v_pool_ptr.dtype.element_ty
    layout: gl.constexpr = _cpasync_blocked_layout(4, dtype.primitive_bitwidth)
    row_offs = gl.arange(0, BLOCK_N, gl.SliceLayout(1, layout))
    col_offs = gl.arange(0, d_pad, gl.SliceLayout(0, layout))
    row_idx = tok_start + row_offs
    row_mask = row_idx < k_len
    logical_page = row_idx // block_size
    page_offset = row_idx % block_size
    pt_ptrs = page_table_ptr + bid * pt_batch_stride + logical_page
    phys_page = gl.load(pt_ptrs, mask=row_mask, other=0).to(gl.int32)
    v_bases = (
        v_pool_ptr + phys_page * k_page_stride + kv_hid * k_head_stride
        + page_offset * k_row_stride
    )
    elem_mask = row_mask[:, None] & (col_offs[None, :] < d)
    v_ptrs = v_bases[:, None] + col_offs[None, :]
    cp.async_copy_global_to_shared(v_smem, v_ptrs, mask=elem_mask)
    cp.commit_group()
    cp.wait_group(0)
    fence_async_shared()


@gluon.jit
def _mask_qk_tile_paged(
    qk,
    tok_start,
    k_len,
    q_len,
    m_block,
    tile_rows,
    is_causal,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    qk_blocked_layout: gl.constexpr,
    PACK_GQA: gl.constexpr,
    h_hk_ratio: gl.constexpr,
):
    col_idx = gl.arange(0, BLOCK_N, gl.SliceLayout(0, qk_blocked_layout))
    row_idx = gl.arange(0, BLOCK_M, gl.SliceLayout(1, qk_blocked_layout))
    qk = gl.where(col_idx[None, :] < tile_rows, qk, float("-inf"))
    if is_causal:
        col_global = tok_start + col_idx
        if PACK_GQA:
            packed_row = m_block * BLOCK_M + row_idx
            row_limit = _packed_row_to_token(packed_row, h_hk_ratio) + (k_len - q_len)
        else:
            row_limit = m_block * BLOCK_M + row_idx + (k_len - q_len)
        qk = gl.where(col_global[None, :] <= row_limit[:, None], qk, float("-inf"))
    return qk


@gluon.jit
def _paged_kv_tma_page_offset(n_block, block_size, BLOCK_N: gl.constexpr):
    """Map n_block -> (logical_page_slot, row offset within physical page)."""
    tiles_per_page = block_size // BLOCK_N
    logical_page = n_block // tiles_per_page
    in_page_tile = n_block % tiles_per_page
    row_offset = in_page_tile * BLOCK_N
    return logical_page, row_offset


@gluon.jit
def _tma_load_paged_k_tile(
    k_slot,
    k_ready,
    k_ptr,
    page_table_ptr,
    bid,
    kv_hid,
    n_block,
    pt_batch_stride,
    k_page_stride,
    k_head_stride_page,
    k_row_stride,
    block_size,
    d: gl.constexpr,
    BLOCK_N: gl.constexpr,
    kv_layout: gl.constexpr,
    kv_nbytes: gl.constexpr,
):
    logical_page, row_offset = _paged_kv_tma_page_offset(n_block, block_size, BLOCK_N)
    phys_page = gl.load(
        page_table_ptr + bid * pt_batch_stride + logical_page,
    ).to(gl.int32)
    k_page = (
        k_ptr + phys_page * k_page_stride + kv_hid * k_head_stride_page
        + row_offset * k_row_stride
    )
    rows_in_page = block_size - row_offset
    mbarrier.expect(k_ready, kv_nbytes)
    desc_k = tma.make_tensor_descriptor(
        k_page, shape=[rows_in_page, d], strides=[k_row_stride, 1],
        block_shape=[BLOCK_N, _nvmma_d(d)], layout=kv_layout,
    )
    tma.async_copy_global_to_shared(desc_k, [0, 0], k_ready, k_slot)


@gluon.jit
def _tma_load_paged_v_tile(
    v_slot,
    v_ready,
    v_ptr,
    page_table_ptr,
    bid,
    kv_hid,
    n_block,
    pt_batch_stride,
    k_page_stride,
    k_head_stride_page,
    k_row_stride,
    block_size,
    d: gl.constexpr,
    BLOCK_N: gl.constexpr,
    kv_layout: gl.constexpr,
    kv_nbytes: gl.constexpr,
):
    logical_page, row_offset = _paged_kv_tma_page_offset(n_block, block_size, BLOCK_N)
    phys_page = gl.load(
        page_table_ptr + bid * pt_batch_stride + logical_page,
    ).to(gl.int32)
    v_page = (
        v_ptr + phys_page * k_page_stride + kv_hid * k_head_stride_page
        + row_offset * k_row_stride
    )
    rows_in_page = block_size - row_offset
    mbarrier.expect(v_ready, kv_nbytes)
    desc_v = tma.make_tensor_descriptor(
        v_page, shape=[rows_in_page, d], strides=[k_row_stride, 1],
        block_shape=[BLOCK_N, _nvmma_d(d)], layout=kv_layout,
    )
    tma.async_copy_global_to_shared(desc_v, [0, 0], v_ready, v_slot)


# [splitkv-disabled] @gluon.jit
# [splitkv-disabled] def _split_kv_n_bounds(n_block_max, split_idx, num_splits):
# [splitkv-disabled]     """Runtime num_splits (per-batch dynamic or constexpr upper bound)."""
# [splitkv-disabled]     blocks_per_split = gl.cdiv(n_block_max, num_splits)
# [splitkv-disabled]     n_start = split_idx * blocks_per_split
# [splitkv-disabled]     n_end = gl.minimum(n_start + blocks_per_split, n_block_max)
# [splitkv-disabled]     return n_start, n_end
#
#
# [splitkv-disabled] @gluon.jit
# [splitkv-disabled] def _decode_split_grid(hid_packed, NUM_SPLITS: gl.constexpr):
# [splitkv-disabled]     """Map grid z (head * split) -> (head_id, split_idx)."""
# [splitkv-disabled]     if NUM_SPLITS > 1:
# [splitkv-disabled]         hid_actual = hid_packed // NUM_SPLITS
# [splitkv-disabled]         split_idx = hid_packed - hid_actual * NUM_SPLITS
# [splitkv-disabled]         return hid_actual, split_idx
# [splitkv-disabled]     return hid_packed, gl.to_tensor(0)
#
#
# [splitkv-disabled] @gluon.jit
# [splitkv-disabled] def _load_num_splits_act(num_splits_dynamic_ptr, bid, NUM_SPLITS: gl.constexpr):
# [splitkv-disabled]     if NUM_SPLITS > 1:
# [splitkv-disabled]         return gl.load(num_splits_dynamic_ptr + bid).to(gl.int32)
# [splitkv-disabled]     return gl.to_tensor(1)

@gluon.jit
def _n_block_max(
    k_len,
    q_len,
    m_block,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    is_causal,
    PACK_GQA: gl.constexpr = False,
    h_hk_ratio: gl.constexpr = 1,
):
    n_max = gl.cdiv(k_len, BLOCK_N)
    if is_causal:
        if PACK_GQA:
            m_idx_max_packed = (m_block + 1) * BLOCK_M
            m_idx_max_token = (m_idx_max_packed - 1) // h_hk_ratio + 1
            n_token_max = m_idx_max_token + k_len - q_len
        else:
            n_token_max = (m_block + 1) * BLOCK_M + k_len - q_len
        causal_lim = gl.cdiv(n_token_max, BLOCK_N)
        n_max = gl.minimum(n_max, causal_lim)
    return n_max


@gluon.jit
def _load_q_packed_cpasync(
    q_smem,
    q_ptr,
    q_bos,
    q_len,
    kv_hid,
    h_hk_ratio,
    q_row_stride,
    q_head_stride,
    m_block,
    BLOCK_M: gl.constexpr,
    d: gl.constexpr,
    LAUNCH_NUM_WARPS: gl.constexpr,
):
    """PackGQA Q load: packed M rows → cp.async gather (mirrors pack_gqa.h::load_Q)."""
    d_pad: gl.constexpr = _nvmma_d(d)
    dtype: gl.constexpr = q_ptr.dtype.element_ty
    layout: gl.constexpr = _cpasync_blocked_layout(LAUNCH_NUM_WARPS, dtype.primitive_bitwidth)
    row_offs = gl.arange(0, BLOCK_M, gl.SliceLayout(1, layout))
    col_offs = gl.arange(0, d_pad, gl.SliceLayout(0, layout))
    packed_idx = m_block * BLOCK_M + row_offs
    token_idx = packed_idx // h_hk_ratio
    h_idx = packed_idx % h_hk_ratio
    q_head = kv_hid * h_hk_ratio + h_idx
    q_bases = q_ptr + (q_bos + token_idx) * q_row_stride + q_head * q_head_stride
    ptrs = q_bases[:, None] + col_offs[None, :]
    mask = (packed_idx < q_len * h_hk_ratio)[:, None] & (col_offs[None, :] < d)
    cp.async_copy_global_to_shared(q_smem, ptrs, mask=mask)
    cp.commit_group()
    cp.wait_group(0)
    fence_async_shared()


@gluon.jit
def _epilogue_store_o_lse_packed(
    acc,
    rowmax,
    rowsum,
    o_smem_out,
    o_ptr,
    softmax_lse_ptr,
    m_block,
    q_bos,
    q_len,
    kv_hid,
    h_hk_ratio,
    total_q,
    o_row_stride,
    o_head_stride,
    scale_softmax_log2,
    BLOCK_M: gl.constexpr,
    d: gl.constexpr,
    pv_layout: gl.constexpr,
    qk_layout: gl.constexpr,
    dtype: gl.constexpr,
):
    """PackGQA epilogue: scatter O/LSE via divmod on packed rows."""
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

    row_std_layout: gl.constexpr = gl.SliceLayout(1, qk_layout)
    lse_val = gl.where(
        (rowsum == 0) | (rowsum != rowsum),
        gl.full([BLOCK_M], float("inf"), dtype=gl.float32, layout=row_std_layout),
        rowmax / scale_softmax_log2 + gl.log(rowsum) / scale_softmax_log2,
    )

    d_pad: gl.constexpr = _nvmma_d(d)
    o_blocked = o_smem_out.load(qk_layout)
    row_offs = gl.arange(0, BLOCK_M, gl.SliceLayout(1, qk_layout))
    col_offs = gl.arange(0, d_pad, gl.SliceLayout(0, qk_layout))
    packed_idx = m_block * BLOCK_M + row_offs
    token_idx = packed_idx // h_hk_ratio
    h_idx = packed_idx % h_hk_ratio
    q_head = kv_hid * h_hk_ratio + h_idx
    row_mask = packed_idx < q_len * h_hk_ratio

    o_bases = o_ptr + (q_bos + token_idx) * o_row_stride + q_head * o_head_stride
    o_ptrs = o_bases[:, None] + col_offs[None, :]
    o_mask = row_mask[:, None] & (col_offs[None, :] < d)
    gl.store(o_ptrs, o_blocked, mask=o_mask)

    lse_ptrs = softmax_lse_ptr + q_head * total_q + q_bos + token_idx
    gl.store(lse_ptrs, lse_val, mask=row_mask)


# [splitkv-disabled] Split-KV partial epilogues (_epilogue_store_o_lse_split,
# [splitkv-disabled] _epilogue_store_o_lse_packed_split) removed — num_splits fixed at 1.


# =============================================================================
# §3  Warp-spec partitions
# =============================================================================

@gluon.jit
def load_partition(
    channel,
    k_ptr,
    v_ptr,
    desc_k,
    desc_v,
    cu_seqlens_q_ptr,
    cu_seqlens_k_ptr,
    q_row_stride,
    k_row_stride,
    v_row_stride,
    q_head_stride,
    k_head_stride,
    v_head_stride,
    h,
    hk,
    h_hk_ratio,
    d,
    is_causal,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    PACK_GQA: gl.constexpr,
    NUM_SPLITS: gl.constexpr,
    m_block,
    bid,
    split_idx,
    num_splits_act,
    num_warps,
    tile_count_semaphore_ptr,
    next_tile_gmem_ptr,
    kid,
    num_sm,
    DO_PREFETCH: gl.constexpr,
):
    """Producer: pipelined TMA loads (K then V per block)."""
    _, q_len, _, k_len = _read_seqlen_info(
        cu_seqlens_q_ptr, cu_seqlens_k_ptr, cu_seqlens_q_ptr, bid, IS_PAGED=False,
    )
    dtype: gl.constexpr = k_ptr.dtype.element_ty
    kv_nbytes: gl.constexpr = _tile_nbytes(BLOCK_N, _nvmma_d(d), dtype.primitive_bitwidth)

    k_counter = BarrierCounter(gl.to_tensor(0), gl.to_tensor(0), channel.num_stages)
    v_counter = BarrierCounter(gl.to_tensor(0), gl.to_tensor(0), channel.num_stages)
    n_block_max = _n_block_max(
        k_len, q_len, m_block, BLOCK_M, BLOCK_N, is_causal, PACK_GQA, h_hk_ratio,
    )
    n_start = 0
    n_end = n_block_max
    # [splitkv-disabled] n_start, n_end = _split_kv_n_bounds(n_block_max, split_idx, num_splits_act)

    if n_end > n_start:
        for n_block in range(n_end - 1, n_start - 1, -1):
            ki = k_counter.index
            mbarrier.wait(channel.k_empty_bars.index(ki), k_counter.phase)
            mbarrier.expect(channel.k_ready_bars.index(ki), kv_nbytes)
            tma.async_copy_global_to_shared(
                desc_k, [n_block * BLOCK_N, 0],
                channel.k_ready_bars.index(ki), channel.k_smem.index(ki),
            )
            k_counter = k_counter.increment()

            vi = v_counter.index
            mbarrier.wait(channel.v_empty_bars.index(vi), v_counter.phase)
            mbarrier.expect(channel.v_ready_bars.index(vi), kv_nbytes)
            tma.async_copy_global_to_shared(
                desc_v, [n_block * BLOCK_N, 0],
                channel.v_ready_bars.index(vi), channel.v_smem.index(vi),
            )
            v_counter = v_counter.increment()
    if DO_PREFETCH:
        old = tl.atomic_add(tile_count_semaphore_ptr, 1, sem="relaxed", scope="gpu")
        gl.store(next_tile_gmem_ptr + kid, old + num_sm)


@gluon.jit
def compute_partition(
    channel,
    q_smem,
    qk_smem,
    p_smem,
    o_smem_out,
    desc_o,
    o_ptr,
    softmax_lse_ptr,
    oaccum_ptr,
    lseaccum_ptr,
    oaccum_split_stride,
    lseaccum_split_stride,
    q_ptr,
    desc_q,
    bar_q,
    cu_seqlens_q_ptr,
    cu_seqlens_k_ptr,
    q_row_stride,
    k_row_stride,
    v_row_stride,
    o_row_stride,
    q_head_stride,
    k_head_stride,
    v_head_stride,
    o_head_stride,
    total_q,
    h,
    hk,
    h_hk_ratio,
    d,
    scale_softmax_log2,
    is_causal,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    PACK_GQA: gl.constexpr,
    NUM_SPLITS: gl.constexpr,
    m_block,
    bid,
    hid,
    split_idx,
    num_splits_act,
    CONSUMER_WARPS: gl.constexpr,
    LAUNCH_NUM_WARPS: gl.constexpr,
):
    """Consumer: TMA/cp.async Q, WGMMA QK / softmax / PV, epilogue."""
    _, q_len, _, k_len = _read_seqlen_info(
        cu_seqlens_q_ptr, cu_seqlens_k_ptr, cu_seqlens_q_ptr, bid, IS_PAGED=False,
    )
    d_blk: gl.constexpr = _nvmma_d(d)
    dtype: gl.constexpr = q_ptr.dtype.element_ty
    qk_wpc: gl.constexpr = _pick_warps_per_cta(BLOCK_M, BLOCK_N, LAUNCH_NUM_WARPS)
    pv_wpc: gl.constexpr = _pick_warps_per_cta(BLOCK_M, d_blk, LAUNCH_NUM_WARPS)
    qk_instr_n: gl.constexpr = _pick_instr_n(BLOCK_M, BLOCK_N, LAUNCH_NUM_WARPS)
    pv_instr_n: gl.constexpr = _pick_instr_n(BLOCK_M, d_blk, LAUNCH_NUM_WARPS)
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
    qk_blocked_layout: gl.constexpr = _blocked_mn_layout(LAUNCH_NUM_WARPS)
    softmax_row_layout: gl.constexpr = gl.SliceLayout(1, qk_blocked_layout)
    pv_a_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0,
        parent=pv_layout,
        k_width=32 // dtype.primitive_bitwidth,
    )

    q_bos = gl.load(cu_seqlens_q_ptr + bid).to(gl.int32)
    q_eos = gl.load(cu_seqlens_q_ptr + bid + 1).to(gl.int32)
    q_len_local = q_eos - q_bos

    if PACK_GQA:
        _load_q_packed_cpasync(
            q_smem, q_ptr, q_bos, q_len_local, hid, h_hk_ratio,
            q_row_stride, q_head_stride, m_block, BLOCK_M, d, LAUNCH_NUM_WARPS,
        )
    else:
        mbarrier.expect(bar_q, _tile_nbytes(BLOCK_M, d_blk, dtype.primitive_bitwidth))
        tma.async_copy_global_to_shared(desc_q, [m_block * BLOCK_M, 0], bar_q, q_smem)
        mbarrier.wait(bar_q, phase=0)
        mbarrier.invalidate(bar_q)

    acc = gl.zeros((BLOCK_M, d_blk), dtype=gl.float32, layout=pv_layout)
    rowmax = gl.full([BLOCK_M], float("-inf"), dtype=gl.float32, layout=softmax_row_layout)
    rowsum = gl.zeros([BLOCK_M], dtype=gl.float32, layout=softmax_row_layout)
    k_counter = BarrierCounter(gl.to_tensor(0), gl.to_tensor(0), channel.num_stages)
    v_counter = BarrierCounter(gl.to_tensor(0), gl.to_tensor(0), channel.num_stages)
    n_block_max = _n_block_max(
        k_len, q_len, m_block, BLOCK_M, BLOCK_N, is_causal, PACK_GQA, h_hk_ratio,
    )
    n_start = 0
    n_end = n_block_max
    # [splitkv-disabled] n_start, n_end = _split_kv_n_bounds(n_block_max, split_idx, num_splits_act)

    if n_end > n_start:
        for n_block in range(n_end - 1, n_start - 1, -1):
            ki = k_counter.index
            mbarrier.wait(channel.k_ready_bars.index(ki), k_counter.phase)
            kt_smem = channel.k_smem.index(ki).permute((1, 0))
            qk_async = warpgroup_mma(
                q_smem,
                kt_smem,
                gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=qk_layout),
                is_async=True,
                use_acc=False,
            )
            qk = warpgroup_mma_wait(0, (qk_async,))
            qk = _qk_mma_to_blocked(qk, qk_smem, qk_blocked_layout)
            qk = _mask_qk_tile(
                qk, n_block * BLOCK_N, k_len, q_len, m_block, is_causal,
                BLOCK_M, BLOCK_N, qk_blocked_layout, PACK_GQA, h_hk_ratio,
            )
            acc, rowmax, rowsum, p = _online_softmax_step(
                acc, rowmax, rowsum, qk, scale_softmax_log2,
                BLOCK_M, BLOCK_N, qk_blocked_layout,
            )
            p_smem.store(p.to(dtype))
            fence_async_shared()
            mbarrier.arrive(channel.k_empty_bars.index(ki))
            k_counter = k_counter.increment()

            vi = v_counter.index
            mbarrier.wait(channel.v_ready_bars.index(vi), v_counter.phase)
            p_dot = p_smem.load(pv_a_layout)
            v_slot = channel.v_smem.index(vi)
            acc_async = warpgroup_mma(p_dot, v_slot, acc, is_async=True, use_acc=True)
            acc = warpgroup_mma_wait(0, (acc_async,))
            mbarrier.arrive(channel.v_empty_bars.index(vi))
            v_counter = v_counter.increment()

    # [splitkv-disabled] if NUM_SPLITS > 1: ... partial epilogue to oaccum/lseaccum
    if PACK_GQA:
        _epilogue_store_o_lse_packed(
            acc, rowmax, rowsum, o_smem_out, o_ptr, softmax_lse_ptr,
            m_block, q_bos, q_len_local, hid, h_hk_ratio, total_q,
            o_row_stride, o_head_stride, scale_softmax_log2,
            BLOCK_M, d, pv_layout, qk_blocked_layout, dtype,
        )
    else:
        _epilogue_store_o_lse(
            acc, rowmax, rowsum, o_smem_out, desc_o, softmax_lse_ptr,
            m_block, q_bos, q_len_local, hid, total_q, scale_softmax_log2,
            BLOCK_M, pv_layout, qk_blocked_layout, dtype,
        )


@gluon.jit
def load_paged_partition(
    channel,
    k_ptr,
    v_ptr,
    page_table_ptr,
    cu_seqlens_q_ptr,
    seqused_k_ptr,
    q_row_stride,
    k_row_stride,
    v_row_stride,
    q_head_stride,
    k_head_stride,
    v_head_stride,
    pt_batch_stride,
    k_page_stride,
    k_head_stride_page,
    h,
    hk,
    h_hk_ratio,
    d,
    block_size,
    is_causal,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    PACK_GQA: gl.constexpr,
    NUM_SPLITS: gl.constexpr,
    m_block,
    bid,
    hid,
    split_idx,
    num_splits_act,
    num_warps,
    USE_KV_TMA: gl.constexpr,
):
    """Producer: paged KV loads (K then V per block)."""
    _, q_len, _, k_len = _read_seqlen_info(
        cu_seqlens_q_ptr, cu_seqlens_q_ptr, seqused_k_ptr, bid, IS_PAGED=True,
    )
    kv_hid = hid if PACK_GQA else hid // h_hk_ratio
    dtype: gl.constexpr = k_ptr.dtype.element_ty
    kv_layout: gl.constexpr = _nvmma_kv_layout(BLOCK_N, d, dtype)
    kv_nbytes: gl.constexpr = _tile_nbytes(BLOCK_N, _nvmma_d(d), dtype.primitive_bitwidth)

    n_block_max = _n_block_max(
        k_len, q_len, m_block, BLOCK_M, BLOCK_N, is_causal, PACK_GQA, h_hk_ratio,
    )
    n_start = 0
    n_end = n_block_max
    # [splitkv-disabled] n_start, n_end = _split_kv_n_bounds(n_block_max, split_idx, num_splits_act)

    k_counter = BarrierCounter(gl.to_tensor(0), gl.to_tensor(0), channel.num_stages)
    v_counter = BarrierCounter(gl.to_tensor(0), gl.to_tensor(0), channel.num_stages)

    if n_end > n_start:
        for n_block in range(n_end - 1, n_start - 1, -1):
            ki = k_counter.index
            mbarrier.wait(channel.k_empty_bars.index(ki), k_counter.phase)
            if USE_KV_TMA:
                _tma_load_paged_k_tile(
                    channel.k_smem.index(ki), channel.k_ready_bars.index(ki),
                    k_ptr, page_table_ptr, bid, kv_hid, n_block,
                    pt_batch_stride, k_page_stride, k_head_stride_page, k_row_stride,
                    block_size, d, BLOCK_N, kv_layout, kv_nbytes,
                )
            else:
                _cpasync_load_paged_k_tile(
                    channel.k_smem.index(ki), k_ptr, page_table_ptr, pt_batch_stride, bid,
                    kv_hid, n_block * BLOCK_N, k_len, k_page_stride, k_head_stride_page,
                    k_row_stride, block_size, d, BLOCK_N,
                )
                mbarrier.arrive(channel.k_ready_bars.index(ki), count=1)
            k_counter = k_counter.increment()

            vi = v_counter.index
            mbarrier.wait(channel.v_empty_bars.index(vi), v_counter.phase)
            if USE_KV_TMA:
                _tma_load_paged_v_tile(
                    channel.v_smem.index(vi), channel.v_ready_bars.index(vi),
                    v_ptr, page_table_ptr, bid, kv_hid, n_block,
                    pt_batch_stride, k_page_stride, k_head_stride_page, k_row_stride,
                    block_size, d, BLOCK_N, kv_layout, kv_nbytes,
                )
            else:
                _cpasync_load_paged_v_tile(
                    channel.v_smem.index(vi), v_ptr, page_table_ptr, pt_batch_stride, bid,
                    kv_hid, n_block * BLOCK_N, k_len, k_page_stride, k_head_stride_page,
                    k_row_stride, block_size, d, BLOCK_N,
                )
                mbarrier.arrive(channel.v_ready_bars.index(vi), count=1)
            v_counter = v_counter.increment()


@gluon.jit
def compute_paged_partition(
    channel,
    q_smem,
    qk_smem,
    p_smem,
    o_smem_out,
    desc_o,
    o_ptr,
    softmax_lse_ptr,
    oaccum_ptr,
    lseaccum_ptr,
    oaccum_split_stride,
    lseaccum_split_stride,
    q_ptr,
    desc_q,
    bar_q,
    cu_seqlens_q_ptr,
    seqused_k_ptr,
    page_table_ptr,
    q_row_stride,
    q_head_stride,
    o_row_stride,
    o_head_stride,
    k_page_stride,
    k_row_stride,
    k_head_stride_page,
    pt_batch_stride,
    total_q,
    h,
    hk,
    h_hk_ratio,
    d,
    block_size,
    scale_softmax_log2,
    is_causal,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    PACK_GQA: gl.constexpr,
    NUM_SPLITS: gl.constexpr,
    m_block,
    bid,
    hid,
    split_idx,
    num_splits_act,
    CONSUMER_WARPS: gl.constexpr,
    LAUNCH_NUM_WARPS: gl.constexpr,
):
    """Consumer for paged path (same WGMMA loop as non-paged)."""
    _, q_len, _, k_len = _read_seqlen_info(
        cu_seqlens_q_ptr, cu_seqlens_q_ptr, seqused_k_ptr, bid, IS_PAGED=True,
    )
    d_blk: gl.constexpr = _nvmma_d(d)
    dtype: gl.constexpr = q_ptr.dtype.element_ty
    qk_wpc: gl.constexpr = _pick_warps_per_cta(BLOCK_M, BLOCK_N, LAUNCH_NUM_WARPS)
    pv_wpc: gl.constexpr = _pick_warps_per_cta(BLOCK_M, d_blk, LAUNCH_NUM_WARPS)
    qk_instr_n: gl.constexpr = _pick_instr_n(BLOCK_M, BLOCK_N, LAUNCH_NUM_WARPS)
    pv_instr_n: gl.constexpr = _pick_instr_n(BLOCK_M, d_blk, LAUNCH_NUM_WARPS)
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
    qk_blocked_layout: gl.constexpr = _blocked_mn_layout(LAUNCH_NUM_WARPS)
    softmax_row_layout: gl.constexpr = gl.SliceLayout(1, qk_blocked_layout)
    pv_a_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0,
        parent=pv_layout,
        k_width=32 // dtype.primitive_bitwidth,
    )

    q_bos = gl.load(cu_seqlens_q_ptr + bid).to(gl.int32)
    q_eos = gl.load(cu_seqlens_q_ptr + bid + 1).to(gl.int32)
    q_len_local = q_eos - q_bos

    if PACK_GQA:
        _load_q_packed_cpasync(
            q_smem, q_ptr, q_bos, q_len_local, hid, h_hk_ratio,
            q_row_stride, q_head_stride, m_block, BLOCK_M, d, LAUNCH_NUM_WARPS,
        )
    else:
        mbarrier.expect(bar_q, _tile_nbytes(BLOCK_M, d_blk, dtype.primitive_bitwidth))
        tma.async_copy_global_to_shared(desc_q, [m_block * BLOCK_M, 0], bar_q, q_smem)
        mbarrier.wait(bar_q, phase=0)
        mbarrier.invalidate(bar_q)

    acc = gl.zeros((BLOCK_M, d_blk), dtype=gl.float32, layout=pv_layout)
    rowmax = gl.full([BLOCK_M], float("-inf"), dtype=gl.float32, layout=softmax_row_layout)
    rowsum = gl.zeros([BLOCK_M], dtype=gl.float32, layout=softmax_row_layout)
    k_counter = BarrierCounter(gl.to_tensor(0), gl.to_tensor(0), channel.num_stages)
    v_counter = BarrierCounter(gl.to_tensor(0), gl.to_tensor(0), channel.num_stages)
    n_block_max = _n_block_max(
        k_len, q_len, m_block, BLOCK_M, BLOCK_N, is_causal, PACK_GQA, h_hk_ratio,
    )
    n_start = 0
    n_end = n_block_max
    # [splitkv-disabled] n_start, n_end = _split_kv_n_bounds(n_block_max, split_idx, num_splits_act)

    if n_end > n_start:
        for n_block in range(n_end - 1, n_start - 1, -1):
            ki = k_counter.index
            mbarrier.wait(channel.k_ready_bars.index(ki), k_counter.phase)
            kt_smem = channel.k_smem.index(ki).permute((1, 0))
            qk_async = warpgroup_mma(
                q_smem,
                kt_smem,
                gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=qk_layout),
                is_async=True,
                use_acc=False,
            )
            qk = warpgroup_mma_wait(0, (qk_async,))
            qk = _qk_mma_to_blocked(qk, qk_smem, qk_blocked_layout)
            tok_start = n_block * BLOCK_N
            tile_rows = gl.minimum(k_len - tok_start, BLOCK_N)
            qk = _mask_qk_tile_paged(
                qk, tok_start, k_len, q_len, m_block, tile_rows, is_causal,
                BLOCK_M, BLOCK_N, qk_blocked_layout, PACK_GQA, h_hk_ratio,
            )
            acc, rowmax, rowsum, p = _online_softmax_step(
                acc, rowmax, rowsum, qk, scale_softmax_log2,
                BLOCK_M, BLOCK_N, qk_blocked_layout,
            )
            p_smem.store(p.to(dtype))
            fence_async_shared()
            mbarrier.arrive(channel.k_empty_bars.index(ki))
            k_counter = k_counter.increment()

            vi = v_counter.index
            mbarrier.wait(channel.v_ready_bars.index(vi), v_counter.phase)
            p_dot = p_smem.load(pv_a_layout)
            v_slot = channel.v_smem.index(vi)
            acc_async = warpgroup_mma(p_dot, v_slot, acc, is_async=True, use_acc=True)
            acc = warpgroup_mma_wait(0, (acc_async,))
            mbarrier.arrive(channel.v_empty_bars.index(vi))
            v_counter = v_counter.increment()

    # [splitkv-disabled] if NUM_SPLITS > 1: ... partial epilogue to oaccum/lseaccum
    if PACK_GQA:
        _epilogue_store_o_lse_packed(
            acc, rowmax, rowsum, o_smem_out, o_ptr, softmax_lse_ptr,
            m_block, q_bos, q_len_local, hid, h_hk_ratio, total_q,
            o_row_stride, o_head_stride, scale_softmax_log2,
            BLOCK_M, d, pv_layout, qk_blocked_layout, dtype,
        )
    else:
        _epilogue_store_o_lse(
            acc, rowmax, rowsum, o_smem_out, desc_o, softmax_lse_ptr,
            m_block, q_bos, q_len_local, hid, total_q, scale_softmax_log2,
            BLOCK_M, pv_layout, qk_blocked_layout, dtype,
        )


# =============================================================================
# §4  Kernel entry points
# =============================================================================

@libentry()
@gluon.jit
def flash_varlen_fwd_gluon_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    softmax_lse_ptr,
    oaccum_ptr,
    lseaccum_ptr,
    oaccum_split_stride,
    lseaccum_split_stride,
    cu_seqlens_q_ptr,
    cu_seqlens_k_ptr,
    num_splits_dynamic_ptr,
    q_row_stride,
    k_row_stride,
    v_row_stride,
    o_row_stride,
    q_head_stride,
    k_head_stride,
    v_head_stride,
    o_head_stride,
    total_q,
    h: gl.constexpr,
    hk: gl.constexpr,
    h_hk_ratio: gl.constexpr,
    d: gl.constexpr,
    scale_softmax_log2: gl.constexpr,
    is_causal: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    num_warps: gl.constexpr,
    num_stages: gl.constexpr,
    PACK_GQA: gl.constexpr,
    NUM_SPLITS: gl.constexpr,
    PRODUCER_WARPS: gl.constexpr,
    CONSUMER_WARPS: gl.constexpr,
):
    """
    Non-paged varlen prefill kernel.

    Grid: (cdiv(max_seqlen_q * ratio, BLOCK_M), batch, h_k * NUM_SPLITS) when PACK_GQA else
          (cdiv(max_seqlen_q, BLOCK_M), batch, num_heads * NUM_SPLITS)
    """
    d_blk: gl.constexpr = _nvmma_d(d)
    m_block = gl.program_id(0)
    bid = gl.program_id(1)
    hid = gl.program_id(2)
    split_idx = 0
    num_splits_act = 1
    # [splitkv-disabled] hid_packed = gl.program_id(2)
    # [splitkv-disabled] hid, split_idx = _decode_split_grid(hid_packed, NUM_SPLITS)
    # [splitkv-disabled] num_splits_act = _load_num_splits_act(num_splits_dynamic_ptr, bid, NUM_SPLITS)
    # [splitkv-disabled] if split_idx >= num_splits_act:
    # [splitkv-disabled]     return

    q_bos, q_len, k_bos, k_len = _read_seqlen_info(
        cu_seqlens_q_ptr, cu_seqlens_k_ptr, cu_seqlens_q_ptr, bid, IS_PAGED=False
    )
    if PACK_GQA:
        if m_block * BLOCK_M >= q_len * h_hk_ratio:
            return
    else:
        if m_block * BLOCK_M >= q_len:
            return

    dtype: gl.constexpr = q_ptr.dtype.element_ty
    q_layout: gl.constexpr = _nvmma_qo_layout(BLOCK_M, d, dtype)
    kv_layout: gl.constexpr = _nvmma_kv_layout(BLOCK_N, d, dtype)
    o_layout: gl.constexpr = _nvmma_qo_layout(BLOCK_M, d, dtype)

    kv_hid = hid if PACK_GQA else hid // h_hk_ratio
    k_seq = k_ptr + k_bos * k_row_stride + kv_hid * k_head_stride
    v_seq = v_ptr + k_bos * v_row_stride + kv_hid * v_head_stride

    desc_k = tma.make_tensor_descriptor(
        k_seq, shape=[k_len, d], strides=[k_row_stride, 1],
        block_shape=[BLOCK_N, d_blk], layout=kv_layout,
    )
    desc_v = tma.make_tensor_descriptor(
        v_seq, shape=[k_len, d], strides=[v_row_stride, 1],
        block_shape=[BLOCK_N, d_blk], layout=kv_layout,
    )

    q_smem = gl.allocate_shared_memory(dtype, [BLOCK_M, d_blk], q_layout)
    qk_smem_layout: gl.constexpr = _qk_smem_layout(BLOCK_M, BLOCK_N)
    qk_smem = gl.allocate_shared_memory(gl.float32, [BLOCK_M, BLOCK_N], qk_smem_layout)
    p_smem_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_N], dtype)
    p_smem = gl.allocate_shared_memory(dtype, [BLOCK_M, BLOCK_N], p_smem_layout)
    o_smem_out = gl.allocate_shared_memory(dtype, [BLOCK_M, d_blk], o_layout)
    bar_q = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar_q, count=1)

    if PACK_GQA:
        # Descriptors unused (Q/O use cp.async + scatter); minimal placeholders for JIT types.
        desc_q = tma.make_tensor_descriptor(
            q_ptr, shape=[1, d], strides=[q_row_stride, 1],
            block_shape=[BLOCK_M, d_blk], layout=q_layout,
        )
        desc_o = tma.make_tensor_descriptor(
            o_ptr, shape=[1, d], strides=[o_row_stride, 1],
            block_shape=[BLOCK_M, d_blk], layout=o_layout,
        )
    else:
        q_seq = q_ptr + q_bos * q_row_stride + hid * q_head_stride
        o_seq = o_ptr + q_bos * o_row_stride + hid * o_head_stride
        desc_q = tma.make_tensor_descriptor(
            q_seq, shape=[q_len, d], strides=[q_row_stride, 1],
            block_shape=[BLOCK_M, d_blk], layout=q_layout,
        )
        desc_o = tma.make_tensor_descriptor(
            o_seq, shape=[q_len, d], strides=[o_row_stride, 1],
            block_shape=[BLOCK_M, d_blk], layout=o_layout,
        )

    channel = Channel.alloc(BLOCK_M, BLOCK_N, d_blk, dtype, kv_layout, num_stages)

    gl.warp_specialize(
        [
            (
                compute_partition,
                (
                    channel, q_smem, qk_smem, p_smem, o_smem_out, desc_o, o_ptr, softmax_lse_ptr,
                    oaccum_ptr, lseaccum_ptr, oaccum_split_stride, lseaccum_split_stride,
                    q_ptr, desc_q, bar_q, cu_seqlens_q_ptr, cu_seqlens_k_ptr,
                    q_row_stride, k_row_stride, v_row_stride, o_row_stride,
                    q_head_stride, k_head_stride, v_head_stride, o_head_stride,
                    total_q, h, hk, h_hk_ratio, d, scale_softmax_log2, is_causal,
                    BLOCK_M, BLOCK_N, PACK_GQA, NUM_SPLITS,
                    m_block, bid, hid, split_idx, num_splits_act, CONSUMER_WARPS, num_warps,
                ),
            ),
            (
                load_partition,
                (
                    channel, k_ptr, v_ptr, desc_k, desc_v,
                    cu_seqlens_q_ptr, cu_seqlens_k_ptr,
                    q_row_stride, k_row_stride, v_row_stride,
                    q_head_stride, k_head_stride, v_head_stride,
                    h, hk, h_hk_ratio, d, is_causal,
                    BLOCK_M, BLOCK_N, PACK_GQA, NUM_SPLITS,
                    m_block, bid, split_idx, num_splits_act, num_warps,
                    cu_seqlens_q_ptr, cu_seqlens_q_ptr, gl.program_id(0), num_warps, False,
                ),
            ),
        ],
        [PRODUCER_WARPS],
        [24],
    )
    channel.release()


@libentry()
@gluon.jit
def flash_varlen_fwd_gluon_persistent_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    softmax_lse_ptr,
    oaccum_ptr,
    lseaccum_ptr,
    oaccum_split_stride,
    lseaccum_split_stride,
    cu_seqlens_q_ptr,
    cu_seqlens_k_ptr,
    num_splits_dynamic_ptr,
    q_row_stride,
    k_row_stride,
    v_row_stride,
    o_row_stride,
    q_head_stride,
    k_head_stride,
    v_head_stride,
    o_head_stride,
    total_q,
    h: gl.constexpr,
    hk: gl.constexpr,
    h_hk_ratio: gl.constexpr,
    d: gl.constexpr,
    scale_softmax_log2: gl.constexpr,
    is_causal: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    num_warps: gl.constexpr,
    num_stages: gl.constexpr,
    PACK_GQA: gl.constexpr,
    NUM_SPLITS: gl.constexpr,
    PRODUCER_WARPS: gl.constexpr,
    CONSUMER_WARPS: gl.constexpr,
    GRID_M: gl.constexpr,
    BATCH: gl.constexpr,
    GRID_H: gl.constexpr,
    NUM_SM: gl.constexpr,
    TOTAL_TILES: gl.constexpr,
    TILES_PER_CTA: gl.constexpr,
):
    """
    Static persistent non-paged varlen prefill.

    Grid: (NUM_SM,).  CTA kid processes tile_idx = kid + i * NUM_SM for i in TILES_PER_CTA.
    Tile order matches MVP grid (m_block, bid, hid) linearization.
    """
    d_blk: gl.constexpr = _nvmma_d(d)
    kid = gl.program_id(0)
    split_idx = 0
    num_splits_act = 1

    dtype: gl.constexpr = q_ptr.dtype.element_ty
    q_layout: gl.constexpr = _nvmma_qo_layout(BLOCK_M, d, dtype)
    kv_layout: gl.constexpr = _nvmma_kv_layout(BLOCK_N, d, dtype)
    o_layout: gl.constexpr = _nvmma_qo_layout(BLOCK_M, d, dtype)

    q_smem = gl.allocate_shared_memory(dtype, [BLOCK_M, d_blk], q_layout)
    qk_smem_layout: gl.constexpr = _qk_smem_layout(BLOCK_M, BLOCK_N)
    qk_smem = gl.allocate_shared_memory(gl.float32, [BLOCK_M, BLOCK_N], qk_smem_layout)
    p_smem_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_N], dtype)
    p_smem = gl.allocate_shared_memory(dtype, [BLOCK_M, BLOCK_N], p_smem_layout)
    o_smem_out = gl.allocate_shared_memory(dtype, [BLOCK_M, d_blk], o_layout)
    bar_q = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar_q, count=1)
    channel = Channel.alloc(BLOCK_M, BLOCK_N, d_blk, dtype, kv_layout, num_stages)

    for i in range(TILES_PER_CTA):
        tile_idx = kid + i * NUM_SM
        if tile_idx < TOTAL_TILES:
            m_block = tile_idx % GRID_M
            bid = (tile_idx // GRID_M) % BATCH
            hid = tile_idx // (GRID_M * BATCH)

            q_bos, q_len, k_bos, k_len = _read_seqlen_info(
                cu_seqlens_q_ptr, cu_seqlens_k_ptr, cu_seqlens_q_ptr, bid, IS_PAGED=False
            )
            if PACK_GQA:
                valid = m_block * BLOCK_M < q_len * h_hk_ratio
            else:
                valid = m_block * BLOCK_M < q_len

            if valid:
                kv_hid = hid if PACK_GQA else hid // h_hk_ratio
                k_seq = k_ptr + k_bos * k_row_stride + kv_hid * k_head_stride
                v_seq = v_ptr + k_bos * v_row_stride + kv_hid * v_head_stride

                desc_k = tma.make_tensor_descriptor(
                    k_seq, shape=[k_len, d], strides=[k_row_stride, 1],
                    block_shape=[BLOCK_N, d_blk], layout=kv_layout,
                )
                desc_v = tma.make_tensor_descriptor(
                    v_seq, shape=[k_len, d], strides=[v_row_stride, 1],
                    block_shape=[BLOCK_N, d_blk], layout=kv_layout,
                )

                if PACK_GQA:
                    desc_q = tma.make_tensor_descriptor(
                        q_ptr, shape=[1, d], strides=[q_row_stride, 1],
                        block_shape=[BLOCK_M, d_blk], layout=q_layout,
                    )
                    desc_o = tma.make_tensor_descriptor(
                        o_ptr, shape=[1, d], strides=[o_row_stride, 1],
                        block_shape=[BLOCK_M, d_blk], layout=o_layout,
                    )
                else:
                    q_seq = q_ptr + q_bos * q_row_stride + hid * q_head_stride
                    o_seq = o_ptr + q_bos * o_row_stride + hid * o_head_stride
                    desc_q = tma.make_tensor_descriptor(
                        q_seq, shape=[q_len, d], strides=[q_row_stride, 1],
                        block_shape=[BLOCK_M, d_blk], layout=q_layout,
                    )
                    desc_o = tma.make_tensor_descriptor(
                        o_seq, shape=[q_len, d], strides=[o_row_stride, 1],
                        block_shape=[BLOCK_M, d_blk], layout=o_layout,
                    )

                gl.warp_specialize(
                    [
                        (
                            compute_partition,
                            (
                                channel, q_smem, qk_smem, p_smem, o_smem_out, desc_o, o_ptr, softmax_lse_ptr,
                                oaccum_ptr, lseaccum_ptr, oaccum_split_stride, lseaccum_split_stride,
                                q_ptr, desc_q, bar_q, cu_seqlens_q_ptr, cu_seqlens_k_ptr,
                                q_row_stride, k_row_stride, v_row_stride, o_row_stride,
                                q_head_stride, k_head_stride, v_head_stride, o_head_stride,
                                total_q, h, hk, h_hk_ratio, d, scale_softmax_log2, is_causal,
                                BLOCK_M, BLOCK_N, PACK_GQA, NUM_SPLITS,
                                m_block, bid, hid, split_idx, num_splits_act, CONSUMER_WARPS, num_warps,
                            ),
                        ),
                        (
                            load_partition,
                            (
                                channel, k_ptr, v_ptr, desc_k, desc_v,
                                cu_seqlens_q_ptr, cu_seqlens_k_ptr,
                                q_row_stride, k_row_stride, v_row_stride,
                                q_head_stride, k_head_stride, v_head_stride,
                                h, hk, h_hk_ratio, d, is_causal,
                                BLOCK_M, BLOCK_N, PACK_GQA, NUM_SPLITS,
                                m_block, bid, split_idx, num_splits_act, num_warps,
                                cu_seqlens_q_ptr, cu_seqlens_q_ptr, kid, num_warps, False,
                            ),
                        ),
                    ],
                    [PRODUCER_WARPS],
                    [24],
                )
                channel.reinit()
                mbarrier.init(bar_q, count=1)

    channel.release()


# [splitkv-disabled] @libentry()
# [splitkv-disabled] @triton.jit
# [splitkv-disabled] def flash_varlen_gluon_splitkv_combine_kernel(
# [splitkv-disabled]     out_ptr,
# [splitkv-disabled]     lse_ptr,
# [splitkv-disabled]     out_accum_ptr,
# [splitkv-disabled]     lse_accum_ptr,
# [splitkv-disabled]     cu_seqlens_q_ptr,
# [splitkv-disabled]     num_splits_dynamic_ptr,
# [splitkv-disabled]     total_q,
# [splitkv-disabled]     num_batch: tl.constexpr,
# [splitkv-disabled]     num_heads: tl.constexpr,
# [splitkv-disabled]     d: tl.constexpr,
# [splitkv-disabled]     o_row_stride,
# [splitkv-disabled]     o_head_stride,
# [splitkv-disabled]     o_split_stride,
# [splitkv-disabled]     lse_split_stride,
# [splitkv-disabled]     n_splits: tl.constexpr,
# [splitkv-disabled]     MAX_N_SPLITS: tl.constexpr,
# [splitkv-disabled]     BLOCK_M: tl.constexpr,
# [splitkv-disabled]     BLOCK_K: tl.constexpr,
# [splitkv-disabled] ):
# [splitkv-disabled]     """Combine split-KV partials for varlen layout o:[total_q,h,d], lse:[h,total_q]."""
# [splitkv-disabled]     pid_m = tl.program_id(0)
# [splitkv-disabled]     hid = tl.program_id(1)
# [splitkv-disabled]     rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
# [splitkv-disabled]     row_mask = rows < total_q
# [splitkv-disabled]     cols = tl.arange(0, BLOCK_K)
# [splitkv-disabled]     col_mask = cols < d
# [splitkv-disabled]
# [splitkv-disabled]     batch_id = tl.zeros([BLOCK_M], dtype=tl.int32)
# [splitkv-disabled]     for b in tl.static_range(num_batch):
# [splitkv-disabled]         q_start = tl.load(cu_seqlens_q_ptr + b)
# [splitkv-disabled]         q_end = tl.load(cu_seqlens_q_ptr + b + 1)
# [splitkv-disabled]         in_batch = (rows >= q_start) & (rows < q_end)
# [splitkv-disabled]         batch_id = tl.where(in_batch, b, batch_id)
# [splitkv-disabled]     n_splits_row = tl.load(num_splits_dynamic_ptr + batch_id)
# [splitkv-disabled]     split_arange = tl.arange(0, MAX_N_SPLITS)[None, :]
# [splitkv-disabled]     valid_split = split_arange < n_splits_row[:, None]
# [splitkv-disabled]
# [splitkv-disabled]     lse_splits = tl.load(
# [splitkv-disabled]         lse_accum_ptr + hid * total_q + rows[:, None]
# [splitkv-disabled]         + split_arange * lse_split_stride,
# [splitkv-disabled]         mask=row_mask[:, None] & valid_split,
# [splitkv-disabled]         other=float("-inf"),
# [splitkv-disabled]     )
# [splitkv-disabled]     max_lse = tl.max(lse_splits, 1)
# [splitkv-disabled]     zi = tl.exp(lse_splits - max_lse[:, None])
# [splitkv-disabled]     z_sum = tl.sum(zi, 1)
# [splitkv-disabled]     weights = zi / z_sum[:, None]
# [splitkv-disabled]     lse_out = tl.log(z_sum) + max_lse
# [splitkv-disabled]     tl.store(lse_ptr + hid * total_q + rows, lse_out, mask=row_mask)
# [splitkv-disabled]
# [splitkv-disabled]     out_splits = tl.load(
# [splitkv-disabled]         out_accum_ptr + rows[:, None, None] * o_row_stride
# [splitkv-disabled]         + hid * o_head_stride
# [splitkv-disabled]         + cols[None, None, :]
# [splitkv-disabled]         + split_arange[:, :, None] * o_split_stride,
# [splitkv-disabled]         mask=row_mask[:, None, None]
# [splitkv-disabled]         & valid_split[:, :, None]
# [splitkv-disabled]         & col_mask[None, None, :],
# [splitkv-disabled]         other=0.0,
# [splitkv-disabled]     )
# [splitkv-disabled]     o_out = tl.sum(weights[:, :, None] * out_splits, 1)
# [splitkv-disabled]     tl.store(
# [splitkv-disabled]         out_ptr + rows[:, None] * o_row_stride + hid * o_head_stride + cols[None, :],
# [splitkv-disabled]         o_out.to(out_ptr.dtype.element_ty),
# [splitkv-disabled]         mask=row_mask[:, None] & col_mask[None, :],
# [splitkv-disabled]     )


@libentry()
@triton.jit
def flash_varlen_decode_gluon_kernel(
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


@libentry()
@gluon.jit
def flash_paged_fwd_gluon_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    softmax_lse_ptr,
    oaccum_ptr,
    lseaccum_ptr,
    oaccum_split_stride,
    lseaccum_split_stride,
    cu_seqlens_q_ptr,
    seqused_k_ptr,
    page_table_ptr,
    num_splits_dynamic_ptr,
    q_row_stride,
    q_head_stride,
    o_row_stride,
    o_head_stride,
    k_page_stride,
    k_row_stride,
    k_head_stride,
    pt_batch_stride,
    total_q,
    h: gl.constexpr,
    hk: gl.constexpr,
    h_hk_ratio: gl.constexpr,
    d: gl.constexpr,
    block_size: gl.constexpr,
    scale_softmax_log2: gl.constexpr,
    is_causal: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    num_warps: gl.constexpr,
    num_stages: gl.constexpr,
    USE_KV_TMA: gl.constexpr,
    PACK_GQA: gl.constexpr,
    NUM_SPLITS: gl.constexpr,
    PRODUCER_WARPS: gl.constexpr,
    CONSUMER_WARPS: gl.constexpr,
):
    """Paged varlen: seqused_k length, cp.async KV (or TMA if USE_KV_TMA)."""
    d_blk: gl.constexpr = _nvmma_d(d)
    m_block = gl.program_id(0)
    bid = gl.program_id(1)
    hid = gl.program_id(2)
    split_idx = 0
    num_splits_act = 1
    # [splitkv-disabled] hid_packed = gl.program_id(2)
    # [splitkv-disabled] hid, split_idx = _decode_split_grid(hid_packed, NUM_SPLITS)
    # [splitkv-disabled] num_splits_act = _load_num_splits_act(num_splits_dynamic_ptr, bid, NUM_SPLITS)
    # [splitkv-disabled] if split_idx >= num_splits_act:
    # [splitkv-disabled]     return

    q_bos, q_len, _, k_len = _read_seqlen_info(
        cu_seqlens_q_ptr, cu_seqlens_q_ptr, seqused_k_ptr, bid, IS_PAGED=True
    )
    if PACK_GQA:
        if m_block * BLOCK_M >= q_len * h_hk_ratio:
            return
    else:
        if m_block * BLOCK_M >= q_len:
            return

    dtype: gl.constexpr = q_ptr.dtype.element_ty
    q_layout: gl.constexpr = _nvmma_qo_layout(BLOCK_M, d, dtype)
    kv_layout: gl.constexpr = _nvmma_kv_layout(BLOCK_N, d, dtype)
    o_layout: gl.constexpr = _nvmma_qo_layout(BLOCK_M, d, dtype)

    if PACK_GQA:
        desc_q = tma.make_tensor_descriptor(
            q_ptr, shape=[1, d], strides=[q_row_stride, 1],
            block_shape=[BLOCK_M, d_blk], layout=q_layout,
        )
        desc_o = tma.make_tensor_descriptor(
            o_ptr, shape=[1, d], strides=[o_row_stride, 1],
            block_shape=[BLOCK_M, d_blk], layout=o_layout,
        )
    else:
        q_seq = q_ptr + q_bos * q_row_stride + hid * q_head_stride
        o_seq = o_ptr + q_bos * o_row_stride + hid * o_head_stride
        desc_q = tma.make_tensor_descriptor(
            q_seq, shape=[q_len, d], strides=[q_row_stride, 1],
            block_shape=[BLOCK_M, d_blk], layout=q_layout,
        )
        desc_o = tma.make_tensor_descriptor(
            o_seq, shape=[q_len, d], strides=[o_row_stride, 1],
            block_shape=[BLOCK_M, d_blk], layout=o_layout,
        )

    q_smem = gl.allocate_shared_memory(dtype, [BLOCK_M, d_blk], q_layout)
    qk_smem_layout: gl.constexpr = _qk_smem_layout(BLOCK_M, BLOCK_N)
    qk_smem = gl.allocate_shared_memory(gl.float32, [BLOCK_M, BLOCK_N], qk_smem_layout)
    p_smem_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_N], dtype)
    p_smem = gl.allocate_shared_memory(dtype, [BLOCK_M, BLOCK_N], p_smem_layout)
    o_smem_out = gl.allocate_shared_memory(dtype, [BLOCK_M, d_blk], o_layout)
    bar_q = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar_q, count=1)

    channel = Channel.alloc(BLOCK_M, BLOCK_N, d_blk, dtype, kv_layout, num_stages)

    gl.warp_specialize(
        [
            (
                compute_paged_partition,
                (
                    channel, q_smem, qk_smem, p_smem, o_smem_out, desc_o, o_ptr, softmax_lse_ptr,
                    oaccum_ptr, lseaccum_ptr, oaccum_split_stride, lseaccum_split_stride,
                    q_ptr, desc_q, bar_q, cu_seqlens_q_ptr, seqused_k_ptr, page_table_ptr,
                    q_row_stride, q_head_stride, o_row_stride, o_head_stride,
                    k_page_stride, k_row_stride, k_head_stride, pt_batch_stride,
                    total_q, h, hk, h_hk_ratio, d, block_size, scale_softmax_log2, is_causal,
                    BLOCK_M, BLOCK_N, PACK_GQA, NUM_SPLITS,
                    m_block, bid, hid, split_idx, num_splits_act, CONSUMER_WARPS, num_warps,
                ),
            ),
            (
                load_paged_partition,
                (
                    channel, k_ptr, v_ptr, page_table_ptr,
                    cu_seqlens_q_ptr, seqused_k_ptr,
                    q_row_stride, k_row_stride, k_row_stride,
                    q_head_stride, k_head_stride, k_head_stride,
                    pt_batch_stride, k_page_stride, k_head_stride,
                    h, hk, h_hk_ratio, d, block_size, is_causal,
                    BLOCK_M, BLOCK_N, PACK_GQA, NUM_SPLITS,
                    m_block, bid, hid, split_idx, num_splits_act, num_warps, USE_KV_TMA,
                ),
            ),
        ],
        [PRODUCER_WARPS],
        [24],
    )
    channel.release()


@libentry()
@gluon.jit
def flash_paged_fwd_gluon_decode_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    softmax_lse_ptr,
    cu_seqlens_q_ptr,
    seqused_k_ptr,
    page_table_ptr,
    q_row_stride,
    q_head_stride,
    o_row_stride,
    o_head_stride,
    k_page_stride,
    k_row_stride,
    k_head_stride,
    pt_batch_stride,
    total_q,
    h: gl.constexpr,
    hk: gl.constexpr,
    h_hk_ratio: gl.constexpr,
    d: gl.constexpr,
    block_size: gl.constexpr,
    scale_softmax_log2: gl.constexpr,
    is_causal: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    num_warps: gl.constexpr,
    USE_KV_TMA: gl.constexpr,
):
    """Paged decode kernel. TODO Phase 3."""
    m_block = gl.program_id(0)
    bid = gl.program_id(1)
    q_bos, q_len, _, _ = _read_seqlen_info(
        cu_seqlens_q_ptr, cu_seqlens_q_ptr, seqused_k_ptr, bid, IS_PAGED=True
    )
    if m_block * BLOCK_M >= q_len:
        return
    return


# libtriton_jit C++ POC: raw JIT handles (public names are LibEntry-wrapped).
flash_varlen_decode_gluon_kernel_jit = flash_varlen_decode_gluon_kernel.fn
flash_varlen_fwd_gluon_kernel_jit = flash_varlen_fwd_gluon_kernel.fn


# =============================================================================
# §5  Public launcher
# =============================================================================

# [splitkv-disabled] def _alloc_splitkv_buffers(
# [splitkv-disabled]     num_splits: int,
# [splitkv-disabled]     total_q: int,
# [splitkv-disabled]     num_heads: int,
# [splitkv-disabled]     d: int,
# [splitkv-disabled]     device: torch.device,
# [splitkv-disabled]     out: torch.Tensor,
# [splitkv-disabled]     lse: torch.Tensor,
# [splitkv-disabled] ) -> tuple[torch.Tensor, torch.Tensor, int, int]:
# [splitkv-disabled]     """Allocate split-KV partials or reuse out/lse when num_splits == 1."""
# [splitkv-disabled]     if num_splits > 1:
# [splitkv-disabled]         oaccum = torch.zeros(
# [splitkv-disabled]             (num_splits, total_q, num_heads, d),
# [splitkv-disabled]             dtype=torch.float32,
# [splitkv-disabled]             device=device,
# [splitkv-disabled]         )
# [splitkv-disabled]         lseaccum = torch.full(
# [splitkv-disabled]             (num_splits, num_heads, total_q),
# [splitkv-disabled]             float("-inf"),
# [splitkv-disabled]             dtype=torch.float32,
# [splitkv-disabled]             device=device,
# [splitkv-disabled]         )
# [splitkv-disabled]         return oaccum, lseaccum, int(oaccum.stride(0)), int(lseaccum.stride(0))
# [splitkv-disabled]     return out, lse, 0, 0
#
#
# [splitkv-disabled] def _prepare_num_splits_dynamic(
# [splitkv-disabled]     cu_seqlens_q: torch.Tensor,
# [splitkv-disabled]     kv_lens: torch.Tensor,
# [splitkv-disabled]     block_n: int,
# [splitkv-disabled]     block_m: int,
# [splitkv-disabled]     num_heads_grid: int,
# [splitkv-disabled]     num_sm: int,
# [splitkv-disabled]     num_splits_static: int,
# [splitkv-disabled]     pack_gqa: bool,
# [splitkv-disabled]     h_hk_ratio: int,
# [splitkv-disabled] ) -> torch.Tensor:
# [splitkv-disabled]     """
# [splitkv-disabled]     Per-batch dynamic split count (vLLM prepare_varlen_num_blocks pass-2, host mirror).
# [splitkv-disabled]     """
# [splitkv-disabled]     q_lens = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).detach().cpu().tolist()
# [splitkv-disabled]     kl = kv_lens.detach().cpu().tolist()
# [splitkv-disabled]     total_blocks = 0
# [splitkv-disabled]     num_n_blocks_list = []
# [splitkv-disabled]     for q_len, k_len in zip(q_lens, kl):
# [splitkv-disabled]         n_blk = _cdiv(int(k_len), block_n)
# [splitkv-disabled]         m_blk = _cdiv(int(q_len) * (h_hk_ratio if pack_gqa else 1), block_m)
# [splitkv-disabled]         num_n_blocks_list.append(n_blk)
# [splitkv-disabled]         total_blocks += m_blk * n_blk
# [splitkv-disabled]     blocks_per_sm = max(1, math.ceil(total_blocks * 1.1 * num_heads_grid / num_sm))
# [splitkv-disabled]     splits = [
# [splitkv-disabled]         max(1, min(num_splits_static, _cdiv(n_blk, blocks_per_sm)))
# [splitkv-disabled]         for n_blk in num_n_blocks_list
# [splitkv-disabled]     ]
# [splitkv-disabled]     return torch.tensor(splits, dtype=torch.int32, device=cu_seqlens_q.device)
#
#
# [splitkv-disabled] def _launch_varlen_splitkv_combine(
# [splitkv-disabled]     out,
# [splitkv-disabled]     lse,
# [splitkv-disabled]     out_accum,
# [splitkv-disabled]     lse_accum,
# [splitkv-disabled]     cu_seqlens_q,
# [splitkv-disabled]     num_splits_dynamic,
# [splitkv-disabled]     num_splits: int,
# [splitkv-disabled]     total_q: int,
# [splitkv-disabled]     num_heads: int,
# [splitkv-disabled]     d: int,
# [splitkv-disabled]     batch: int,
# [splitkv-disabled] ) -> None:
# [splitkv-disabled]     block_m = 16 if d >= 64 else 32
# [splitkv-disabled]     block_k = triton.next_power_of_2(d)
# [splitkv-disabled]     grid = (_cdiv(total_q, block_m), num_heads)
# [splitkv-disabled]     flash_varlen_gluon_splitkv_combine_kernel[grid](
# [splitkv-disabled]         out, lse, out_accum, lse_accum,
# [splitkv-disabled]         cu_seqlens_q, num_splits_dynamic,
# [splitkv-disabled]         total_q,
# [splitkv-disabled]         num_batch=batch,
# [splitkv-disabled]         num_heads=num_heads,
# [splitkv-disabled]         d=d,
# [splitkv-disabled]         o_row_stride=out.stride(0),
# [splitkv-disabled]         o_head_stride=out.stride(1),
# [splitkv-disabled]         o_split_stride=out_accum.stride(0),
# [splitkv-disabled]         lse_split_stride=lse_accum.stride(0),
# [splitkv-disabled]         n_splits=num_splits,
# [splitkv-disabled]         MAX_N_SPLITS=triton.next_power_of_2(num_splits),
# [splitkv-disabled]         BLOCK_M=block_m,
# [splitkv-disabled]         BLOCK_K=block_k,
# [splitkv-disabled]     )


def _launch_nonpaged_varlen_decode(
    q,
    k,
    v,
    out,
    lse,
    cu_seqlens_q,
    cu_seqlens_k,
    batch_ids: torch.Tensor,
    softmax_scale: float,
    is_causal: bool,
    *,
    num_heads: int,
    num_heads_k: int,
    h_hk_ratio: int,
    d: int,
    total_q: int,
) -> None:
    """Decode path: Triton dot kernel for batches with q_len <= _GLUON_DECODE_Q_THRESH."""
    n_decode = int(batch_ids.numel())
    if n_decode == 0:
        return
    pack_gqa = h_hk_ratio > 1
    block_n = _pick_block_n(d, is_decode=True)
    d_pad = triton.next_power_of_2(d)
    grid_h = num_heads_k if pack_gqa else num_heads
    grid = (n_decode, grid_h)
    flash_varlen_decode_gluon_kernel[grid](
        q, k, v, out, lse,
        cu_seqlens_q, cu_seqlens_k, batch_ids,
        q.stride(0), q.stride(1), k.stride(0), v.stride(0),
        out.stride(0), out.stride(1), k.stride(1), v.stride(1),
        total_q,
        softmax_scale,
        is_causal=is_causal,
        PACK_GQA=pack_gqa,
        H_HK_RATIO=h_hk_ratio,
        MAX_Q_LEN=_GLUON_DECODE_Q_THRESH,
        BLOCK_N=block_n,
        D=d,
        D_PAD=d_pad,
        num_warps=4,
    )


def _launch_nonpaged_varlen_wgmma(
    q,
    k,
    v,
    out,
    lse,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    softmax_scale,
    is_causal,
    batch_ids: torch.Tensor | None,
    *,
    num_stages: int = 2,
    maxnreg: int = 256,
) -> None:
    """Prefill / long-q path: one WGMMA launch (subset uses compact global cu_seqlens)."""
    total_q, num_heads, d = q.shape
    _, num_heads_k, _ = k.shape
    full_batch = cu_seqlens_q.numel() - 1
    launch_cu_q = cu_seqlens_q
    launch_cu_k = cu_seqlens_k
    if batch_ids is not None:
        batch = int(batch_ids.numel())
        launch_cu_q = _subset_cu_seqlens(cu_seqlens_q, batch_ids)
        launch_cu_k = _subset_cu_seqlens(cu_seqlens_k, batch_ids)
    else:
        batch = full_batch
    if batch == 0:
        return

    h_hk_ratio = num_heads // num_heads_k
    scale_log2 = _scale_softmax_log2(softmax_scale)
    elem_bytes = q.element_size()

    cfg = _resolve_varlen_launch_params(
        d=d,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        batch=full_batch,
        num_heads=num_heads,
        num_heads_k=num_heads_k,
        h_hk_ratio=h_hk_ratio,
        is_causal=is_causal,
        is_paged=False,
        use_kv_tma=False,
        device=q.device,
        elem_bytes=elem_bytes,
        num_stages=num_stages,
    )
    if cfg["pack_gqa"]:
        grid_m = _cdiv(max_seqlen_q * h_hk_ratio, cfg["block_m"])
    else:
        grid_m = _cdiv(max_seqlen_q, cfg["block_m"])

    grid_h = cfg["n_heads_grid"] if cfg["pack_gqa"] else num_heads
    # [splitkv-disabled] oaccum / num_splits_dynamic / combine path removed

    launch_kwargs = dict(
        num_warps=cfg["launch_num_warps"],
        num_stages=cfg["num_stages"],
        PACK_GQA=cfg["pack_gqa"],
        NUM_SPLITS=1,
        PRODUCER_WARPS=cfg["producer_warps"],
        CONSUMER_WARPS=cfg["consumer_warps"],
    )
    if cfg["maxnreg"] is not None:
        launch_kwargs["maxnreg"] = cfg["maxnreg"]

    kernel_args = (
        q, k, v, out, lse,
        out, lse, 0, 0,
        launch_cu_q, launch_cu_k, launch_cu_q,
        q.stride(0), k.stride(0), v.stride(0), out.stride(0),
        q.stride(1), k.stride(1), v.stride(1), out.stride(1),
        total_q,
        num_heads, num_heads_k, h_hk_ratio, d,
        scale_log2, is_causal,
        cfg["block_m"], cfg["block_n"],
    )
    num_sm = cfg["num_sm"]
    total_tiles = grid_m * batch * grid_h
    if _should_use_static_persistent(total_tiles, num_sm):
        tiles_per_cta = _static_persistent_tiles_per_cta(total_tiles, num_sm)
        flash_varlen_fwd_gluon_persistent_kernel[(num_sm,)](
            *kernel_args,
            GRID_M=grid_m,
            BATCH=batch,
            GRID_H=grid_h,
            NUM_SM=num_sm,
            TOTAL_TILES=total_tiles,
            TILES_PER_CTA=tiles_per_cta,
            **launch_kwargs,
        )
    else:
        grid = (grid_m, batch, grid_h)
        flash_varlen_fwd_gluon_kernel[grid](
            *kernel_args,
            **launch_kwargs,
        )


def _launch_nonpaged_varlen(
    q,
    k,
    v,
    out,
    lse,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    softmax_scale,
    is_causal,
    *,
    num_stages: int = 2,
    maxnreg: int = 256,
) -> None:
    _ensure_allocator()

    total_q, num_heads, d = q.shape
    _, num_heads_k, _ = k.shape
    h_hk_ratio = num_heads // num_heads_k

    if _GLUON_PREFILL_ONLY:
        _launch_nonpaged_varlen_wgmma(
            q, k, v, out, lse,
            cu_seqlens_q, cu_seqlens_k,
            max_seqlen_q, max_seqlen_k,
            softmax_scale, is_causal,
            None,
            num_stages=num_stages, maxnreg=maxnreg,
        )
        return

    decode_ids, prefill_ids = _split_varlen_batch_ids(cu_seqlens_q, cu_seqlens_k)

    if decode_ids.numel() > 0:
        _launch_nonpaged_varlen_decode(
            q, k, v, out, lse,
            cu_seqlens_q, cu_seqlens_k, decode_ids,
            softmax_scale, is_causal,
            num_heads=num_heads, num_heads_k=num_heads_k,
            h_hk_ratio=h_hk_ratio, d=d, total_q=total_q,
        )

    if prefill_ids.numel() > 0:
        prefill_max_q = _max_q_len_for_batches(cu_seqlens_q, prefill_ids)
        prefill_subset = (
            prefill_ids
            if prefill_ids.numel() < cu_seqlens_q.numel() - 1
            else None
        )
        _launch_nonpaged_varlen_wgmma(
            q, k, v, out, lse,
            cu_seqlens_q, cu_seqlens_k,
            prefill_max_q, max_seqlen_k,
            softmax_scale, is_causal,
            prefill_subset,
            num_stages=num_stages, maxnreg=maxnreg,
        )


def _launch_paged_varlen(
    q,
    k,
    v,
    out,
    lse,
    cu_seqlens_q,
    seqused_k,
    page_table,
    max_seqlen_q,
    max_seqlen_k,
    softmax_scale,
    is_causal,
    *,
    num_stages: int = 2,
    maxnreg: int = 256,
) -> None:
    _ensure_allocator()

    total_q, num_heads, d = q.shape
    _, block_size, num_heads_k, _ = k.shape
    batch = cu_seqlens_q.numel() - 1
    h_hk_ratio = num_heads // num_heads_k
    scale_log2 = _scale_softmax_log2(softmax_scale)
    elem_bytes = q.element_size()

    d_rounded = round_up_headdim(d)
    dv_rounded = round_up_headdimv(d)
    use_kv_tma = get_pagedkv_tma(
        90,
        block_size,
        True,
        None,
        max_seqlen_q,
        0,
        num_heads,
        num_heads_k,
        d_rounded,
        dv_rounded,
        is_causal,
        False,
        elem_bytes,
        False,
    )

    cfg = _resolve_varlen_launch_params(
        d=d,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        batch=batch,
        num_heads=num_heads,
        num_heads_k=num_heads_k,
        h_hk_ratio=h_hk_ratio,
        is_causal=is_causal,
        is_paged=True,
        use_kv_tma=use_kv_tma,
        device=q.device,
        elem_bytes=elem_bytes,
        num_stages=num_stages,
        block_size=block_size,
    )
    block_n = cfg["block_n"]
    if use_kv_tma and block_size % block_n != 0:
        use_kv_tma = False
    m_dim = max_seqlen_q * (h_hk_ratio if cfg["pack_gqa"] else 1)
    grid_h = cfg["n_heads_grid"]
    # [splitkv-disabled] oaccum / num_splits_dynamic / combine path removed
    grid = (_cdiv(m_dim, cfg["block_m"]), batch, grid_h)
    launch_kwargs = dict(
        num_warps=cfg["launch_num_warps"],
        num_stages=cfg["num_stages"],
        USE_KV_TMA=use_kv_tma,
        PACK_GQA=cfg["pack_gqa"],
        NUM_SPLITS=1,
        PRODUCER_WARPS=cfg["producer_warps"],
        CONSUMER_WARPS=cfg["consumer_warps"],
    )
    if cfg["maxnreg"] is not None:
        launch_kwargs["maxnreg"] = cfg["maxnreg"]
    flash_paged_fwd_gluon_kernel[grid](
        q, k, v, out, lse,
        out, lse, 0, 0,
        cu_seqlens_q, seqused_k, page_table, cu_seqlens_q,
        q.stride(0), q.stride(1), out.stride(0), out.stride(1),
        k.stride(0), k.stride(1), k.stride(2), page_table.stride(0),
        total_q,
        num_heads, num_heads_k, h_hk_ratio, d, block_size,
        scale_log2, is_causal,
        cfg["block_m"], block_n,
        **launch_kwargs,
    )


def flash_attn_varlen_gluon_fwd(
    q,
    k,
    v,
    out,
    lse,
    cu_seqlens_q,
    cu_seqlens_k,
    seqused_k,
    page_table,
    max_seqlen_q: int,
    max_seqlen_k: int,
    scale_softmax: float,
    is_causal: bool,
    *,
    num_stages: int = 2,
    maxnreg: int = 256,
) -> None:
    """
    Unified Gluon FA3 varlen launcher — matches flash_api.py:1146 call site.

    Dispatch (compile-time at launch):
      - page_table is None → non-paged MVP grid (cu_seqlens_k)
      - page_table set     → paged MVP grid (seqused_k)
    """
    if page_table is None:
        if cu_seqlens_k is None:
            raise ValueError("non-paged Gluon path requires cu_seqlens_k")
        _launch_nonpaged_varlen(
            q, k, v, out, lse,
            cu_seqlens_q, cu_seqlens_k,
            max_seqlen_q, max_seqlen_k, scale_softmax, is_causal,
            num_stages=num_stages, maxnreg=maxnreg,
        )
    else:
        if seqused_k is None:
            raise ValueError("paged Gluon path requires seqused_k")
        _launch_paged_varlen(
            q, k, v, out, lse,
            cu_seqlens_q, seqused_k, page_table,
            max_seqlen_q, max_seqlen_k, scale_softmax, is_causal,
            num_stages=num_stages, maxnreg=maxnreg,
        )