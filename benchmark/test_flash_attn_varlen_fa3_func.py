from typing import Any, List, Optional

import pytest
import torch

import flag_gems

from . import base, utils

vendor_name = flag_gems.vendor_name


def _is_hopper():
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9


class FlashAttnVarlenFa3Benchmark(base.Benchmark):
    """
    Benchmark for flash_attn_varlen_func with fa_version=3.

    Shape categories
    ----------------
    A. 真实 trace（Qwen3-1.7B）：与 FA2 benchmark 同源，便于直接横向对比
    B. 合成 prefill：单序列 / 小 batch，覆盖不同 seqlen 和 head_size
    C. 合成 decode：大 batch seqlen_q=1，覆盖不同 kvcache 长度
    D. GQA 变体：num_heads_q / num_heads_k 比值不同
    E. head_size 变体：64 / 128 / 256
    F. block_size 变体：16 / 32
    G. softcap：tanh 激活场景（Gemma 系列）
    H. sliding window：局部注意力场景
    """

    def set_shapes(self, shape_file_path: Optional[List[Any]] = None):
        shapes = []

        # ── A. 真实 trace：Qwen3-1.7B ─────────────────────────────────────
        # Format per entry:
        #   (cu_seq_lens_q, seqused_k, nhead_q, nhead_k,
        #    head_dim, block_size, num_blocks, soft_cap, window_size)

        all_cu_q = [
            (0, 512),
            (0, 1, 2, 72),
            tuple(range(0, 45)) + (105, 121, 137, 153, 169, 185, 201, 217, 233, 249, 265),
            tuple(range(0, 196)) + (211, 226, 240, 253, 265),
        ]
        all_sk = [
            (512,),
            (1, 1, 70),
            (515,) + (514,) * 20 + (513,) * 20 + (512,) * 14,
            (2333,)
            + (2331,) * 20 + (2330,) * 20 + (2329,) * 14
            + (2328,) * 18 + (2327,) * 15 + (2326,) * 17
            + (2325,) * 18 + (2324,) * 21 + (2323,) * 22
            + (2322,) * 24 + (2321,) * 5
            + (2320, 2319, 2318, 2317, 2316),
        ]
        for cu_q, sk in zip(all_cu_q, all_sk):
            shapes.append((cu_q, sk, 16, 8, 128, 16, 2000, None, (-1, -1)))

        # ── B. 合成 prefill：bs=1 不同 seqlen ─────────────────────────────
        for seqlen in (128, 256, 512, 1024, 2048, 4096):
            shapes.append(((0, seqlen), (seqlen,), 16, 8, 128, 16, (seqlen + 15) // 16 + 32, None, (-1, -1)))

        # prefill bs=1 不同 head_size
        for head_size in (64, 128, 256):
            shapes.append(((0, 1024), (1024,), 16, 8, head_size, 16, 128, None, (-1, -1)))

        # prefill bs=4 不同 seqlen
        for seqlen in (256, 512, 1024, 2048):
            cu_q = tuple(i * seqlen for i in range(5))
            sk = (seqlen,) * 4
            shapes.append((cu_q, sk, 16, 8, 128, 16, (4 * seqlen + 15) // 16 + 32, None, (-1, -1)))

        # prefill bs=8, seqlen=512
        cu_q8 = tuple(i * 512 for i in range(9))
        sk8 = (512,) * 8
        shapes.append((cu_q8, sk8, 16, 8, 128, 16, 512, None, (-1, -1)))

        # ── C. 合成 decode：seqlen_q=1，不同 batch/kvcache ─────────────────
        for bs, kv_len in (
            (1,   512),
            (1,  2048),
            (8,   512),
            (8,  2048),
            (16,  512),
            (16, 2048),
            (32,  512),
            (32, 1024),
            (32, 2048),
            (64, 1024),
            (64, 2048),
            (128, 1024),
            (128, 2048),
        ):
            cu_q = tuple(range(bs + 1))
            sk = (kv_len,) * bs
            num_blocks = (bs * kv_len + 15) // 16 + 64
            shapes.append((cu_q, sk, 16, 8, 128, 16, num_blocks, None, (-1, -1)))

        # ── D. GQA 变体 ────────────────────────────────────────────────────
        # decode bs=32, kv=2048
        cu_q_d32 = tuple(range(33))
        sk_d32 = (2048,) * 32
        for nq, nk in ((4, 4), (8, 2), (8, 1), (16, 2), (16, 1), (32, 8), (32, 4)):
            shapes.append((cu_q_d32, sk_d32, nq, nk, 128, 16, 4096, None, (-1, -1)))

        # prefill bs=1, seqlen=1024
        for nq, nk in ((4, 4), (8, 2), (16, 2), (32, 8)):
            shapes.append(((0, 1024), (1024,), nq, nk, 128, 16, 256, None, (-1, -1)))

        # ── E. head_size 变体（decode 场景）────────────────────────────────
        for head_size in (64, 128, 256):
            shapes.append((cu_q_d32, sk_d32, 16, 8, head_size, 16, 4096, None, (-1, -1)))

        # ── F. block_size 变体 ─────────────────────────────────────────────
        for block_size in (16, 32):
            num_blk = (32 * 2048 + block_size - 1) // block_size + 64
            shapes.append((cu_q_d32, sk_d32, 16, 8, 128, block_size, num_blk, None, (-1, -1)))
        # prefill block_size 变体
        for block_size in (16, 32):
            num_blk = (1024 + block_size - 1) // block_size + 32
            shapes.append(((0, 1024), (1024,), 16, 8, 128, block_size, num_blk, None, (-1, -1)))

        # ── G. softcap 场景（Gemma 系列）──────────────────────────────────
        for soft_cap in (10.0, 50.0):
            # decode
            shapes.append((cu_q_d32, sk_d32, 16, 8, 128, 16, 4096, soft_cap, (-1, -1)))
            # prefill
            shapes.append(((0, 1024), (1024,), 16, 8, 128, 16, 256, soft_cap, (-1, -1)))

        # ── H. sliding window ──────────────────────────────────────────────
        # prefill seqlen=2048
        for win in ((512, 0), (256, 256)):
            num_blk = (2048 + 15) // 16 + 32
            shapes.append(((0, 2048), (2048,), 16, 8, 128, 16, num_blk, None, win))
        # decode bs=32, kv=2048
        for win in ((512, 0), (256, 256)):
            shapes.append((cu_q_d32, sk_d32, 16, 8, 128, 16, 4096, None, win))

        self.shapes = shapes

    def get_input_iter(self, dtype):
        for config in self.shapes:
            inp = self._make_input(config, dtype, self.device)
            if inp is not None:
                yield inp

    def _make_input(self, config, dtype, device):
        (
            cu_query_lens,
            seqused_k,
            nhead_q,
            nhead_k,
            head_size,
            block_size,
            num_blocks,
            soft_cap,
            window_size,
        ) = config

        num_seqs  = len(cu_query_lens) - 1
        max_q_len = max(cu_query_lens[i + 1] - cu_query_lens[i] for i in range(num_seqs))
        max_kv_len = max(seqused_k)
        scale = head_size**-0.5

        query = torch.randn(
            cu_query_lens[-1], nhead_q, head_size, dtype=dtype, device=device
        )
        out = torch.empty_like(query)
        key_cache = torch.randn(
            num_blocks, block_size, nhead_k, head_size, dtype=dtype, device=device
        )
        value_cache = torch.randn_like(key_cache)

        cu_q_t = torch.tensor(cu_query_lens, dtype=torch.int32, device=device)
        sk_t   = torch.tensor(seqused_k,    dtype=torch.int32, device=device)

        max_blk_per_seq = (max_kv_len + block_size - 1) // block_size
        block_tables = torch.randint(
            0, num_blocks,
            (num_seqs, max_blk_per_seq),
            dtype=torch.int32, device=device,
        )

        # Positional args match flash_attn_varlen_func signature;
        # trailing dict is picked up as **kwargs by base.unpack_to_args_kwargs.
        return (
            query,           # q
            key_cache,       # k
            value_cache,     # v
            max_q_len,       # max_seqlen_q
            cu_q_t,          # cu_seqlens_q
            max_kv_len,      # max_seqlen_k
            None,            # cu_seqlens_k
            sk_t,            # seqused_k
            None,            # q_v
            0.0,             # dropout_p
            scale,           # softmax_scale
            True,            # causal
            list(window_size),  # window_size
            soft_cap if soft_cap is not None else 0.0,  # softcap
            None,            # alibi_slopes
            False,           # deterministic
            False,           # return_attn_probs
            block_tables,    # block_table
            False,           # return_softmax_lse
            out,             # out
            None,            # scheduler_metadata
            None,            # q_descale
            None,            # k_descale
            None,            # v_descale
            {"fa_version": 3},
        )


@pytest.mark.skipif(
    not _is_hopper(),
    reason="FA3 requires Hopper GPU (sm_90+)",
)
@pytest.mark.skipif(
    utils.SkipVersion("vllm", "<0.9"),
    reason="vLLM version prior to 0.9 does not include the flash_attn_varlen_func API.",
)
@pytest.mark.skipif(vendor_name == "hygon", reason="Not working")
@pytest.mark.skipif(vendor_name == "cambricon", reason="Not supported")
@pytest.mark.flash_attn_varlen_func
def test_flash_attn_varlen_fa3_func(monkeypatch):
    monkeypatch.setenv("VLLM_CONFIGURE_LOGGING", "0")

    from vllm.vllm_flash_attn.flash_attn_interface import flash_attn_varlen_func as _vllm_fa

    def vllm_fa3(*args, **kwargs):
        # base.unpack_to_args_kwargs 把末尾 dict 展开为 kwargs，
        # 这里把 fa_version 剥出来单独传给 vllm，避免重复
        kwargs.pop("fa_version", None)
        return _vllm_fa(*args, fa_version=3, **kwargs)

    bench = FlashAttnVarlenFa3Benchmark(
        op_name="flash_attn_varlen_fa3_func",
        torch_op=vllm_fa3,
        gems_op=flag_gems.ops.flash_attn_varlen_func,
        dtypes=[torch.float16, torch.bfloat16],
    )
    bench.run()


# ── Non-paged benchmark ────────────────────────────────────────────────────────
# non-paged 路径：k/v 是 flat [total_kv, nhead_k, d] tensor，通过 cu_seqlens_k
# 指定每条序列的 KV 范围，不需要 block_tables。
# Size Detail 里 k shape 为 3 维，可与上面的 paged（4 维）直接区分。


class FlashAttnVarlenFa3NonPagedBenchmark(base.Benchmark):
    """
    Benchmark for flash_attn_varlen_func with fa_version=3, non-paged KV path.

    Shape categories
    ----------------
    A. 真实 trace（Qwen3-1.7B）：与 paged benchmark 相同 shape，便于对比两路径开销
    B. 合成 prefill：bs=1 不同 seqlen / head_size
    C. 合成 prefill：bs=4 不同 seqlen
    D. 合成 decode：bs=1/8/16/32/64/128 × sk=512/1024/2048
    E. GQA 变体：decode bs=32 sk=2048
    F. head_size 变体：decode bs=32 sk=2048 / prefill bs=1 sq=1024
    G. softcap：decode + prefill
    H. sliding window：prefill sq=2048 / decode bs=32 sk=2048
    """

    def set_shapes(self, shape_file_path=None):
        # config 格式：(cu_seq_lens_q, cu_seq_lens_k, nhead_q, nhead_k,
        #               head_dim, soft_cap, window_size)
        # cu_seq_lens_k 由 seqused_k 累加而来，直接在这里构造。
        shapes = []

        def make_cu(seq_lens):
            """把每条序列的长度列表转成 cumsum 形式的 tuple"""
            result = [0]
            for l in seq_lens:
                result.append(result[-1] + l)
            return tuple(result)

        # ── A. 真实 trace：Qwen3-1.7B ──────────────────────────────────────
        all_cu_q = [
            (0, 512),
            (0, 1, 2, 72),
            tuple(range(0, 45)) + (105, 121, 137, 153, 169, 185, 201, 217, 233, 249, 265),
            tuple(range(0, 196)) + (211, 226, 240, 253, 265),
        ]
        all_sk = [
            (512,),
            (1, 1, 70),
            (515,) + (514,) * 20 + (513,) * 20 + (512,) * 14,
            (2333,)
            + (2331,) * 20 + (2330,) * 20 + (2329,) * 14
            + (2328,) * 18 + (2327,) * 15 + (2326,) * 17
            + (2325,) * 18 + (2324,) * 21 + (2323,) * 22
            + (2322,) * 24 + (2321,) * 5
            + (2320, 2319, 2318, 2317, 2316),
        ]
        for cu_q, sk in zip(all_cu_q, all_sk):
            shapes.append((cu_q, make_cu(sk), 16, 8, 128, None, (-1, -1)))

        # ── B. 合成 prefill bs=1 ───────────────────────────────────────────
        for seqlen in (128, 256, 512, 1024, 2048, 4096):
            shapes.append(((0, seqlen), (0, seqlen), 16, 8, 128, None, (-1, -1)))
        # head_size 变体
        for head_size in (64, 128, 256):
            shapes.append(((0, 1024), (0, 1024), 16, 8, head_size, None, (-1, -1)))

        # ── C. 合成 prefill bs=4 ───────────────────────────────────────────
        for seqlen in (256, 512, 1024, 2048):
            cu_q = tuple(i * seqlen for i in range(5))
            cu_k = tuple(i * seqlen for i in range(5))
            shapes.append((cu_q, cu_k, 16, 8, 128, None, (-1, -1)))

        # ── D. 合成 decode sq=1 ────────────────────────────────────────────
        for bs, kv in (
            (1, 512), (1, 2048),
            (8, 512), (8, 2048),
            (16, 512), (16, 2048),
            (32, 512), (32, 1024), (32, 2048),
            (64, 1024), (64, 2048),
            (128, 1024), (128, 2048),
        ):
            cu_q = tuple(range(bs + 1))
            cu_k = tuple(i * kv for i in range(bs + 1))
            shapes.append((cu_q, cu_k, 16, 8, 128, None, (-1, -1)))

        # ── E. GQA 变体 decode bs=32 sk=2048 ──────────────────────────────
        cu_q_d32 = tuple(range(33))
        cu_k_d32 = tuple(i * 2048 for i in range(33))
        for nq, nk in ((4, 4), (8, 2), (8, 1), (16, 2), (16, 1), (32, 8), (32, 4)):
            shapes.append((cu_q_d32, cu_k_d32, nq, nk, 128, None, (-1, -1)))

        # ── F. head_size 变体 ─────────────────────────────────────────────
        # decode bs=32 sk=2048
        for hd in (64, 128, 256):
            shapes.append((cu_q_d32, cu_k_d32, 16, 8, hd, None, (-1, -1)))
        # prefill bs=1 sq=1024
        for hd in (64, 128, 256):
            shapes.append(((0, 1024), (0, 1024), 16, 8, hd, None, (-1, -1)))

        # ── G. softcap ────────────────────────────────────────────────────
        for sc in (10.0, 50.0):
            shapes.append((cu_q_d32, cu_k_d32, 16, 8, 128, sc, (-1, -1)))
            shapes.append(((0, 1024), (0, 1024), 16, 8, 128, sc, (-1, -1)))

        # ── H. sliding window ─────────────────────────────────────────────
        for win in ((512, 0), (256, 256)):
            shapes.append(((0, 2048), (0, 2048), 16, 8, 128, None, win))
        for win in ((512, 0), (256, 256)):
            shapes.append((cu_q_d32, cu_k_d32, 16, 8, 128, None, win))

        self.shapes = shapes

    def get_input_iter(self, dtype):
        for config in self.shapes:
            inp = self._make_input(config, dtype, self.device)
            if inp is not None:
                yield inp

    def _make_input(self, config, dtype, device):
        (
            cu_query_lens,
            cu_kv_lens,
            nhead_q,
            nhead_k,
            head_size,
            soft_cap,
            window_size,
        ) = config

        num_seqs   = len(cu_query_lens) - 1
        total_q    = cu_query_lens[-1]
        total_kv   = cu_kv_lens[-1]
        max_q_len  = max(cu_query_lens[i+1] - cu_query_lens[i] for i in range(num_seqs))
        max_kv_len = max(cu_kv_lens[i+1] - cu_kv_lens[i] for i in range(num_seqs))
        scale      = head_size ** -0.5

        query       = torch.randn(total_q,  nhead_q, head_size, dtype=dtype, device=device)
        key         = torch.randn(total_kv, nhead_k, head_size, dtype=dtype, device=device)
        value       = torch.randn_like(key)
        out         = torch.empty_like(query)

        cu_q_t = torch.tensor(cu_query_lens, dtype=torch.int32, device=device)
        cu_k_t = torch.tensor(cu_kv_lens,    dtype=torch.int32, device=device)

        # non-paged 路径：传 cu_seqlens_k，不传 seqused_k 和 block_table
        return (
            query,           # q          — [total_q, nhead_q, d]
            key,             # k          — [total_kv, nhead_k, d]  ← 3维，区别于 paged 的4维
            value,           # v          — [total_kv, nhead_k, d]
            max_q_len,       # max_seqlen_q
            cu_q_t,          # cu_seqlens_q
            max_kv_len,      # max_seqlen_k
            cu_k_t,          # cu_seqlens_k  ← non-paged 的关键标志
            None,            # seqused_k
            None,            # q_v
            0.0,             # dropout_p
            scale,           # softmax_scale
            True,            # causal
            list(window_size),  # window_size
            soft_cap if soft_cap is not None else 0.0,  # softcap
            None,            # alibi_slopes
            False,           # deterministic
            False,           # return_attn_probs
            None,            # block_table  ← None 表示 non-paged
            False,           # return_softmax_lse
            out,             # out
            None,            # scheduler_metadata
            None,            # q_descale
            None,            # k_descale
            None,            # v_descale
            {"fa_version": 3},
        )


@pytest.mark.skipif(
    not _is_hopper(),
    reason="FA3 requires Hopper GPU (sm_90+)",
)
@pytest.mark.skipif(
    utils.SkipVersion("vllm", "<0.9"),
    reason="vLLM version prior to 0.9 does not include the flash_attn_varlen_func API.",
)
@pytest.mark.skipif(vendor_name == "hygon", reason="Not working")
@pytest.mark.skipif(vendor_name == "cambricon", reason="Not supported")
@pytest.mark.flash_attn_varlen_func
def test_flash_attn_varlen_fa3_func_non_paged(monkeypatch):
    monkeypatch.setenv("VLLM_CONFIGURE_LOGGING", "0")

    from vllm.vllm_flash_attn.flash_attn_interface import flash_attn_varlen_func as _vllm_fa

    def vllm_fa3(*args, **kwargs):
        kwargs.pop("fa_version", None)
        return _vllm_fa(*args, fa_version=3, **kwargs)

    bench = FlashAttnVarlenFa3NonPagedBenchmark(
        op_name="flash_attn_varlen_fa3_func_non_paged",
        torch_op=vllm_fa3,
        gems_op=flag_gems.ops.flash_attn_varlen_func,
        dtypes=[torch.float16, torch.bfloat16],
    )
    bench.run()
