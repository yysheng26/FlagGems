#pragma once

#include <optional>
#include <tuple>

#include "torch/torch.h"

namespace flag_gems {

/// FA3 varlen Triton wrapper (non-Gluon fallback for softcap / SWA / FP8 descale).
/// Mirrors flash_api.py::mha_varlan_fwd_fa3 with use_gluon=False.
/// Launches flash_varlen_fwd_fa3_kernel via libtriton_jit.
std::tuple<at::Tensor, at::Tensor> flash_attn_varlen_fa3_triton_fwd(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& cu_seqlens_q,
    const std::optional<at::Tensor>& cu_seqlens_k,
    const std::optional<at::Tensor>& seqused_k,
    const std::optional<at::Tensor>& page_table,
    int64_t max_seqlen_q,
    int64_t max_seqlen_k,
    double softmax_scale,
    bool is_causal,
    int64_t window_size_left,
    int64_t window_size_right,
    double softcap,
    const std::optional<at::Tensor>& q_descale,
    const std::optional<at::Tensor>& k_descale,
    const std::optional<at::Tensor>& v_descale,
    const std::optional<at::Tensor>& out_opt = std::nullopt);

}  // namespace flag_gems
