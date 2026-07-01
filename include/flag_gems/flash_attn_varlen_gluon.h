#pragma once

#include <optional>
#include <tuple>
#include "torch/torch.h"

namespace flag_gems {

/// FA3 Gluon varlen forward (Hopper sm90+). Host prep + libtriton_jit launch.
/// Non-paged: cu_seqlens_k + 3D k/v. Paged: seqused_k + page_table + 4D k/v cache.
std::tuple<at::Tensor, at::Tensor> flash_attn_varlen_fa3_gluon_fwd(
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
    const std::optional<at::Tensor>& out = std::nullopt);

bool flash_attn_varlen_fa3_gluon_capable(bool is_paged,
                                         bool is_softcap,
                                         bool is_local,
                                         bool has_descale);

}  // namespace flag_gems