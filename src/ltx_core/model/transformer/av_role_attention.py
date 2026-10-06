# RAVEdit local modifications to LTX-2; see THIRD_PARTY_NOTICES.md.
"""Opt-in source/target-role bias for AV cross-attention.

The bias is rank one in query/key space. An extra Q/K coordinate implements it
without allocating a [batch, heads, queries, keys] mask. Coordinates are appended
AFTER Q/K normalization and RoPE; original positions are unchanged.
"""

from math import sqrt
from typing import Protocol

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class ScaledAttention(Protocol):
    def __call__(self, q: Tensor, k: Tensor, v: Tensor, heads: int, *, scale: float) -> Tensor: ...


class AVRoleBias(nn.Module):
    """Per-query/head affine bias from normalized features and raw noise level.

    All parameters start at zero without consuming RNG. Positive output favors
    reference keys; negative output favors target keys. This is independent of
    LTX's existing per-head output gate and cross-attention AdaLN gates.
    """

    def __init__(self, query_dim: int, heads: int, *, device: torch.device, dtype: torch.dtype) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(heads, query_dim, device=device, dtype=dtype))
        self.bias = nn.Parameter(torch.zeros(heads, device=device, dtype=dtype))
        self.sigma_weight = nn.Parameter(torch.zeros(heads, device=device, dtype=dtype))

    def forward(self, x: Tensor, sigma: Tensor) -> Tensor:
        """Return [B, N_query, H]; sigma is a scalar or [B], NOT token timesteps."""
        features = F.rms_norm(x, (x.shape[-1],), eps=1e-6)
        # PEFT keeps modules_to_save in FP32 even when the frozen model is BF16.
        # This also supports inference callers outside an autocast context.
        bias = F.linear(features.to(self.weight.dtype), self.weight, self.bias)
        return bias + sigma.to(bias).reshape(-1, 1, 1) * self.sigma_weight.to(bias)


def role_biased_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    heads: int,
    query_bias: Tensor,
    query_reference_mask: Tensor,
    key_reference_mask: Tensor,
    attention: ScaledAttention,
) -> Tensor:
    """Compute softmax(QK/sqrt(d) + b_i * target_i * reference_j) V.

    Q/K/V are [B,N,H*d], query_bias [B,Nq,H], role masks boolean [B,N].
    Append one coordinate and zero-pad to a multiple of eight for fused kernels.
    Keep original Q/K unchanged and explicitly retain the original 1/sqrt(d)
    scale. Only the bias coordinate is multiplied by sqrt(d). This avoids an
    extra BF16 rounding of all original queries. V padding is discarded.
    Floating-point rounding/kernel selection can differ from the original path.
    """
    batch, nq, dim = q.shape
    d = dim // heads
    d_new = ((d + 1 + 7) // 8) * 8
    q = q.reshape(batch, nq, heads, d)
    k = k.reshape(batch, -1, heads, d)
    v = v.reshape(batch, -1, heads, d)
    bias = query_bias.to(q) * (~query_reference_mask).unsqueeze(-1)
    key_role = key_reference_mask.to(k)[:, :, None, None].expand(-1, -1, heads, 1)
    q = F.pad(torch.cat((q, bias.unsqueeze(-1) * sqrt(d)), dim=-1), (0, d_new - d - 1))
    k = F.pad(torch.cat((k, key_role), dim=-1), (0, d_new - d - 1))
    v = F.pad(v, (0, d_new - d))
    out = attention(q.flatten(2), k.flatten(2), v.flatten(2), heads, scale=1 / sqrt(d))
    return out.reshape(batch, nq, heads, d_new)[..., :d].reshape(batch, nq, dim)
