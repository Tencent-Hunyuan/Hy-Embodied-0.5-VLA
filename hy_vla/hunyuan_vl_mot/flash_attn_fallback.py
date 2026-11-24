# coding=utf-8
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch fallback for the flash-attn functions used by Hy-VLA.

flash-attn only builds for NVIDIA GPUs. When it is not installed (Intel
XPU, CPU), the two functions below take the place of
``flash_attn.flash_attn_func`` and ``flash_attn.flash_attn_varlen_func``:
same arguments, same ``(..., seqlen, heads, head_dim)`` layout, computed
with ``torch.nn.functional.scaled_dot_product_attention``. When flash-attn
is installed, the callers import it first and nothing changes.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F  # noqa: N812


def _sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float,
    softmax_scale: Optional[float],
    causal: bool,
) -> torch.Tensor:
    """SDPA on ``(B, H, S, D)`` inputs with flash-attn's GQA and causal conventions."""
    if k.shape[1] != q.shape[1]:
        # Grouped-query attention: query head h reads key/value head h // groups.
        groups = q.shape[1] // k.shape[1]
        k = k.repeat_interleave(groups, dim=1)
        v = v.repeat_interleave(groups, dim=1)
    attn_mask = None
    seqlen_q, seqlen_k = q.shape[-2], k.shape[-2]
    if causal and seqlen_q != seqlen_k:
        # flash-attn aligns the causal mask to the bottom-right corner;
        # SDPA's is_causal aligns it to the top-left one.
        rows = torch.arange(seqlen_q, device=q.device)[:, None]
        cols = torch.arange(seqlen_k, device=q.device)[None, :]
        attn_mask = cols <= rows + (seqlen_k - seqlen_q)
        causal = False
    return F.scaled_dot_product_attention(
        q, k, v,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=causal,
        scale=softmax_scale,
    )


def flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
) -> torch.Tensor:
    """Drop-in for ``flash_attn.flash_attn_func``: inputs and output are ``(B, S, H, D)``."""
    out = _sdpa(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
        dropout_p, softmax_scale, causal,
    )
    return out.transpose(1, 2)


def flash_attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
) -> torch.Tensor:
    """Drop-in for ``flash_attn.flash_attn_varlen_func``.

    ``q``, ``k`` and ``v`` are packed ``(total_tokens, H, D)`` tensors; the
    i-th sequence spans ``cu_seqlens[i]:cu_seqlens[i + 1]`` and attends only
    to itself. ``max_seqlen_q`` and ``max_seqlen_k`` are accepted for
    signature compatibility and not needed here.
    """
    bounds_q = cu_seqlens_q.tolist()
    bounds_k = cu_seqlens_k.tolist()
    out = q.new_zeros(q.shape[0], q.shape[1], v.shape[-1])
    for i in range(len(bounds_q) - 1):
        q0, q1 = bounds_q[i], bounds_q[i + 1]
        k0, k1 = bounds_k[i], bounds_k[i + 1]
        if q1 <= q0 or k1 <= k0:
            continue
        seg = _sdpa(
            q[q0:q1].transpose(0, 1).unsqueeze(0),
            k[k0:k1].transpose(0, 1).unsqueeze(0),
            v[k0:k1].transpose(0, 1).unsqueeze(0),
            dropout_p, softmax_scale, causal,
        )
        out[q0:q1] = seg.squeeze(0).transpose(0, 1)
    return out
