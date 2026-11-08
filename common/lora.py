"""Minimal, dependency-free LoRA wrapping for nn.Linear and nn.MultiheadAttention
submodules. No peft dependency (not installed in the cluster's qwen3vl conda
env) -- this is the standard LoRA formulation (Hu et al. 2021) implemented
directly: y = base(x) + (alpha/r) * B(A(x)), with base's own weights frozen
and only the low-rank A/B factors trainable. B is zero-initialized (same
zero-init-for-exact-no-op-at-init convention used elsewhere in this project
for gates/probe heads), so wrapping a layer never changes its output until
training actually moves the LoRA weights.

Covers nn.MultiheadAttention too (not just plain nn.Linear): MHA's q/k/v
projections live in a single fused `in_proj_weight` Parameter, not a child
Linear module, so a naive "replace every nn.Linear child" pass would silently
skip them and leave those layers un-adapted -- which would make "LoRA on
every layer" false for any layer whose self-attention is nn.MultiheadAttention
(as CortexGuidedVideoPredictor's guidance_memory_readers are). LoRAMultiheadAttention
below adds low-rank updates to the fused in_proj (q/k/v jointly) and wraps
out_proj with a plain LoRALinear, so both halves of an MHA block are covered.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=8, alpha=16, dropout=0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r = r
        self.scaling = alpha / r
        # base is typically already on its final device/dtype (LoRA wrapping
        # happens after the host model's own .to(device) call) -- lora_A/B
        # must match it, or the very first forward hits a cpu/cuda mismatch
        # (confirmed by job-149's traceback: "mat2 is on cpu, different from
        # other tensors on cuda:0").
        self.lora_A = nn.Linear(base.in_features, r, bias=False, device=base.weight.device, dtype=base.weight.dtype)
        self.lora_B = nn.Linear(r, base.out_features, bias=False, device=base.weight.device, dtype=base.weight.dtype)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        return self.base(x) + self.lora_B(self.lora_A(self.dropout(x))) * self.scaling


class LoRAMultiheadAttention(nn.Module):
    """Wraps an existing nn.MultiheadAttention: freezes its in_proj_weight/
    in_proj_bias and out_proj, adds a low-rank update to the fused QKV
    in_proj (one A/B pair shared across q/k/v, matching how LoRA is normally
    applied to fused qkv projections), and wraps out_proj with LoRALinear.
    Forward signature matches nn.MultiheadAttention's (query, key, value,
    key_padding_mask=..., need_weights=..., attn_mask=...) for drop-in use."""

    def __init__(self, base: nn.MultiheadAttention, r=8, alpha=16, dropout=0.0):
        super().__init__()
        assert base._qkv_same_embed_dim, "LoRAMultiheadAttention assumes fused in_proj_weight"
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.embed_dim = base.embed_dim
        self.num_heads = base.num_heads
        self.r = r
        self.scaling = alpha / r
        _dev, _dtype = base.in_proj_weight.device, base.in_proj_weight.dtype
        self.lora_A = nn.Linear(self.embed_dim, r, bias=False, device=_dev, dtype=_dtype)
        self.lora_B = nn.Linear(r, 3 * self.embed_dim, bias=False, device=_dev, dtype=_dtype)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.out_proj_lora = LoRALinear(base.out_proj, r=r, alpha=alpha, dropout=dropout)
        self.base.out_proj = self.out_proj_lora

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None, **kw):
        # Project q/k/v through the frozen fused in_proj_weight plus this
        # layer's own LoRA delta, then run attention directly -- torch's own
        # F.multi_head_attention_forward always does its OWN in_proj, so
        # re-entering it after projecting here would double-project.
        delta = self.lora_B(self.lora_A(self.dropout(query))) * self.scaling  # (..., 3*embed_dim)
        dq, dk, dv = delta.split(self.embed_dim, dim=-1)
        bias = self.base.in_proj_bias
        q = F.linear(query, self.base.in_proj_weight[: self.embed_dim], bias[: self.embed_dim] if bias is not None else None) + dq
        k = F.linear(key, self.base.in_proj_weight[self.embed_dim: 2 * self.embed_dim], bias[self.embed_dim: 2 * self.embed_dim] if bias is not None else None) + dk
        v = F.linear(value, self.base.in_proj_weight[2 * self.embed_dim:], bias[2 * self.embed_dim:] if bias is not None else None) + dv
        return self._attend(q, k, v, key_padding_mask, attn_mask)

    def _attend(self, q, k, v, key_padding_mask, attn_mask):
        # q/k/v are already fully projected (base in_proj + LoRA delta).
        # Shapes: (B, L, E) if batch_first else (L, B, E); normalize to
        # batch_first for F.scaled_dot_product_attention, then transpose
        # back for the caller's expected layout.
        bf = self.base.batch_first
        if not bf:
            q, k, v = (t.transpose(0, 1) for t in (q, k, v))
        B, Lq, E = q.shape
        Lk = k.shape[1]
        H, Dh = self.num_heads, E // self.num_heads
        qh = q.view(B, Lq, H, Dh).transpose(1, 2)
        kh = k.view(B, Lk, H, Dh).transpose(1, 2)
        vh = v.view(B, Lk, H, Dh).transpose(1, 2)
        attn_bias = None
        if key_padding_mask is not None:
            attn_bias = torch.zeros(B, 1, 1, Lk, device=q.device, dtype=q.dtype)
            attn_bias.masked_fill_(key_padding_mask[:, None, None, :], float("-inf"))
        if attn_mask is not None:
            mask_bias = torch.zeros_like(attn_mask, dtype=q.dtype) if attn_mask.dtype == torch.bool else attn_mask.to(q.dtype)
            if attn_mask.dtype == torch.bool:
                mask_bias = mask_bias.masked_fill(attn_mask, float("-inf"))
            attn_bias = mask_bias if attn_bias is None else attn_bias + mask_bias
        out = F.scaled_dot_product_attention(
            qh, kh, vh, attn_mask=attn_bias,
            dropout_p=self.base.dropout if self.training else 0.0,
        )
        out = out.transpose(1, 2).contiguous().view(B, Lq, E)
        out = self.out_proj_lora(out)
        if not bf:
            out = out.transpose(0, 1)
        return out, None


def apply_lora_to_all_layers(module, r=8, alpha=16, dropout=0.0, exclude_substrings=()):
    """In-place: wraps every nn.Linear and nn.MultiheadAttention submodule of
    `module` with a LoRA adapter (frozen base weights + trainable low-rank
    A/B), skipping any whose dotted name contains one of exclude_substrings.
    Returns the list of wrapped module names, for logging/verification."""
    wrapped_names = []
    for parent_name, parent in list(module.named_modules()):
        for child_name, child in list(parent.named_children()):
            full_name = f"{parent_name}.{child_name}" if parent_name else child_name
            if any(s in full_name for s in exclude_substrings):
                continue
            if isinstance(child, nn.Linear):
                setattr(parent, child_name, LoRALinear(child, r=r, alpha=alpha, dropout=dropout))
                wrapped_names.append(full_name)
            elif isinstance(child, nn.MultiheadAttention):
                setattr(parent, child_name, LoRAMultiheadAttention(child, r=r, alpha=alpha, dropout=dropout))
                wrapped_names.append(full_name)
    return wrapped_names


def freeze_all_except_lora(module):
    """Freezes every parameter in `module` except ones belonging to a
    lora_A/lora_B submodule (by name match) -- a final safety pass in case
    some parameter wasn't covered by apply_lora_to_all_layers's own
    base.requires_grad_(False) calls (e.g. LayerNorms, biases outside any
    wrapped Linear)."""
    for name, p in module.named_parameters():
        p.requires_grad_("lora_A" in name or "lora_B" in name)


def lora_trainable_param_count(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
