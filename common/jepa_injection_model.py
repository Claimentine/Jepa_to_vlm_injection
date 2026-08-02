"""
Soft-prompt injection of cached V-JEPA2 features ("vjepa_feats" in the
ThinkJEPA npz cache) into a FROZEN Qwen3-VL-2B-Thinking, for the task
classification probe.

Why a hook instead of just concatenating embeddings at the top-level call:
Qwen3VLModel.forward() (transformers/models/qwen3_vl/modeling_qwen3_vl.py)
requires *exactly one* of input_ids / inputs_embeds, and does the
image/video-token merge internally using input_ids to locate placeholder
positions. So we can't just hand it a custom inputs_embeds and also keep
input_ids for merge-position lookup at the top level.

Instead we let the normal call happen (input_ids + pixel_values_videos, same
code path as the zero-shot probe -- proven to work), and register a forward
pre-hook on `model.model.language_model` (the Qwen3VLTextModel decoder
stack). By the time Qwen3VLModel.forward calls
`self.language_model(inputs_embeds=..., ...)`, the video merge has already
happened -- our hook overwrites the reserved soft-prompt positions in that
already-merged `inputs_embeds` tensor right before it enters the transformer
layers. Verified against the installed transformers source
(qwen3vl conda env, transformers 5.1.0) at the specific call site:
Qwen3VLModel.forward, ~line 1274: `self.language_model(input_ids=None,
inputs_embeds=inputs_embeds, visual_pos_masks=..., ...)`.
"""
import torch
import torch.nn as nn


class JepaPooler(nn.Module):
    """Attention-pool a (T, N, D) vjepa_feats tensor -> a single (D,) vector.

    v1 (global) pooling. POST-MORTEM: this is the pooling that shipped in the
    classification experiment where `random` beat `jepa` (macro 0.626 vs
    0.440, see injection_probe_report_v2.txt). Collapsing all T*N=8192
    tokens through a single learned query into one vector almost certainly
    destroys the fine-grained detail needed to tell visually similar
    furniture sub-categories apart -- kept here only for comparison, prefer
    JepaPoolerTemporal below for new runs.
    """

    def __init__(self, dim=1024, nhead=8, p=0.1):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.attn = nn.MultiheadAttention(dim, nhead, batch_first=True, dropout=p)
        self.ln = nn.LayerNorm(dim)

    def forward(self, feats):
        # feats: (B, T, N, D) -> flatten (T*N) as the pooling axis
        B, T, N, D = feats.shape
        x = self.ln(feats.reshape(B, T * N, D))
        q = self.query.expand(B, -1, -1)
        pooled, _ = self.attn(q, x, x)
        return pooled.squeeze(1)  # (B, D)


class JepaPoolerTemporal(nn.Module):
    """Attention-pool a (B, T, N, D) vjepa_feats tensor -> (B, T, D), pooling
    only over the N=128 per-frame spatial/patch tokens and KEEPING the T=64
    temporal axis intact.

    This mirrors cache_train/models.py's TrajectoryReadoutMLP._pool_temporal_tokens
    exactly (same one-query-per-frame attention pool) -- ThinkJEPA's own
    trajectory readout never collapses time, only space. The v1 JepaPooler
    above collapsed both, which is the leading suspect for why `random`
    beat `jepa` in the first classification run.
    """

    def __init__(self, dim=1024, nhead=8, p=0.1):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.attn = nn.MultiheadAttention(dim, nhead, batch_first=True, dropout=p)
        self.ln = nn.LayerNorm(dim)

    def forward(self, feats):
        B, T, N, D = feats.shape
        x = self.ln(feats.reshape(B * T, N, D))
        q = self.query.expand(B * T, -1, -1)
        pooled, _ = self.attn(q, x, x)
        return pooled.view(B, T, D)  # (B, T, D) -- temporal structure preserved


class SoftPromptAdapter(nn.Module):
    """pooled (B, in_dim) -> (B, n_tokens, hidden_size) soft-prompt embeddings.

    v1: bottlenecks through a single flat MLP producing n_tokens*hidden_size
    at once from ONE pooled vector -- see SoftPromptAdapterTemporal for the
    per-frame-token version paired with JepaPoolerTemporal.
    """

    def __init__(self, in_dim=1024, hidden_size=2048, n_tokens=8, mlp_hidden=2048, p=0.1):
        super().__init__()
        self.n_tokens = n_tokens
        self.hidden_size = hidden_size
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(p),
            nn.Linear(mlp_hidden, n_tokens * hidden_size),
        )

    def forward(self, pooled):
        B = pooled.shape[0]
        out = self.mlp(pooled)
        return out.view(B, self.n_tokens, self.hidden_size)


class SoftPromptAdapterTemporal(nn.Module):
    """(B, T, in_dim) -> (B, T, hidden_size), a per-token projection shared
    across all T positions (no cross-token bottleneck) -- each of the T=64
    pooled per-frame vectors becomes its own soft-prompt token, so the VLM
    sees the actual temporal evolution instead of one global summary.
    """

    def __init__(self, in_dim=1024, hidden_size=2048, mlp_hidden=2048, p=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(p),
            nn.Linear(mlp_hidden, hidden_size),
        )

    def forward(self, pooled_seq):
        # pooled_seq: (B, T, in_dim) -> (B, T, hidden_size), same MLP per position
        return self.mlp(pooled_seq)


class JepaInjector(nn.Module):
    """Full adapter: vjepa_feats -> pooled -> soft-prompt tokens.

    `mode="jepa"`   -> pools and projects the real vjepa_feats.
    `mode="random"` -> ignores vjepa_feats, projects a fixed-seed random
                       tensor of the same shape through an *identical*
                       architecture/param-count adapter. This is the control
                       condition: if this does just as well as mode="jepa",
                       the gain is coming from "extra trainable soft-prompt
                       capacity", not from the JEPA world-model content.
    `pooling="temporal"` (default, new) -> JepaPoolerTemporal + SoftPromptAdapterTemporal,
                       n_tokens is forced to T (64): one soft-prompt token per
                       frame, spatial-only pooling. Fixes the diagnosed
                       over-pooling problem from the v1 run.
    `pooling="global"` (legacy) -> the original JepaPooler + SoftPromptAdapter
                       that collapsed everything into n_tokens tokens from a
                       single global vector. Kept for direct comparison against
                       the already-reported random-beats-jepa result.
    """

    def __init__(self, jepa_dim=1024, hidden_size=2048, n_tokens=8, mode="jepa", seed=0, pooling="temporal"):
        super().__init__()
        assert mode in {"jepa", "random"}
        assert pooling in {"temporal", "global"}
        self.mode = mode
        self.pooling = pooling
        if pooling == "temporal":
            self.pooler = JepaPoolerTemporal(dim=jepa_dim)
            self.adapter = SoftPromptAdapterTemporal(in_dim=jepa_dim, hidden_size=hidden_size)
        else:
            self.pooler = JepaPooler(dim=jepa_dim)
            self.adapter = SoftPromptAdapter(in_dim=jepa_dim, hidden_size=hidden_size, n_tokens=n_tokens)
        self._rng = torch.Generator().manual_seed(seed)
        self._jepa_dim = jepa_dim

    def forward(self, vjepa_feats):
        # vjepa_feats: (B, T, N, D) float tensor
        if self.mode == "random":
            vjepa_feats = torch.randn(
                vjepa_feats.shape, generator=self._rng, dtype=vjepa_feats.dtype
            ).to(vjepa_feats.device)
        pooled = self.pooler(vjepa_feats)
        return self.adapter(pooled)  # temporal: (B, T, hidden_size); global: (B, n_tokens, hidden_size)


class LanguageModelInjectionHook:
    """Registers a forward-pre-hook on `model.model.language_model` that
    overwrites the first `n_tokens` positions of `inputs_embeds` with
    whatever soft-prompt tensor is currently set via `.set(...)`.

    Usage per forward call:
        hook.set(soft_prompt_tensor)   # (B, n_tokens, hidden_size)
        model(input_ids=..., attention_mask=..., pixel_values_videos=..., ...)
        hook.clear()
    """

    def __init__(self, model, n_tokens):
        self.n_tokens = n_tokens
        self._current = None
        lm = model.model.language_model
        self._handle = lm.register_forward_pre_hook(self._pre_hook, with_kwargs=True)

    def set(self, soft_prompt):
        self._current = soft_prompt

    def clear(self):
        self._current = None

    def remove(self):
        self._handle.remove()

    def _pre_hook(self, module, args, kwargs):
        if self._current is None:
            return args, kwargs
        inputs_embeds = kwargs.get("inputs_embeds", None)
        if inputs_embeds is None:
            return args, kwargs
        # During generate()'s incremental (KV-cached) decode steps after the
        # first prefill call, inputs_embeds only holds the single new token
        # -- the placeholder positions were already injected during prefill
        # and now live in the KV cache, so there is nothing to overwrite.
        if inputs_embeds.shape[1] < self.n_tokens:
            return args, kwargs
        sp = self._current.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.clone()
        inputs_embeds[:, : self.n_tokens, :] = sp
        kwargs["inputs_embeds"] = inputs_embeds
        return args, kwargs


def prepend_placeholder_tokens(input_ids, attention_mask, n_tokens, placeholder_id):
    """Prepend `n_tokens` copies of `placeholder_id` to input_ids/attention_mask.
    Their embeddings get overwritten by LanguageModelInjectionHook before the
    decoder ever sees them -- which token id we pick doesn't matter as long
    as it isn't the model's image/video token id (asserted by the caller).
    """
    B, _ = input_ids.shape
    prefix_ids = torch.full((B, n_tokens), placeholder_id, dtype=input_ids.dtype, device=input_ids.device)
    prefix_mask = torch.ones((B, n_tokens), dtype=attention_mask.dtype, device=attention_mask.device)
    return torch.cat([prefix_ids, input_ids], dim=1), torch.cat([prefix_mask, attention_mask], dim=1)
