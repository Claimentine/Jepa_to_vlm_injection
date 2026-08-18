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
        assert mode in {"jepa", "random", "constant"}
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
        # A content-free learned control.  It has the same output shape as the
        # temporal JEPA condition, but cannot encode anything about a clip.
        # It deliberately bypasses the JEPA pooler: its purpose is to test
        # whether *video-dependent* conditioning, rather than merely adding
        # trainable prompt parameters, is useful.
        if mode == "constant":
            self.constant_tokens = nn.Parameter(
                torch.zeros(1, 64 if pooling == "temporal" else n_tokens, hidden_size)
            )

    def forward(self, vjepa_feats):
        # vjepa_feats: (B, T, N, D) float tensor
        if self.mode == "constant":
            if self.pooling == "temporal" and vjepa_feats.shape[1] != self.constant_tokens.shape[1]:
                raise ValueError("constant temporal injector was initialized for a different number of frames")
            return self.constant_tokens.expand(vjepa_feats.shape[0], -1, -1)
        if self.mode == "random":
            vjepa_feats = torch.randn(
                vjepa_feats.shape, generator=self._rng, dtype=vjepa_feats.dtype
            ).to(vjepa_feats.device)
        pooled = self.pooler(vjepa_feats)
        return self.adapter(pooled)  # temporal: (B, T, hidden_size); global: (B, n_tokens, hidden_size)


class LayerWiseJEPAInjector(nn.Module):
    """Turn JEPA tokens into zero-initialized, gated decoder-layer updates.

    ``mode='film'`` applies sample-specific FiLM-style scale/shift to the
    hidden states entering selected decoder blocks.  ``mode='cross_attn'``
    lets every language token attend to the 64 temporally ordered JEPA tokens.
    The gates start at zero, so attaching this module is initially an exact
    no-op for the frozen VLM rather than a destructive distribution shift.
    """

    def __init__(self, hidden_size, layer_indices, condition_mode="jepa", seed=0,
                 mode="film", nhead=8):
        super().__init__()
        assert mode in {"film", "cross_attn"}
        self.mode = mode
        self.layer_indices = tuple(layer_indices)
        self.conditioner = JepaInjector(
            hidden_size=hidden_size, mode=condition_mode, seed=seed, pooling="temporal"
        )
        keys = [str(i) for i in self.layer_indices]
        if mode == "film":
            self.modulators = nn.ModuleDict({k: nn.Linear(hidden_size, 2 * hidden_size) for k in keys})
            for modulator in self.modulators.values():
                nn.init.zeros_(modulator.weight)
                nn.init.zeros_(modulator.bias)
        else:
            self.cross_attention = nn.ModuleDict({
                k: nn.MultiheadAttention(hidden_size, nhead, batch_first=True) for k in keys
            })
            self.query_norm = nn.ModuleDict({k: nn.LayerNorm(hidden_size) for k in keys})
        self.gates = nn.ParameterDict({k: nn.Parameter(torch.zeros(())) for k in keys})

    def condition(self, vjepa_feats):
        """Return (B, T, H) once; reuse it for both A/B likelihood calls."""
        return self.conditioner(vjepa_feats)

    def apply(self, layer_idx, hidden_states, condition):
        key = str(layer_idx)
        if self.mode == "film":
            global_condition = condition.mean(dim=1)
            gamma, beta = self.modulators[key](global_condition).chunk(2, dim=-1)
            # Bounded scale avoids an early optimizer step exploding a frozen
            # decoder's activations.
            update = 0.1 * torch.tanh(gamma).unsqueeze(1) * hidden_states + beta.unsqueeze(1)
        else:
            update, _ = self.cross_attention[key](
                self.query_norm[key](hidden_states), condition, condition, need_weights=False
            )
        return hidden_states + self.gates[key] * update


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


def resolve_decoder_layers(model):
    """Return Qwen decoder blocks across the small layout variations in HF."""
    lm = model.model.language_model
    layers = getattr(lm, "layers", None)
    if layers is None and hasattr(lm, "model"):
        layers = getattr(lm.model, "layers", None)
    if layers is None:
        raise AttributeError("could not find decoder layers below model.model.language_model")
    return layers


def select_layer_indices(n_layers, strategy):
    """Pick a small, reproducible set of decoder layers for deep injection."""
    if strategy == "all":
        return list(range(n_layers))
    if strategy == "middle4":
        start = max(0, n_layers // 2 - 2)
        return list(range(start, min(start + 4, n_layers)))
    if strategy == "last4":
        return list(range(max(0, n_layers - 4), n_layers))
    if strategy == "uniform4":
        if n_layers <= 4:
            return list(range(n_layers))
        return sorted({round(i * (n_layers - 1) / 3) for i in range(4)})
    raise ValueError(f"unknown layer strategy: {strategy}")


class DecoderLayerInjectionHook:
    """Apply a :class:`LayerWiseJEPAInjector` before selected decoder blocks.

    The condition is set once per video and remains constant while scoring the
    two answer continuations.  This avoids stochastic dropout differences
    between the A and B likelihoods.
    """

    def __init__(self, model, injector):
        self.injector = injector
        self._current = None
        self._handles = []
        layers = resolve_decoder_layers(model)
        for idx in injector.layer_indices:
            if idx < 0 or idx >= len(layers):
                raise IndexError(f"decoder layer {idx} outside [0, {len(layers)})")
            self._handles.append(layers[idx].register_forward_pre_hook(
                self._make_hook(idx), with_kwargs=True
            ))

    def _make_hook(self, layer_idx):
        def pre_hook(module, args, kwargs):
            if self._current is None:
                return args, kwargs
            if args:
                hidden_states = args[0]
                return (self.injector.apply(layer_idx, hidden_states, self._current),) + args[1:], kwargs
            hidden_states = kwargs.get("hidden_states")
            if hidden_states is None:
                return args, kwargs
            kwargs = dict(kwargs)
            kwargs["hidden_states"] = self.injector.apply(layer_idx, hidden_states, self._current)
            return args, kwargs
        return pre_hook

    def set(self, condition):
        self._current = condition

    def clear(self):
        self._current = None

    def remove(self):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


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
