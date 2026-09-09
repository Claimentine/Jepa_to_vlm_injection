"""A small shared module trained live by both injection directions at once,
replacing the abandoned warm-start-initialization approach.

fit_warmstart_alignment.py tried a one-shot closed-form fit (ridge
regression on pooled JEPA/VLM features) to *initialize* each direction's
cross-modal adapter before independent training. A held-out R^2 check
(fit_warmstart_alignment.py --holdout_frac) showed that was fitting noise
for the reverse direction (held-out R^2 of -0.25 and -1.21 -- worse than
predicting the mean -- from regressing a 2048-dim input against ~1463
samples, more free parameters than data points) and only a modest real
signal for the forward direction (held-out R^2=0.33). Ablation runs
(job-18/job-19) confirmed no measurable training benefit either way.

CrossModalBridge instead makes the two linear projections LIVE, TRAINABLE
parameters, shared by construction between the forward and reverse training
loops (see train_joint_coupled.py's wiring) -- so they're pulled by both
directions' real task losses every step, plus an explicit cycle-consistency
term, rather than fit once on a small sample before either model has ever
been trained. This is a standard CycleGAN-style consistency mechanism, not
a hard weight-tying constraint -- both directions can still primarily fit
their own task, with the cycle term as a soft regularizer pulling the
bridge toward a mutually-consistent solution.

Pooling convention matches fit_warmstart_alignment.py's: both heads operate
on globally pooled (mean over every temporal/spatial/token/layer position)
feature vectors, since that's what the forward direction's
SoftPromptAdapterTemporal.mlp and the reverse direction's
guidance_old_adapter/guidance_new_adapter both actually consume (per-position
weights shared across positions in both cases -- see
common/jepa_injection_model.py and cache_train/thinker_predictor.py).
"""
import torch
import torch.nn as nn


class CrossModalBridge(nn.Module):
    def __init__(self, jepa_dim=1024, vlm_dim=2048):
        super().__init__()
        self.jepa_to_vlm = nn.Linear(jepa_dim, vlm_dim)
        self.vlm_to_jepa = nn.Linear(vlm_dim, jepa_dim)

    def cycle_loss(self, pooled_jepa=None, pooled_vlm=None):
        """Bidirectional cycle-consistency: round-tripping a pooled vector
        through both heads should approximate the identity. Either argument
        can be omitted (e.g. a training step that only has a forward-
        direction batch that step) -- the loss is averaged over whichever
        terms are actually provided.
        """
        terms = []
        if pooled_jepa is not None:
            recon = self.vlm_to_jepa(self.jepa_to_vlm(pooled_jepa))
            terms.append(nn.functional.mse_loss(recon, pooled_jepa))
        if pooled_vlm is not None:
            recon = self.jepa_to_vlm(self.vlm_to_jepa(pooled_vlm))
            terms.append(nn.functional.mse_loss(recon, pooled_vlm))
        if not terms:
            return torch.zeros((), device=next(self.parameters()).device)
        return sum(terms) / len(terms)


class ForwardingAdapter(nn.Module):
    """A parameter-free wrapper that calls into one or two already-registered
    submodules elsewhere, without re-registering them as its own children.

    train_joint_coupled.py wires this in on both sides so CrossModalBridge
    stays the SOLE place jepa_to_vlm/vlm_to_jepa's parameters are registered
    -- injector.conditioner.adapter.mlp[0] (forward direction) and
    predictor.guidance_old_adapter/guidance_new_adapter (reverse direction)
    all become instances of this class instead of holding real Linear
    layers, so the optimizer's parameter list is exactly
    injector.parameters() + predictor.parameters() + bridge.parameters()
    with no double-counted tensors (a plain attribute assignment of the
    SAME nn.Linear into two different parents would otherwise register it
    twice, corrupting AdamW's per-parameter moment estimates).

    Reverse-direction use also composes through CortexGuidedVideoPredictor's
    OWN context_adapter -- the same layer it uses to embed its real V-JEPA2
    rollout tokens into its working space -- so guidance produced by the
    bridge lands in the exact same coordinate frame real tokens do, not just
    approximately via a one-time init composition (see
    fit_warmstart_alignment.py's docstring for why composing through
    context_adapter matters: predictor_embed_dim has no independent
    pretrained meaning of its own to fit/train against directly).
    """

    def __init__(self, *modules):
        super().__init__()
        self._modules_hidden = list(modules)  # not registered -- see class docstring

    def forward(self, x):
        for m in self._modules_hidden:
            x = m(x)
        return x
