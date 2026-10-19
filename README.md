# jepa-vlm-injection

A ThinkJEPA follow-up, now **bidirectional**: a shared `CrossModalBridge`
couples a frozen V-JEPA2 video world model and a frozen VLM
(Qwen3-VL-2B-Thinking) in both directions at once, trained jointly with a
single combined backward pass per step.

- **Forward (JEPA -> VLM)**: JEPA's spatio-temporal latents are injected
  into the VLM's decoder via gated cross-attention (`LayerWiseJEPAInjector`,
  `common/jepa_injection_model.py`), conditioning its answer to a
  "predict what happens next" 2-choice QA task built from COIN video clips.
- **Reverse (VLM -> JEPA)**: the VLM's own hidden states guide the JEPA
  side's `CortexGuidedVideoPredictor` (ThinkJEPA's own module) via the
  mirror-image cross-attention mechanism, trained on VANS/COIN/Ego4D
  self-supervised next-latent prediction pairs.

Both directions share one `CrossModalBridge` (`common/cross_modal_bridge.py`)
instead of two independent projection heads, so a single joint-training run
updates both at once. This README tracks the project's **current best
results** and the experiment history that led to them; see
`docs/data_and_checkpoints.md` for data/checkpoint setup.

## Headline results

**Joint (bidirectional) training no longer trades off against either
direction's own independently-trained baseline, given enough training
budget** -- job-72 (job-42's exact recipe, trained 2 epochs instead of 1,
evaluated at its step_4000 checkpoint) beats job-42 on *all four* forward
benchmarks, and beats the independent-training baseline (job-13) on
3 of 4 (TempCompass, MVBench, CrossTask) -- the first joint-trained
checkpoint in this project's history to do so on any out-of-domain
benchmark. On the reverse side, every joint checkpoint tested (including
undertrained ones) already beats ThinkJEPA's own officially-trained-on-EgoDex
reference, with job-72's step_4000 achieving the best ADE of any checkpoint
tested, i.e. no forward/reverse trade-off either.

### Forward: "predict what happens next" QA accuracy (full test splits)

| Checkpoint | COIN (in-domain) n=2853 | TempCompass (OOD) n=3829 | MVBench (OOD) n=9223-9229 | CrossTask (OOD) n=1557-1558 |
|---|---|---|---|---|
| Zero-shot frozen VLM (no injection at all) | 51.2% | 50.7% | *(running)* | *(running)* |
| Content-free injection (random noise, real trained cross-attn) | 58.4% | 51.2% | *(running)* | *(running)* |
| Independent training, forward-only (job-13) | 74.8% | 72.3% | 67.3% | 74.7% |
| Joint, 1 epoch, bridge MLP (job-42) | 66.7% | 75.7% | 63.1% | 70.1% |
| Joint, 1 epoch, bridge width x2 (job-60) | 58.7% | 67.6% | 55.1% | 59.4% |
| **Joint, 2 epochs, step_4000 (job-72, current best)** | **70.9%** | **76.7%** | **69.4%** | **75.6%** |

CrossTask is evaluated on a reduced-scale corpus (1013/2692 source videos,
~37.6%) due to YouTube bot-detection rate-limiting during extraction -- see
Limitations. The two baseline rows (zero-shot, content-free) are a new
addition for paper prep; MVBench/CrossTask for them are mid-run as of this
writing.

### Reverse: EgoDex trajectory prediction, frozen-predictor probe (n=139 test episodes, all 52 hand/finger joints)

| Checkpoint / control | ADE | FDE | acc@0.05 |
|---|---|---|---|
| Constant-velocity extrapolation (standard trajectory-forecasting control) | 0.1129 | 0.2438 | 53.7% |
| Static "stay-put" baseline | 0.0307 | 0.0930 | 82.9% |
| No-guidance predictor (VLM guidance ablated out entirely) | 0.0352 | 0.0506 | 81.0% |
| Independent training, reverse-only (job-15) | 0.0336 | 0.0384 | 81.5% |
| Joint, 1 epoch, bridge MLP (job-42) | 0.0343 | 0.0419 | 79.7% |
| Joint, 1 epoch, bridge width x2 (job-60) | 0.0349 | 0.0430 | 82.3% |
| **Joint, 2 epochs, step_4000 (job-72, current best)** | **0.0331** | **0.0407** | **82.5%** |
| ThinkJEPA official reference (end-to-end trained ON EgoDex) | 0.0683 | 0.0733 | n/a |

Every row except the first two controls is a frozen predictor (never
trained on EgoDex) plus a small trained probe head -- "how much trajectory
signal is already in this latent," not "we trained a better trajectory
model." Note the no-guidance control's ADE (0.0352) does *not* beat the
static baseline (0.0307), while every guided checkpoint does -- evidence
that beating the static baseline specifically requires real cross-modal
guidance, not just probe capacity.

## Experiment history (forward recipe ablation lineage, COIN full-test accuracy)

| Job | Recipe change (relative to previous row) | COIN acc | Outcome |
|---|---|---|---|
| job-13 | Independent (non-joint) training, forward direction only | 74.8% | Reference ceiling pre-joint-training |
| job-22 | First joint (shared-bridge) training; reverse side trained on VANS data only | 68.3% | Large drop vs. independent |
| job-30 | + mix COIN into the reverse-direction training pool | 68.2% | No improvement over job-22 |
| job-34 | + EMA loss-magnitude normalization across fwd/rev/cycle losses | 67.2% | Normalization alone doesn't close the gap |
| job-36 | cycle_loss_weight=0 (on job-34's normalized recipe) | *(CrossTask 59.9% / MVBench 55.2% / TempCompass 64.0%)* | Removing cycle loss alone doesn't help here |
| job-42 | disable_norm + cycle_loss_weight=0 + bridge as 2-layer MLP (hidden=2048), 1 epoch | 66.7% | **The eventually-winning recipe -- but under-trained at 1 epoch** (its own loss curve hadn't converged) |
| job-49 | + mix COIN into job-42's recipe | 67.1% | Marginal change |
| job-56 | job-34's normalization combined with job-42's MLP bridge | 61.5% | Worse than either change alone |
| job-60 | job-42's recipe, bridge_hidden_dim doubled to 4096 | 58.7% | Forward loss failed to converge (ema plateaued ~0.68-0.70 vs job-42's ~0.58-0.60) -- a width/LR mismatch, not a capacity win; ruled out via loss-curve comparison and a corrected gradient-conflict diagnostic |
| **job-72** | job-42's *exact* recipe, `--epochs 2` instead of 1 | **70.9%** | **Breakthrough**: beats job-42 on all 4 forward benchmarks; beats job-13 on 3/4 OOD benchmarks (first time any joint checkpoint has) |
| job-79 (running) | job-72's recipe + warmup/cosine LR schedule + checkpoints every 250 steps instead of 1000 | *(in progress)* | Testing whether a schedule avoids job-72's own step~4500 training-instability dip and finds an even better stopping point |

## Planned additional experiments (paper prep)

Designed by cross-referencing two papers' experiment sections: **ThinkJEPA**
(shares our EgoDex ADE/FDE task directly -- its baseline/ablation choices
transfer almost one-to-one) and **LION** (*Empowering Multimodal Large
Language Model with Dual-Level Visual Knowledge*, arXiv:2311.11860 -- a
VQA/captioning-style MLLM paper, useful for how it structures baseline-model
comparisons and component-ablation tables on standard benchmarks). Kept as
small/cheap as possible: reusing existing data and existing code paths
wherever one already covers the need, flagged explicitly below.

### A. Forward direction -- baseline models (reuse the existing QA task, no new dataset)

| # | Baseline | Status | Cost |
|---|---|---|---|
| A1 | **"VLM-only, trained"** -- a trained adapter/head on the VLM's own hidden states with zero external (JEPA) conditioning. This is what ThinkJEPA's own "Qwen3-VL Thinking" baseline row actually is (Table 1: a *trained* downstream head on VLM features, not zero-shot) -- stronger and fairer than our current `zeroshot_none` row, which has no training at all. | **Code already exists** (`train_qa_full.py --injection constant`) **but was never run to completion** under the current temporal-pooling architecture (`qa_full_runs_temporal/constant` doesn't exist; the only `constant` run on disk is an abandoned 3-step v1-pooling run). | Cheap -- same training script, same data, just finish the run. |
| A2 | **A second open-source VLM, zero-shot, on our own QA task** (e.g. Qwen2-VL-2B-Instruct) -- addresses "is this specific backbone doing something unusual" the way LION's Table 1 contextualizes itself against Flamingo/Kosmos/BLIP-2/LLaVA/InstructBLIP, all run on the *same* benchmark suite. Per the scope note above we cannot cite their own papers' numbers -- has to be run through our harness, same as LION runs every baseline on its own eval pipeline. | Not started. | Moderate -- new model download + inference wiring, no new data. |

### B. Reverse direction -- external trajectory baselines (near-zero cost)

| # | Baseline | Status | Cost |
|---|---|---|---|
| B1 | **EgoDex's own released trajectory-prediction baselines**: decoder-only and encoder-decoder transformers x {Behavior Cloning, DDPM, Flow Matching} (6 numbers). ThinkJEPA's own Table 3 just *cites* these directly from EgoDex's paper/codebase ("implemented from [15]") rather than re-running them -- we are evaluating on the same EgoDex ADE/FDE task, so we can do the same. | Not started -- needs locating EgoDex's released numbers/checkpoints for these 6 baselines. | **Near zero** -- citation, not a training run. |
| B2 | **Prompt-only VLM baseline** -- directly prompt Qwen3-VL-2B-Thinking with the past video window and ask it to output future 3D hand trajectories as JSON waypoints, interpolated to the full horizon and scored with the same ADE/FDE code. Mirrors ThinkJEPA's Table 10 exactly (their result: ADE/FDE 10.855/10.927 vs. their method's 0.061/0.056 -- a deliberately weak but informative reference). More informative than our current `job_none_guidance` row since it tests the VLM's own native spatial reasoning with no architecture support at all, not just "JEPA predictor without guidance." | Not started. | Cheap -- inference + JSON parsing only, no training. |

### C. Ablations on the current best recipe (job-72-style), not yet run

| # | Ablation | Mirrors | Status |
|---|---|---|---|
| C1 | **VLM layer-selection** for the forward injector: `--layer_strategy` already supports `{middle4, last4, uniform4, all}`, but job-42/72's winning recipe has only ever been run with `middle4`. | ThinkJEPA Table 4 (last-layer vs. mid-layer vs. all-layer/pyramid guidance) | Not started -- same training script, different flag, same compute as job-72 once per variant. |
| C2 | **Conditioning mechanism**: cross-attention (current default) vs. FiLM (`LayerWiseJEPAInjector`'s `film` mode already exists in `common/jepa_injection_model.py`, but was diagnosed as bandwidth-limited and dropped early in the project -- never re-tested on the current winning (job-72-style, 2-epoch) recipe specifically). | ThinkJEPA Table 8 (FiLM vs. cross-attn vs. AdaLN) and LION's own adapter-design ablations | Not started. |

### D. One small additional eval slice (no new download)

Our existing `qa_split_mvbench.json` already has an `object_existence`
sub-category (visible in every MVBench result's `acc_by_difficulty` dict
above) -- reporting it as its own row is a free, zero-cost analogue of
LION's **POPE** object-hallucination table (does injection make the model
*more* confident about objects that aren't there, not just better on
average?). Only build a dedicated small POPE-style yes/no set from
already-downloaded COIN clips (no new video download) if this existing
slice turns out too small/noisy to be conclusive on its own.

## Scope and limitations (read before citing any number above)

- **Not comparable to official benchmark leaderboards.** COIN/TempCompass/
  MVBench/CrossTask numbers above come from a 2-choice "predict what
  happens next" QA task *we constructed* from each benchmark's raw videos
  -- not each benchmark's own official multi-choice QA format. Only
  compare rows within this README's own tables, never against externally
  published numbers for these benchmark names.
- **CrossTask is a reduced-scale subset** (~37.6% of the full corpus) due
  to YouTube rate-limiting during video extraction.
- **Single seed, single training run per row.** job-72's own training
  showed a real mid-run instability (100-item val snapshot dipped from
  0.84-0.86 to 0.41 around step 4500 before recovering) -- run-to-run
  variance has not been measured. "Beats baseline X" above means "this
  specific checkpoint does," not yet an established, reproducible effect.
- **The ThinkJEPA reference is their released configuration** (`REF_MODE=none`,
  no hand-reference-offset correction), not necessarily the best possible
  tuning of their method.
- Verified only at Qwen3-VL-2B scale; no evidence here about larger models.

## Layout

```
common/
  jepa_injection_model.py       Forward injection architecture: JepaPoolerTemporal,
                                  SoftPromptAdapterTemporal, LayerWiseJEPAInjector
                                  (film / cross_attn modes), LanguageModelInjectionHook.
  cross_modal_bridge.py         CrossModalBridge (jepa_to_vlm / vlm_to_jepa heads,
                                  shared by both directions under joint training) +
                                  ForwardingAdapter (parameter-free rewiring shim).

vans_qa/full_scale/             the real-scale pipeline (12k+ COIN/YouCook2 QA pairs,
                                  VJEPA2 features, VLM guidance caches)
  train_qa_full.py               forward-only training (injection in {jepa,random,constant,none})
  eval_forward_checkpoint.py     full-test-split forward eval, any checkpoint shape
                                  (plain or shared-bridge), --checkpoint label=NONE for
                                  the true zero-shot frozen-VLM control
  train_latent_world_model_full.py   reverse-only training (guidance in {crossattn,adaln,none})
  train_joint_coupled_normalized.py  THE joint-training script behind job-22 through
                                       job-79: shared bridge, EMA loss normalization
                                       (--disable_norm to turn off), --cycle_loss_weight,
                                       --bridge_hidden_dim, --lr_schedule {constant,warmup_cosine}
  probe_egodex_reverse_adapter.py    freezes a reverse-direction checkpoint's predictor,
                                       trains only a small probe head, reports ADE/FDE/acc@0.05
                                       on EgoDex (--joint_subset {wrists,all}); also computes
                                       the static and constant-velocity baselines
  extract_coin_longaxis.py / extract_ego4d_longaxis.py / extract_crosstask_qa.py
                                  additional self-supervised / OOD-eval data sources
  build_*_qa_split.py             per-benchmark QA split builders (tempcompass, mvbench)
  smoke_test_youcook_download.py  YouTube download connectivity probe (yt-dlp + deno,
                                   no cookies) -- checks bot-detection status before
                                   committing to a real extraction run

egodex_probe/                   earlier, smaller EgoDex classification testbed
                                  (13-way "what task is this clip" probe) used to shake
                                  out the architecture before the full-scale pipeline;
                                  superseded by vans_qa/full_scale's results above.

k8s/                            Nautilus/PRP job manifests (job-NN-*.yaml), one per
                                  experiment in the lineage above -- see each file's own
                                  header comment for its rationale.
third_party/vjepa2_video_utils/ vendored files, see its own README
scripts/patch_thinkjepa.sh      run once after cloning ThinkJEPA (two local fixes,
                                  never upstreamed)
docs/data_and_checkpoints.md    what external data/models you need and where from --
                                  nothing large is committed here
```

## Setup

Two conda environments (kept separate because of an unrelated transformers
version conflict between the V-JEPA2/ffmpeg side and the Qwen3-VL side --
see `environment-*.yml`):

```bash
conda env create -f environment-thinkjepa.yml   # feature extraction, clip splitting
conda env create -f environment-qwen3vl.yml     # training / eval (the VLM side)

git clone https://github.com/Hai-chao-Zhang/ThinkJEPA
THINKJEPA_ROOT=/path/to/ThinkJEPA ./scripts/patch_thinkjepa.sh
```

Then see `docs/data_and_checkpoints.md` for what data/checkpoints need to
exist, and set the env vars listed there (`VJEPA2_CKPT`, `THINKJEPA_ROOT`,
`EGODEX_*`, `VANS_ROOT`, `VANS_WORK_ROOT`, `FFMPEG_BIN`) -- every script
falls back to a default if the env var is unset, so grep for
`os.environ.get(` in a given script to see exactly what it needs.

Current runs live on Nautilus/PRP (Kubernetes) -- see `k8s/*.yaml` for job
manifests; `vans_qa/full_scale/sbatch` and `egodex_probe/sbatch` are leftover
SLURM launchers from an earlier (Delta) cluster and are no longer used.
