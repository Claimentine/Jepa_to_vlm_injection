# jepa-vlm-injection

A ThinkJEPA follow-up: instead of a VLM guiding a JEPA world model's video
prediction (ThinkJEPA's direction), this injects a **frozen V-JEPA2 world
model's latent features into a frozen VLM** (Qwen3-VL-2B-Thinking), as a
soft prompt, to test whether that makes the VLM's answers more grounded in
actual video dynamics. Two testbeds:

- **`egodex_probe/`** -- a fast proxy task: 13-way "what task is this
  egocentric clip" classification on EgoDex, used to shake out the
  architecture before committing to the slower/messier VANS pipeline.
- **`vans_qa/`** -- the real target task: "predict what happens next" as a
  2-choice QA over COIN/YouCook2 instructional-video clips (VANS-Data-100K).

## Status (as of the last full-scale run)

| experiment | result |
|---|---|
| EgoDex, temporal-pooling architecture (`v2`) | jepa beats random: micro_acc 0.627 vs 0.520 |
| EgoDex, earlier over-pooled architecture (`v1`) | jepa **loses** to random: 0.440 vs 0.626 -- diagnosed as pooling collapsing the T=64 temporal axis; see `common/jepa_injection_model.py` docstrings |
| VANS QA, demo scale (500 pairs) | jepa loses to both none and random: 0.653 vs 0.800/0.853 |
| VANS QA, full scale (12,032 pairs, 17x demo's training data) | jepa **still** loses, gap gets statistically stronger not weaker (McNemar jepa-vs-random chi2 4.76 -> 20.1) -- "not enough data" is ruled out as the explanation |

The EgoDex positive result and the VANS negative result are in real tension
-- see the bottom of this README for open hypotheses. This is the main
open problem to pick up on whatever cluster runs next.

## Layout

```
common/
  jepa_injection_model.py     JepaPoolerTemporal + SoftPromptAdapterTemporal +
                               LanguageModelInjectionHook -- the injection
                               architecture, shared by both testbeds. Also
                               keeps the earlier "v1" (over-pooled) versions
                               around for comparison; prefer the *Temporal
                               classes for new work.

egodex_probe/                 EgoDex 13-way classification testbed
  task_categories.py
  sample_probe_episodes.py    -> sample which EgoDex episodes to use
  build_classification_split.py -> train/val/test split
  extract_kinematic_signals.py  (deprioritized alternate proxy task, kept for reference)
  run_task_recognition_probe.py + score_task_recognition_probe.py
                               -> zero-shot "does the VLM already know this
                                  task" baseline (it doesn't: 22-29% acc,
                                  and often doesn't follow the answer format)
  train_jepa_injection_probe.py / eval_jepa_injection_probe.py
                               -> the actual injection experiment
  eval_direct_answer_baseline.py
  build_viz.py
  sbatch/                      SLURM launchers (Delta-specific account/partition,
                                edit for Nautilus)

vans_qa/
  score_qa_pilot.py            shared scorer (Wilson CI + paired McNemar test),
                                used by both demo and full_scale
  demo/                        500-pair demo bundle pipeline
    build_qa_pairs.py extract_vjepa_features.py extract_vlm_guidance.py
    train_latent_world_model.py   <- validates ThinkJEPA's ORIGINAL direction
                                      (VLM guides JEPA prediction), on this data
    train_vans_qa_injection.py eval_vans_qa_injection.py  <- the reverse-injection pilot
  full_scale/                  12,032-pair pipeline built from raw COIN/YouCook2
    build_target_list.py download_videos.py            <- yt-dlp bulk download
    step1_coin_fixed.py step1_youcook_fixed.py          <- split into per-step clips
                                                             (fixes real schema bugs in
                                                             VANS's official step1.py,
                                                             see file docstrings)
    build_qa_pairs_full.py
    extract_vjepa_features_full.py extract_vlm_guidance_full.py
    train_qa_full.py eval_qa_full.py
    scan_corrupt_npz.py         <- integrity checker; caught 471 truncated feature
                                    files after a filesystem relocation, see docstring
    sbatch/

third_party/vjepa2_video_utils/   vendored files, see its own README
scripts/patch_thinkjepa.sh        run once after cloning ThinkJEPA (two local
                                   fixes, never upstreamed -- worth a PR)
docs/data_and_checkpoints.md      what external data/models you need and where
                                   from -- nothing large is committed here
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
falls back to its current Delta path as a default if the env var is unset,
so grep for `os.environ.get(` in a given script to see exactly what it
needs.

`sbatch/*.sbatch` files are Delta-specific (`--account`, `--partition
ghx4`) -- rewrite the resource-request headers for Nautilus (this project
was using SLURM; Nautilus is Kubernetes-based, so these need to become pod
specs / Argo workflows rather than a straight port).

## Open problem: why does the same architecture help on EgoDex but hurt on VANS?

Candidates worth checking, in roughly the order I'd check them:

1. **Output-format interference.** In the full-scale VANS eval, the `jepa`
   condition has a 4-5x higher "model didn't produce a parseable answer"
   rate than `none`/`random` (8.0% vs ~1.5%). That's not just "picks the
   wrong option" -- something about the injected tokens may be disrupting
   the model's ability to follow the answer-format instruction at all.
   Worth pulling the actual unparsed generations and reading them.
2. **Task granularity mismatch.** EgoDex is 13-way coarse classification;
   VANS is picking between two long, detailed natural-language captions.
   JEPA's world-model features may carry a coarse "what kind of
   activity/scene" signal that helps a category-level task but is the
   wrong grain for discriminating two similar long captions.
3. **Domain diversity.** VANS spans ~270 COIN+YouCook2 task categories vs
   EgoDex's 13; 12,032 examples is only ~45/category on average. Untested:
   whether performance varies by how much training data a given task
   category actually got.
4. **Clip-length / frame-sampling mismatch.** `read_clip_frames()` resamples
   every clip to a fixed T=64 frames via `np.linspace`, regardless of
   native length. Sampled VANS clip durations: median 9s, but 15% are
   under 4s (frames get heavily duplicated to fill 64 slots -- closer to a
   static image than real motion) and 45% are over 10s up to 144s (64
   frames sampled that sparsely loses fine motion). EgoDex clips are more
   uniform. Untested: does injection quality correlate with clip duration?
