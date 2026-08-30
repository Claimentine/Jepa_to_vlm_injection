"""
Full-scale (~12K matched QA pairs, vs the demo's 499) version of
train_vans_qa_injection.py. Same task (2-choice "what happens next"),
same temporal-pooling injection architecture validated on EgoDex, same
training recipe -- only the data source changed: qa_split_full.json +
vjepa_cache_full/ + vlm_guidance_cache_full/ instead of the 500-pair demo
bundle.

Run extract_vjepa_features_full.py and extract_vlm_guidance_full.py first.
"""
import argparse
import glob
import json
import multiprocessing as mp
import os
import random
import sys
import time
import traceback

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from common.jepa_injection_model import (  # noqa: E402
    DecoderLayerInjectionHook,
    JepaInjector,
    LanguageModelInjectionHook,
    LayerWiseJEPAInjector,
    prepend_placeholder_tokens,
    resolve_decoder_layers,
    select_layer_indices,
)

BASE = os.environ.get("VANS_ROOT", "/projects/bhay/william/ruixin/vans_world_model")
WORK_BASE = os.environ.get("VANS_WORK_ROOT", "/work/nvme/bdqf/yli8/vans_raw_data")  # relocated off the near-full /projects/bhay allocation
MODEL_ID = "Qwen/Qwen3-VL-2B-Thinking"
VJEPA_CACHE = os.path.join(WORK_BASE, "vjepa_cache_full")
VLM_CACHE = os.path.join(WORK_BASE, "vlm_guidance_cache_full")
CLIPS_ROOT = os.path.join(WORK_BASE, "clips")


def safe_name(clip_path):
    rel = os.path.relpath(clip_path, CLIPS_ROOT)
    return rel.replace("/", "__").replace(".mp4", "")


def find_vlm_npz(clip_path):
    name = safe_name(clip_path)
    matches = glob.glob(os.path.join(VLM_CACHE, f"{name}_*.npz"))
    return matches[0] if matches else None


def load_vjepa_in_feats(clip_path):
    name = safe_name(clip_path)
    p = os.path.join(VJEPA_CACHE, f"{name}.npz")
    d = np.load(p)
    return d["feats"].astype(np.float32)


def has_features(item):
    # This script's injection direction (JepaInjector) only ever loads
    # vjepa_feats -- VLM_CACHE/find_vlm_npz above is dead code for it, kept
    # only because a sibling script (train_latent_world_model.py, the
    # opposite VLM-guides-JEPA direction) needs both caches populated for
    # the *same* item set to stay comparable. On Nautilus, vlm_guidance_cache_full
    # never got populated: extract_vlm_guidance_full.py shells out to
    # ThinkJEPA's cache_train/qwen3_cache_extractor.py, which doesn't exist
    # in the current public ThinkJEPA release (patch_thinkjepa.sh already
    # guards for this with `if [ -f "$EXTRACTOR" ]`). Gating on vlm_ok here
    # would zero out train/val entirely, so it's dropped -- restore it once
    # that extractor is available again.
    return os.path.exists(os.path.join(VJEPA_CACHE, f"{safe_name(item['in_clip'])}.npz"))


def build_option_block(item, rng):
    options = [item["correct_caption"], item["distractor_caption"]]
    order = [0, 1]
    rng.shuffle(order)
    letters = ["A", "B"]
    letter_to_is_correct = {}
    lines = []
    for i, letter in enumerate(letters):
        opt_idx = order[i]
        lines.append(f"{letter}. {options[opt_idx]}")
        letter_to_is_correct[letter] = (opt_idx == 0)
    correct_letter = "A" if letter_to_is_correct["A"] else "B"
    return lines, correct_letter


def build_prompt_text(question, option_lines):
    return (
        "You are shown footage filmed from a first-person viewpoint.\n"
        f"{question}\n\n"
        + "\n".join(option_lines)
        + "\n\nAnswer with only the option letter:"
    )


def clamp_frame_count(mp4_path, requested):
    import cv2
    cap = cv2.VideoCapture(mp4_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if total <= 0:
        return requested
    nf = min(requested, total)
    nf = max(2, nf)
    return (nf // 2) * 2


DECODE_TIMEOUT_S = 60

# By the time build_inputs() is first called, the parent process has already
# initialized a CUDA context (the frozen VLM is loaded onto `device` earlier
# in main()). fork() -- the default start method on Linux -- after CUDA init
# is documented as unsafe by NVIDIA and manifests here as a near-universal
# hang in the child: every single distinct clip timed out at exactly
# DECODE_TIMEOUT_S with 0% GPU util and near-zero CPU, which isn't what a
# slow *decode* looks like (that would vary per clip and burn CPU) -- it's
# what a child stuck at startup on inherited, fork-unsafe CUDA driver state
# looks like. spawn re-execs a fresh interpreter instead of forking, so the
# child never touches the parent's CUDA context.
_MP_CTX = mp.get_context("spawn")


def _to_ipc_safe(obj):
    # torch registers a custom multiprocessing reducer for every Tensor
    # (even over a plain mp.Pipe, not just torch.multiprocessing.Queue) that
    # ships it via a shared-memory segment plus a file-descriptor handoff
    # through a side-channel `multiprocessing.resource_sharer` Unix socket,
    # instead of just writing the bytes into the pipe. That handoff only
    # works while the sending (child) process is still alive to serve it --
    # nothing here synchronizes "parent finished unpickling" with "child may
    # now exit", so a child that tears down right after conn.send() races the
    # parent's parent_conn.recv(), which sometimes loses: confirmed in
    # practice via ConnectionResetError / FileNotFoundError inside
    # torch/multiprocessing/reductions.py's rebuild_storage_fd, accounting
    # for close to a third of all "corrupt clip" skips before this fix.
    # Converting to plain numpy sidesteps torch's reducer entirely -- numpy
    # arrays just get pickled inline into the pipe's normal byte stream.
    if isinstance(obj, torch.Tensor):
        return obj.numpy()
    if isinstance(obj, dict):
        return {k: _to_ipc_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_ipc_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_to_ipc_safe(v) for v in obj)
    return obj


def _from_ipc_safe(obj):
    if isinstance(obj, np.ndarray):
        return torch.from_numpy(obj)
    if isinstance(obj, dict):
        return {k: _from_ipc_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_from_ipc_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_from_ipc_safe(v) for v in obj)
    return obj


def _decode_video_worker(conn, abs_path, requested_frames, prompt_text):
    # Runs in a throwaway child process (see build_inputs) so a hung native
    # call on a badly corrupted clip can be killed from outside -- both
    # cv2.VideoCapture and decord.VideoReader are native/ffmpeg extensions,
    # and a call that never returns to Python can't be interrupted by a
    # signal or a timeout in the same process (confirmed in practice: a
    # training job sat stuck 6+ hours at 0% GPU util on one clip with the
    # per-item try/except doing nothing, since no exception was ever raised
    # to catch). clamp_frame_count used to run un-wrapped in the parent
    # before this worker was spawned -- also confirmed in practice, via a
    # separate ~15min "Stream timeout triggered" hang from OpenCV's own
    # internal ffmpeg probe on a different corrupted clip -- so it's called
    # in here now, covered by the same DECODE_TIMEOUT_S deadline.
    try:
        nframes = clamp_frame_count(abs_path, requested_frames)
        video_content = {
            "type": "video", "video": f"file://{abs_path}",
            "resized_height": 256, "resized_width": 256, "nframes": nframes,
        }
        messages = [{"role": "user", "content": [video_content, {"type": "text", "text": prompt_text}]}]
        images, videos, video_kwargs = process_vision_info(
            messages, image_patch_size=16, return_video_kwargs=True, return_video_metadata=True,
        )
        video_metadatas = None
        if videos is not None:
            videos, video_metadatas = zip(*videos)
            videos, video_metadatas = list(videos), list(video_metadatas)
        conn.send(("ok", _to_ipc_safe((nframes, images, videos, video_metadatas, video_kwargs))))
    except Exception as e:
        conn.send(("err", e))
    finally:
        conn.close()


def build_inputs(processor, in_clip_path, prompt_text, max_frames=32):
    abs_path = os.path.abspath(in_clip_path)

    parent_conn, child_conn = _MP_CTX.Pipe(duplex=False)
    proc = _MP_CTX.Process(target=_decode_video_worker, args=(child_conn, abs_path, max_frames, prompt_text))
    proc.start()
    child_conn.close()  # parent's copy of the write end; child still holds its own
    if parent_conn.poll(DECODE_TIMEOUT_S):
        status, payload = parent_conn.recv()
    else:
        status, payload = "timeout", None
    parent_conn.close()
    proc.join(5)
    if proc.is_alive():
        proc.terminate()
        proc.join()
    if status == "timeout":
        raise TimeoutError(f"video decode exceeded {DECODE_TIMEOUT_S}s on {in_clip_path}")
    if status == "err":
        raise payload
    nframes, images, videos, video_metadatas, video_kwargs = _from_ipc_safe(payload)

    # Rebuilt here (not sent back through the pipe) since it's cheap and pure
    # text -- no need to serialize it across the process boundary twice.
    video_content = {
        "type": "video", "video": f"file://{abs_path}",
        "resized_height": 256, "resized_width": 256, "nframes": nframes,
    }
    messages = [{"role": "user", "content": [video_content, {"type": "text", "text": prompt_text}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=text, images=images, videos=videos, video_metadata=video_metadatas,
        return_tensors="pt", do_resize=False, **(video_kwargs or {}),
    )
    return inputs


def append_answer_tokens(tokenizer, input_ids, attention_mask, letter):
    """Append a candidate continuation and return its token positions.

    The leading space is intentional: Qwen's tokenizer represents answer
    letters differently at a word boundary.  Scoring the complete candidate
    sequence also remains correct if a future tokenizer splits a letter.
    """
    # .to(input_ids.device): the tokenizer always returns a fresh CPU tensor,
    # but input_ids is already on the model's device by the time this is
    # called -- without this cast, torch.cat below raises a CUDA/CPU device
    # mismatch (confirmed on a real GPU run, not just by inspection).
    suffix_ids = tokenizer(f" {letter}", add_special_tokens=False, return_tensors="pt")["input_ids"].to(input_ids.device)
    full_ids = torch.cat([input_ids, suffix_ids], dim=1)
    full_mask = torch.cat([attention_mask, torch.ones_like(suffix_ids)], dim=1)
    return full_ids, full_mask, suffix_ids


def candidate_logprob(model, model_inputs, tokenizer, letter):
    """Log P(letter | video, question, options), summed over answer tokens."""
    prefix_len = model_inputs["input_ids"].shape[1]
    input_ids, attn_mask, suffix_ids = append_answer_tokens(
        tokenizer, model_inputs["input_ids"], model_inputs["attention_mask"], letter
    )
    call_inputs = dict(model_inputs)
    call_inputs["input_ids"] = input_ids
    call_inputs["attention_mask"] = attn_mask
    out = model(**call_inputs)
    positions = torch.arange(prefix_len - 1, prefix_len - 1 + suffix_ids.shape[1], device=out.logits.device)
    token_logprobs = out.logits[0, positions].float().log_softmax(dim=-1)
    targets = suffix_ids[0].to(out.logits.device)
    return token_logprobs.gather(1, targets.unsqueeze(1)).sum()


def prepare_model_inputs(inputs, injection_site, n_tokens, placeholder_id):
    model_inputs = {k: v for k, v in inputs.items() if k not in ("input_ids", "attention_mask")}
    input_ids, attn_mask = inputs["input_ids"], inputs["attention_mask"]
    if injection_site == "prefix":
        input_ids, attn_mask = prepend_placeholder_tokens(input_ids, attn_mask, n_tokens, placeholder_id)
    model_inputs["input_ids"] = input_ids
    model_inputs["attention_mask"] = attn_mask
    return model_inputs


class AccTally:
    """Overall + per-`difficulty` accuracy (qa_split_temporal.json tags each
    val/test row "easy" or "hard"; qa_split_full.json has no such field, so
    by_difficulty just stays empty and summary_str() degrades to a bare
    overall number -- old-split runs are unaffected).
    """

    def __init__(self):
        self.n_correct = 0
        self.n_total = 0
        self.by_difficulty = {}

    def add(self, item, is_correct):
        self.n_correct += int(is_correct)
        self.n_total += 1
        diff = item.get("difficulty")
        if diff is not None:
            c = self.by_difficulty.setdefault(diff, [0, 0])
            c[0] += int(is_correct)
            c[1] += 1

    @property
    def acc(self):
        return self.n_correct / max(self.n_total, 1)

    @property
    def acc_by_difficulty(self):
        return {k: v[0] / max(v[1], 1) for k, v in self.by_difficulty.items()}

    def summary_str(self):
        s = f"{self.acc:.4f}"
        if self.by_difficulty:
            parts = " ".join(f"{k}={v:.4f}" for k, v in sorted(self.acc_by_difficulty.items()))
            s += f" ({parts})"
        return s


def evaluate_logprob(model, processor, hook, injector, items, injection_site, n_tokens, placeholder_id, device, rng):
    model.eval()
    # Without this, JepaPoolerTemporal's internal attention dropout (p=0.1)
    # stays active during periodic in-training validation, adding noise to
    # the val_acc used to pick best.pt.
    injector.eval()
    tally = AccTally()
    seen_exc_types = set()
    with torch.no_grad():
        for item in items:
            try:
                feats_np = load_vjepa_in_feats(item["in_clip"])
                option_lines, correct_letter = build_option_block(item, rng)
                prompt_text = build_prompt_text(item["question"], option_lines)
                inputs = build_inputs(processor, item["in_clip"], prompt_text)
                model_inputs = prepare_model_inputs(inputs, injection_site, n_tokens, placeholder_id)
                model_inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in model_inputs.items()}

                feats = torch.from_numpy(feats_np).unsqueeze(0).to(device)
                condition = injector(feats) if injection_site == "prefix" else injector.condition(feats)
                hook.set(condition)
                scores = torch.stack([
                    candidate_logprob(model, model_inputs, processor.tokenizer, "A"),
                    candidate_logprob(model, model_inputs, processor.tokenizer, "B"),
                ])
                hook.clear()
                tally.add(item, ("A" if scores[0] > scores[1] else "B") == correct_letter)
            except Exception as e:
                # matches the training loop's per-item try/except: a small fraction of
                # clips are corrupt/too-short (bad feature cache or torchvision frame-count
                # errors) -- skip rather than crash a multi-hour job on one item
                exc_name = type(e).__name__
                print(f"[WARN] eval: skipping item {item.get('pid')}: {exc_name}: {e}", flush=True)
                # Full traceback once per distinct exception type -- enough to
                # diagnose a systematic bug (e.g. a device mismatch) without
                # spamming the same trace for every corrupt-cache item.
                if exc_name not in seen_exc_types:
                    seen_exc_types.add(exc_name)
                    traceback.print_exc()
    if tally.n_total == 0 and items:
        print(f"[WARN] eval: 0/{len(items)} items succeeded -- val_acc is meaningless, "
              f"not a real 0.0 score", flush=True)
    model.train()
    injector.train()
    return tally


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default=os.path.join(BASE, "raw_data/qa_split_full.json"))
    ap.add_argument("--injection", choices=["jepa", "random", "constant", "none"], required=True,
                    help="condition content; constant is a content-free learned control, "
                         "none is a frozen zero-shot baseline with no injector at all")
    ap.add_argument("--injection_site", choices=["prefix", "film", "cross_attn"], default="prefix")
    ap.add_argument("--layer_strategy", choices=["middle4", "last4", "uniform4", "all"], default="middle4",
                    help="selected decoder layers for film/cross_attn")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--val_every_steps", type=int, default=500)
    ap.add_argument("--save_every_steps", type=int, default=1000,
                    help="periodic checkpoint cadence, independent of best.pt (0 disables)")
    ap.add_argument("--max_val_items", type=int, default=200, help="cap for periodic in-training eval")
    ap.add_argument("--max_train_items", type=int, default=None, help="smoke-test cap")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default=os.path.join(BASE, "raw_data/qa_full_runs"))
    args = ap.parse_args()

    n_tokens = 64
    os.environ.setdefault("FORCE_QWENVL_VIDEO_READER", "decord")
    run_dir = os.path.join(args.out_dir, args.injection)
    os.makedirs(run_dir, exist_ok=True)

    with open(args.split) as f:
        split = json.load(f)
    train_items = [it for it in split["train"] if has_features(it)]
    val_items = [it for it in split["val"] if has_features(it)]
    print(f"[INFO] usable (features present): train={len(train_items)} val={len(val_items)}")
    if args.max_train_items:
        train_items = train_items[: args.max_train_items]
    val_items = val_items[: args.max_val_items]

    rng = random.Random(args.seed)

    if args.injection == "none":
        # Blank control: no injector, no hook, no trainable params at all --
        # measures the frozen VLM's zero-shot accuracy on this task. Deliberately
        # bypasses JepaInjector entirely (unlike "constant", which still routes
        # through it with a learned-but-content-free tensor) so this condition
        # can't inherit the injector's shape assumptions (see is_valid_npz's
        # note in extract_vjepa_features_full.py about the 32-vs-64-frame cache
        # mismatch that broke "constant"). There's nothing to train, so this
        # is a single zero-shot pass over val_items, not an epoch loop.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, dtype="auto").to(device)
        processor = AutoProcessor.from_pretrained(MODEL_ID)
        model.requires_grad_(False)
        model.eval()
        print("[INFO] injection=none (frozen zero-shot baseline, no trainable params)")

        tally = AccTally()
        seen_exc_types = set()
        log_f = open(os.path.join(run_dir, "train_log.jsonl"), "w")
        with torch.no_grad():
            for item in val_items:
                try:
                    option_lines, correct_letter = build_option_block(item, rng)
                    prompt_text = build_prompt_text(item["question"], option_lines)
                    inputs = build_inputs(processor, item["in_clip"], prompt_text)
                    model_inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inputs.items()}
                    scores = torch.stack([
                        candidate_logprob(model, model_inputs, processor.tokenizer, "A"),
                        candidate_logprob(model, model_inputs, processor.tokenizer, "B"),
                    ])
                    tally.add(item, ("A" if scores[0] > scores[1] else "B") == correct_letter)
                except Exception as e:
                    exc_name = type(e).__name__
                    print(f"[WARN] skipping item {item.get('pid')}: {exc_name}: {e}")
                    if exc_name not in seen_exc_types:
                        seen_exc_types.add(exc_name)
                        traceback.print_exc()
        log_f.write(json.dumps({
            "step": 0, "val_acc": tally.acc, "val_acc_by_difficulty": tally.acc_by_difficulty,
        }) + "\n")
        log_f.close()
        print(f"[DONE] injection=none  val_acc(logprob)={tally.summary_str()}  items_scored={tally.n_total}/{len(val_items)}")
        return

    print(f"[INFO] loading {MODEL_ID} (frozen) ...")
    # device_map="auto" can silently CPU-offload a handful of decoder layers
    # when the node's free VRAM is tight. That's harmless for a plain
    # generate() call, but DecoderLayerInjectionHook hooks specific decoder
    # layers directly and combines their hidden_states with this script's own
    # (single-device) injector params -- an offloaded layer then raises a
    # CPU/CUDA device-mismatch RuntimeError on every single forward pass
    # (confirmed via kubectl logs on the film-injection smoke test: 100% of
    # items failed this way, so best_val_acc's -1.0 sentinel was never
    # overwritten). Pin everything to one device instead; a real OOM here is
    # a far more diagnosable failure than a silent partial offload.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, dtype="auto").to(device)
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model.requires_grad_(False)
    model.eval()
    hidden_size = model.config.text_config.hidden_size

    video_token_id = model.config.video_token_id
    image_token_id = model.config.image_token_id
    placeholder_id = processor.tokenizer.pad_token_id
    assert placeholder_id not in (video_token_id, image_token_id)

    if args.injection_site == "prefix":
        injector = JepaInjector(
            jepa_dim=1024, hidden_size=hidden_size, mode=args.injection, seed=args.seed, pooling="temporal",
        ).to(device)
        hook = LanguageModelInjectionHook(model, n_tokens)
        layer_indices = []
    else:
        layer_indices = select_layer_indices(len(resolve_decoder_layers(model)), args.layer_strategy)
        injector = LayerWiseJEPAInjector(
            hidden_size=hidden_size, layer_indices=layer_indices, condition_mode=args.injection,
            seed=args.seed, mode=args.injection_site,
        ).to(device)
        hook = DecoderLayerInjectionHook(model, injector)
    opt = torch.optim.AdamW(injector.parameters(), lr=args.lr)
    print(f"[INFO] injection={args.injection} site={args.injection_site} layers={layer_indices} "
          f"trainable_params={sum(p.numel() for p in injector.parameters()):,}")

    step = 0
    n_skipped = 0
    seen_exc_types = set()
    best_val_acc = -1.0
    log_f = open(os.path.join(run_dir, "train_log.jsonl"), "w")

    # Nautilus/PRP federates GPUs from dozens of campuses onto one cluster
    # backed by a single shared CephFS -- a node's path to that storage can
    # be fine or badly congested/distant depending on which campus it's on,
    # and that's invisible from the GPU spec alone. A slow path shows up here
    # as the GPU sitting mostly idle while every item's video decode /
    # feature read crawls -- confirmed in practice via a run stuck at
    # ~70s/step (vs. ~20-35s/step on a healthy node) for 13+ hours before
    # anyone noticed. Surface that within the first handful of steps instead
    # of leaving it to be discovered many hours later.
    SLOW_NODE_SAMPLE_SIZE = 20
    SLOW_NODE_THRESHOLD_S = 45.0
    step_times = []
    slow_node_checked = False

    for epoch in range(args.epochs):
        rng.shuffle(train_items)
        for item in train_items:
            item_start = time.time()
            try:
                option_lines, correct_letter = build_option_block(item, rng)
                prompt_text = build_prompt_text(item["question"], option_lines)
                inputs = build_inputs(processor, item["in_clip"], prompt_text)

                model_inputs = prepare_model_inputs(inputs, args.injection_site, n_tokens, placeholder_id)
                model_inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in model_inputs.items()}

                feats = torch.from_numpy(load_vjepa_in_feats(item["in_clip"])).unsqueeze(0).to(device)
                condition = injector(feats) if args.injection_site == "prefix" else injector.condition(feats)
                hook.set(condition)
                scores = torch.stack([
                    candidate_logprob(model, model_inputs, processor.tokenizer, "A"),
                    candidate_logprob(model, model_inputs, processor.tokenizer, "B"),
                ]).unsqueeze(0)
                hook.clear()
                target = torch.tensor([0 if correct_letter == "A" else 1], device=device)
                loss = nn.functional.cross_entropy(scores, target)
                opt.zero_grad()
                loss.backward()
                opt.step()
            except Exception as e:
                n_skipped += 1
                exc_name = type(e).__name__
                print(f"[WARN] skipping item {item.get('pid')}: {exc_name}: {e}")
                # Full traceback once per distinct exception type -- enough to
                # diagnose a systematic bug (e.g. a device mismatch) without
                # spamming the same trace for every corrupt-cache item.
                if exc_name not in seen_exc_types:
                    seen_exc_types.add(exc_name)
                    traceback.print_exc()
                continue

            step += 1
            step_times.append(time.time() - item_start)
            if not slow_node_checked and len(step_times) >= SLOW_NODE_SAMPLE_SIZE:
                slow_node_checked = True
                avg_s = sum(step_times[-SLOW_NODE_SAMPLE_SIZE:]) / SLOW_NODE_SAMPLE_SIZE
                if avg_s > SLOW_NODE_THRESHOLD_S:
                    print(f"[WARN] this node looks I/O-bound: avg {avg_s:.1f}s/step over the "
                          f"first {SLOW_NODE_SAMPLE_SIZE} successful steps (healthy nodes have "
                          f"run ~20-35s/step) -- the GPU is likely idle most of the time waiting "
                          f"on video/feature reads from network storage. Consider killing this "
                          f"job and resubmitting to get scheduled onto a different node.", flush=True)
            if step % 50 == 0:
                print(f"[epoch {epoch} step {step}] loss={loss.item():.4f}")
            log_f.write(json.dumps({"step": step, "epoch": epoch, "loss": loss.item()}) + "\n")
            log_f.flush()

            if step % args.val_every_steps == 0:
                val_tally = evaluate_logprob(model, processor, hook, injector, val_items, args.injection_site, n_tokens, placeholder_id, device, rng)
                print(f"[epoch {epoch} step {step}] val_acc(logprob)={val_tally.summary_str()}")
                log_f.write(json.dumps({
                    "step": step, "val_acc": val_tally.acc, "val_acc_by_difficulty": val_tally.acc_by_difficulty,
                }) + "\n")
                log_f.flush()
                if val_tally.acc > best_val_acc:
                    best_val_acc = val_tally.acc
                    torch.save({"args": vars(args), "injector_state": injector.state_dict()},
                               os.path.join(run_dir, "best.pt"))

            if args.save_every_steps and step % args.save_every_steps == 0:
                # Independent of best.pt: best.pt only ever holds the single
                # snapshot that first reached the run's high-water mark, so a
                # later tie or a noisy dip-then-recover (e.g. val_acc cratering
                # at one checkpoint and climbing back by the next) leaves no
                # way to go back and compare intermediate states -- confirmed
                # in practice on the jepa x cross_attn temporal-split run,
                # where best.pt froze at step 3000 despite training continuing
                # to step 6641. These periodic snapshots keep that history.
                torch.save({"args": vars(args), "injector_state": injector.state_dict(), "step": step},
                           os.path.join(run_dir, f"step_{step}.pt"))
                print(f"[epoch {epoch} step {step}] saved checkpoint -> step_{step}.pt")

    log_f.close()
    if step == 0:
        # best_val_acc is still its -1.0 init sentinel because validation
        # (gated on successful *training* steps hitting val_every_steps)
        # never ran once -- every train item raised an exception. Flag this
        # loudly instead of printing a bare -1.0000 that reads like a real
        # (if bad) score.
        print(f"[DONE] injection={args.injection}  best_val_acc(logprob)=N/A "
              f"(0/{len(train_items) * args.epochs} items succeeded -- see [WARN] lines above)")
    else:
        print(f"[DONE] injection={args.injection}  best_val_acc(logprob)={best_val_acc:.4f}  "
              f"steps_completed={step}  items_skipped={n_skipped}")


if __name__ == "__main__":
    main()
