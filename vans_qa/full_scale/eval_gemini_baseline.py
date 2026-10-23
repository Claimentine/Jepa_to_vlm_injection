"""A2 baseline: Gemini 2.5 Flash, zero-shot, via the Google AI Studio API
(free tier -- NOT Vertex AI, which has no free tier). Same "predict what
happens next" 2-choice QA task as our other zero-shot VLM baselines
(train_qa_full.py's own build_option_block/build_prompt_text, reused
unmodified so the prompt text is identical across every baseline), but
answered by direct generation ("respond with only the option letter")
and parsed, not by comparing token logprobs the way our LOCAL-model
baselines do (candidate_logprob) -- the Gemini API does not expose raw
next-token logprobs the way a locally-loaded HF model does, and direct
generation + parsing is also how closed-API models are conventionally
scored in the broader literature (e.g. ThinkJEPA's own Table 10
prompt-only baseline).

Free-tier rate limits (user-confirmed, 2026-10-22): RPD ~500-1500/day,
RPM 15, TPM 250k. RPD is the binding constraint by a wide margin at our
item counts (COIN 2846 + CrossTask 1558 = 4404 -- multiple DAYS even at
1 call/item), so per the user's explicit decision this script evaluates
a SUBSAMPLE (--limit, default 200 per benchmark) rather than the full
split, comfortably inside a single day's quota even at the 500/day
floor. No GPU needed at all -- pure API calls, runs as a CPU-only pod.

Resumable: writes one JSON line per completed item to --out_jsonl
immediately (not batched at the end), and skips pids already present on
a rerun -- a day-boundary RPD exhaustion (or any other interruption)
loses no completed work; just rerun the same command tomorrow.
"""
import argparse
import json
import os
import random
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import train_qa_full as fwd  # noqa: E402

MODEL_ID = "gemini-2.5-flash"
RPM_LIMIT = 15
MIN_SECONDS_BETWEEN_CALLS = 60.0 / RPM_LIMIT + 0.5  # +0.5s margin, don't shave it exact


def load_done_pids(out_jsonl):
    done = {}
    if not os.path.exists(out_jsonl):
        return done
    with open(out_jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                done[rec["pid"]] = rec
            except json.JSONDecodeError:
                continue  # a half-written final line from a killed run -- drop it, not trust it
    return done


def clip_exists(path):
    """Gemini only ever needs the raw video clip -- unlike our own model's
    evaluate_logprob/evaluate_logprob_none, it never touches VJEPA
    features, so fwd.has_features() (which checks for a VJEPA npz cache
    entry, not the clip itself) is the wrong filter here: the first run
    of this script used it and filtered out every single item (0/2866
    usable) because this job's environment never set up the VJEPA_CACHE
    symlink COIN's own eval needs, which is irrelevant to Gemini anyway."""
    return os.path.exists(path) and os.path.getsize(path) > 1000


def parse_letter(text):
    """Pulls a single A/B choice out of the model's free-text response --
    even with an explicit "respond with only the letter" instruction,
    generation models sometimes wrap it ("The answer is A.", "**A**")."""
    text = (text or "").strip()
    for ch in text:
        if ch in ("A", "B"):
            return ch
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--qa_split", required=True)
    ap.add_argument("--split_part", default="test", choices=["val", "test", "train"])
    ap.add_argument("--limit", type=int, default=200,
                     help="subsample size -- see module docstring for the RPD-budget reasoning "
                          "behind defaulting to 200, not the full split")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_jsonl", required=True)
    args = ap.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY must be set (from the gemini-api-key Secret)")

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)

    with open(args.qa_split) as f:
        split = json.load(f)
    items = [it for it in split[args.split_part] if clip_exists(it["in_clip"])]
    print(f"[INFO] {args.split_part} split: {len(items)} usable items "
          f"(of {len(split[args.split_part])} total)", flush=True)

    rng = random.Random(args.seed)
    rng.shuffle(items)  # subsample is a random slice, not just the first N in file order
    if args.limit:
        items = items[: args.limit]
    print(f"[INFO] evaluating {len(items)} items (--limit {args.limit}, --seed {args.seed})", flush=True)

    done = load_done_pids(args.out_jsonl)
    print(f"[INFO] {len(done)} item(s) already completed in {args.out_jsonl} -- will be skipped", flush=True)

    n_correct = sum(1 for r in done.values() if r.get("correct"))
    n_total = len(done)
    n_failed = 0
    seen_exc_types = set()
    last_call_time = 0.0

    with open(args.out_jsonl, "a") as out_f:
        for i, item in enumerate(items, start=1):
            pid = item.get("pid")
            if pid in done:
                continue

            option_lines, correct_letter = fwd.build_option_block(item, rng)
            prompt_text = fwd.build_prompt_text(item["question"], option_lines)

            try:
                # Pace to RPM_LIMIT regardless of how fast the API itself responds.
                elapsed = time.monotonic() - last_call_time
                if elapsed < MIN_SECONDS_BETWEEN_CALLS:
                    time.sleep(MIN_SECONDS_BETWEEN_CALLS - elapsed)

                uploaded = client.files.upload(file=item["in_clip"])
                # Files API processes asynchronously -- must reach ACTIVE
                # before it can be referenced in generateContent.
                while uploaded.state.name == "PROCESSING":
                    time.sleep(2)
                    uploaded = client.files.get(name=uploaded.name)
                if uploaded.state.name != "ACTIVE":
                    raise RuntimeError(f"file upload ended in state {uploaded.state.name!r}")

                response = client.models.generate_content(
                    model=MODEL_ID,
                    contents=[uploaded, prompt_text],
                    config=types.GenerateContentConfig(
                        temperature=0.0,
                        max_output_tokens=32,
                        # Gemini 2.5 Flash thinks by default; thinking tokens
                        # count against max_output_tokens, so the first run
                        # (budget unset, 16-token cap) spent the entire
                        # budget "thinking" and emitted zero visible text --
                        # response.text came back None on every item, not
                        # just unlucky ones (confirmed: first 2/2 items both
                        # failed this way). thinking_budget=0 disables it --
                        # appropriate for a plain forced-choice task, and
                        # also cheaper per call under the free-tier TPM cap.
                        thinking_config=types.ThinkingConfig(thinking_budget=0),
                    ),
                )
                last_call_time = time.monotonic()

                try:
                    client.files.delete(name=uploaded.name)
                except Exception:
                    pass  # Files API auto-expires after 48h regardless -- not worth failing the item over

                pred_letter = parse_letter(response.text)
                if pred_letter is None:
                    finish_reason = None
                    try:
                        finish_reason = response.candidates[0].finish_reason
                    except Exception:
                        pass
                    raise ValueError(f"could not parse A/B from response: {response.text!r} "
                                      f"(finish_reason={finish_reason})")
                correct = (pred_letter == correct_letter)
                n_correct += int(correct)
                n_total += 1
                rec = {"pid": pid, "pred": pred_letter, "correct_letter": correct_letter, "correct": correct}
                out_f.write(json.dumps(rec) + "\n")
                out_f.flush()
                if i % 20 == 0 or i == len(items):
                    print(f"[{i}/{len(items)}] running_acc={n_correct}/{n_total}="
                          f"{n_correct / max(n_total, 1):.4f}", flush=True)
            except Exception as e:
                n_failed += 1
                exc_name = type(e).__name__
                print(f"[WARN] skipping {pid}: {exc_name}: {e}", flush=True)
                if exc_name not in seen_exc_types:
                    seen_exc_types.add(exc_name)
                    import traceback
                    traceback.print_exc()
                last_call_time = time.monotonic()  # a failed call still counts against RPM

    print(f"[DONE] n_total={n_total} n_correct={n_correct} acc={n_correct / max(n_total, 1):.4f} "
          f"n_failed={n_failed} out={args.out_jsonl}", flush=True)


if __name__ == "__main__":
    main()
