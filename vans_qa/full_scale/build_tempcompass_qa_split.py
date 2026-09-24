"""Converts TempCompass's multi-choice QA annotations into this project's
--qa_split JSON schema, for an out-of-domain generalization eval of forward-
direction checkpoints (job-13/22/34/36/...) trained entirely on COIN-derived
qa_split_temporal.json data.

TempCompass (Liu et al., ACL 2024 Findings): 410 videos, 500 clips, 7,540 QA
pairs testing video-LLM temporal perception (action/direction/speed/order/
attribute_change), hosted at huggingface.co/datasets/lmms-lab/TempCompass.
Its multi-choice parquet has columns video_id/question/answer/dim, where
`question` embeds the options inline as "<base question>\nA. opt1\nB. opt2\n..."
and `answer` is the full matched option text e.g. "A. dunking a basketball".

train_qa_full.py's evaluate_logprob/build_option_block (reused unmodified by
eval_forward_checkpoint.py) hardcode exactly one correct + one distractor
option (see their own docstrings) -- no native N-way scoring. TempCompass MC
items have 2-4 options, so each item is expanded into (n_options - 1) rows,
one per (correct, single-wrong-option) pair, all sharing the same in_clip/
question/pid-prefix. This is the same "pairwise A/B" evaluation semantics
this project has used from the start (COIN-derived QA pairs are natively
2-option), just applied per-wrong-option here instead of assuming a single
pre-picked distractor.

`difficulty` is set to TempCompass's own `dim` field (action/direction/
speed/order/attribute_change) so AccTally's existing by-difficulty breakdown
(train_qa_full.py's AccTally.add) reports per-category accuracy for free,
exactly like it already does for temporal's easy/hard tags.

video_id is used verbatim as the clip filename ("<video_id>.mp4"), including
the "_reverse" suffix TempCompass gives to its temporally-reversed variants
(job-40's tempcompass_videos.zip extraction is expected to contain both).

All items go into "test" (out-of-domain generalization eval only -- no
training on this data, so "train"/"val" stay empty).
"""
import argparse
import json
import os
import re

MC_ANSWER_RE = re.compile(r"^([A-Za-z])\.\s*(.*)$")
MC_OPTION_RE = re.compile(r"^\s*([A-Za-z])\.\s*(.*)$")


def parse_mc_item(question, answer):
    """Returns (base_question, correct_text, [wrong_text, ...]) or None if
    the question/answer text doesn't parse as expected (logged and skipped
    by the caller rather than raising, since this converts data we don't
    control the formatting of)."""
    lines = [l for l in question.strip().split("\n") if l.strip()]
    if len(lines) < 3:  # need a question line + >=2 option lines
        return None
    base_question = lines[0].strip()
    options = []
    for line in lines[1:]:
        m = MC_OPTION_RE.match(line)
        if m:
            options.append((m.group(1).upper(), m.group(2).strip()))
    if len(options) < 2:
        return None

    ans_m = MC_ANSWER_RE.match(answer.strip())
    correct_letter = ans_m.group(1).upper() if ans_m else None
    by_letter = dict(options)
    if correct_letter not in by_letter:
        # fall back to matching the answer text directly against an option
        ans_text = (ans_m.group(2) if ans_m else answer).strip().lower()
        correct_letter = next((l for l, t in options if t.lower() == ans_text), None)
    if correct_letter is None:
        return None

    correct_text = by_letter[correct_letter]
    wrong_texts = [t for l, t in options if l != correct_letter]
    return base_question, correct_text, wrong_texts


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parquet", required=True,
                     help="path to TempCompass's multi-choice/test-*.parquet")
    ap.add_argument("--clips_root", required=True,
                     help="absolute path to $VANS_WORK_ROOT/clips on the pod -- in_clip must be "
                          "an absolute path under this directory, since safe_name()'s "
                          "os.path.relpath(item['in_clip'], CLIPS_ROOT) call (train_qa_full.py, "
                          "extract_vjepa_features_full.py) misresolves a relative in_clip against "
                          "the process's CWD instead")
    ap.add_argument("--clips_subdir", default="tempcompass/videos",
                     help="in_clip is built as {clips_root}/{clips_subdir}/{video_id}.mp4 -- "
                          "default matches tempcompass_videos.zip's own internal videos/ folder "
                          "(confirmed via its central directory: videos/<id>.mp4, "
                          "videos/<id>_reverse.mp4, plus some multi-clip videos/<id1>_<id2>_<n>.mp4 "
                          "used by other TempCompass tasks, not multi-choice)")
    ap.add_argument("--out_path", required=True)
    args = ap.parse_args()

    import pyarrow.parquet as pq
    table = pq.read_table(args.parquet)
    rows = table.to_pylist()
    print(f"[INFO] loaded {len(rows)} rows from {args.parquet}", flush=True)

    items = []
    n_parsed, n_skipped = 0, 0
    for i, row in enumerate(rows):
        parsed = parse_mc_item(row["question"], row["answer"])
        if parsed is None:
            n_skipped += 1
            print(f"[WARN] could not parse row {i} (video_id={row.get('video_id')}): "
                  f"question={row['question']!r} answer={row['answer']!r}", flush=True)
            continue
        n_parsed += 1
        base_question, correct_text, wrong_texts = parsed
        in_clip = os.path.join(args.clips_root, args.clips_subdir, f"{row['video_id']}.mp4")
        for j, wrong_text in enumerate(wrong_texts):
            items.append({
                "pid": f"tempcompass_{row['video_id']}_{row['dim']}_{i}_{j}",
                "in_clip": in_clip,
                "out_clip": in_clip,  # unused by the forward-only eval path; set for
                                      # extract_vjepa_features_full.py's unconditional item["out_clip"] read
                "question": base_question,
                "correct_caption": correct_text,
                "distractor_caption": wrong_text,
                "difficulty": row["dim"],
            })

    print(f"[INFO] parsed {n_parsed} items ({n_skipped} skipped) -> {len(items)} A/B eval rows "
          f"(pairwise-expanded over wrong options)", flush=True)

    split = {"train": [], "val": [], "test": items}
    with open(args.out_path, "w") as f:
        json.dump(split, f, indent=2)
    print(f"[DONE] wrote {args.out_path}", flush=True)


if __name__ == "__main__":
    main()
