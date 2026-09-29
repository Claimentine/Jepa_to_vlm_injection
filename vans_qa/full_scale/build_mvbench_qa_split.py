"""Converts MVBench's 20 per-category multi-choice QA JSON files into this
project's --qa_split schema, for an out-of-domain generalization eval of
forward-direction checkpoints (job-13/22/34/36/42/...) trained entirely on
VANS(COIN+YouCook2) data -- same purpose as TempCompass
(build_tempcompass_qa_split.py), a second, broader-coverage benchmark.

MVBench (Li et al., OpenGVLab): 4,000 QA items across 20 temporal-reasoning
categories (action_antonym, action_count, moving_direction, ...), sourced
from 11 different underlying video datasets (STAR, CLEVRER, Perception
Test, Something-Something V2, TVQA, ...) each packed as its own zip under
huggingface.co/datasets/OpenGVLab/MVBench/video/. Videos are matched to QA
items purely by filename (e.g. "166583.webm") -- items whose referenced
video isn't found (e.g. the ~320 NTU RGB+D items, which need separate
manual access per MVBench's own dataset card) are simply skipped, the same
has_features()-style skip this project already uses elsewhere.

Per-category JSON schema (confirmed directly from json/action_antonym.json):
  {"video": "166583.webm", "question": "...", "candidates": [opt1, opt2, ...],
   "answer": "<the correct option's exact text>"}
3 candidates is typical but not guaranteed uniform across all 20
categories, so this script makes no assumption about count. Like
TempCompass, train_qa_full.py's evaluate_logprob/build_option_block
hardcode exactly one correct + one distractor (no native N-way scoring),
so each item is pairwise-expanded over its wrong candidates.

`difficulty` is set to the category name (the JSON filename's stem, e.g.
"action_antonym"), so AccTally's existing by-difficulty breakdown reports
per-category accuracy for free, matching TempCompass's `dim` usage.

All items go into "test" (out-of-domain generalization eval only).
"""
import argparse
import glob
import json
import os


def parse_mc_item(question, candidates, answer):
    """Returns (correct_text, [wrong_text, ...]) or None if the answer
    doesn't match any candidate exactly (logged and skipped by the caller,
    not raised -- this converts data we don't control the formatting of)."""
    if not candidates or answer not in candidates:
        return None
    wrong = [c for c in candidates if c != answer]
    if not wrong:
        return None
    return answer, wrong


def find_video_path(video_filename, video_root):
    # MVBench's 11 source zips are extracted into one flat video_root
    # (see job's own extraction step) -- a plain os.path.join lookup covers
    # the common case; if a source zip preserved its own subfolder instead
    # of flattening, fall back to a recursive glob (slower, but this only
    # runs once per item at conversion time, not per training step).
    direct = os.path.join(video_root, video_filename)
    if os.path.exists(direct):
        return direct
    matches = glob.glob(os.path.join(video_root, "**", video_filename), recursive=True)
    return matches[0] if matches else None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json_dir", required=True,
                     help="directory containing MVBench's 20 category *.json files")
    ap.add_argument("--video_root", required=True,
                     help="directory the 11 source zips were extracted into (searched recursively "
                          "as a fallback if a filename isn't found at the top level)")
    ap.add_argument("--out_path", required=True)
    args = ap.parse_args()

    json_paths = sorted(glob.glob(os.path.join(args.json_dir, "*.json")))
    print(f"[INFO] found {len(json_paths)} category files in {args.json_dir}", flush=True)

    items = []
    n_parsed = n_skipped_answer = n_skipped_video = 0
    for json_path in json_paths:
        category = os.path.splitext(os.path.basename(json_path))[0]
        with open(json_path) as f:
            rows = json.load(f)
        for i, row in enumerate(rows):
            parsed = parse_mc_item(row.get("question", ""), row.get("candidates"), row.get("answer"))
            if parsed is None:
                n_skipped_answer += 1
                continue
            correct_text, wrong_texts = parsed
            video_path = find_video_path(row["video"], args.video_root)
            if video_path is None:
                n_skipped_video += 1
                continue
            n_parsed += 1
            for j, wrong_text in enumerate(wrong_texts):
                items.append({
                    "pid": f"mvbench_{category}_{i}_{j}",
                    "in_clip": video_path,
                    "out_clip": video_path,  # unused by the forward-only eval path; set for
                                             # extract_vjepa_features_full.py's unconditional item["out_clip"] read
                    "question": row["question"],
                    "correct_caption": correct_text,
                    "distractor_caption": wrong_text,
                    "difficulty": category,
                })

    print(f"[INFO] parsed {n_parsed} items ({n_skipped_answer} skipped: answer not in candidates, "
          f"{n_skipped_video} skipped: video file not found) -> {len(items)} A/B eval rows "
          f"(pairwise-expanded over wrong candidates)", flush=True)

    split = {"train": [], "val": [], "test": items}
    with open(args.out_path, "w") as f:
        json.dump(split, f, indent=2)
    print(f"[DONE] wrote {args.out_path}", flush=True)


if __name__ == "__main__":
    main()
