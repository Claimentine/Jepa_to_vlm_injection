"""Merges several qa_split JSON files (same {"train"/"val"/"test": [item,
...]} schema, each item having at least pid/in_clip/out_clip/question/
correct_caption/distractor_caption) into one combined file, for a single
--qa_split argument to point train_qa_full.py/train_joint_coupled_
normalized.py at multiple data sources at once.

Checks pid uniqueness across ALL inputs combined (not just within one
file) before writing anything -- a silent pid collision would make two
different items indistinguishable to anything that indexes by pid (e.g.
the reverse side's cache lookup), which is exactly the kind of bug that
stays invisible until a training run's numbers look wrong for no visible
reason.
"""
import argparse
import json
import sys
from collections import Counter


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inputs", nargs="+", required=True, help="qa_split JSON files to merge, in order")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    merged = {"train": [], "val": [], "test": []}
    all_pids = Counter()
    per_file_counts = []
    for path in args.inputs:
        with open(path) as f:
            split = json.load(f)
        counts = {}
        for part in ("train", "val", "test"):
            items = split.get(part, [])
            counts[part] = len(items)
            merged[part].extend(items)
            all_pids.update(item["pid"] for item in items)
        per_file_counts.append((path, counts))
        print(f"[INFO] {path}: {counts}", flush=True)

    dupes = {pid: n for pid, n in all_pids.items() if n > 1}
    if dupes:
        print(f"[ERROR] {len(dupes)} pid(s) appear in more than one input file -- refusing to write a "
              f"merged file with ambiguous pids. First few: {list(dupes.items())[:10]}", file=sys.stderr)
        sys.exit(1)

    with open(args.out, "w") as f:
        json.dump(merged, f, indent=2)
    total = {part: len(merged[part]) for part in ("train", "val", "test")}
    print(f"[DONE] merged {len(args.inputs)} file(s) -> {args.out}: {total}", flush=True)


if __name__ == "__main__":
    main()
