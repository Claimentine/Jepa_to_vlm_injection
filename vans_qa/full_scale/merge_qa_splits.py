"""Merges several qa_split JSON files (same {"train"/"val"/"test": [item,
...]} schema, each item having at least pid/in_clip/out_clip/question/
correct_caption/distractor_caption) into one combined file, for a single
--qa_split argument to point train_qa_full.py/train_joint_coupled_
normalized.py at multiple data sources at once.

Checks pid uniqueness ACROSS INPUT FILES (not within one) before writing
anything -- a pid shared by two different source files would make two
different items indistinguishable to anything that indexes by pid (e.g.
the reverse side's cache lookup), which is exactly the kind of bug that
stays invisible until a training run's numbers look wrong for no visible
reason. Deliberately does NOT flag a pid repeated within a single input
file -- qa_split_temporal.json's own val/test rows intentionally repeat
each anchor's pid twice (difficulty="easy"/"hard" variants, see
build_qa_pairs_temporal.py's docstring), which is a legitimate existing
convention, not a bug to catch here.
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
    pid_source_files = {}  # pid -> set of input file paths it appeared in
    for path in args.inputs:
        with open(path) as f:
            split = json.load(f)
        counts = {}
        for part in ("train", "val", "test"):
            items = split.get(part, [])
            counts[part] = len(items)
            merged[part].extend(items)
            for item in items:
                pid_source_files.setdefault(item["pid"], set()).add(path)
        print(f"[INFO] {path}: {counts}", flush=True)

    cross_file_dupes = {pid: paths for pid, paths in pid_source_files.items() if len(paths) > 1}
    if cross_file_dupes:
        print(f"[ERROR] {len(cross_file_dupes)} pid(s) appear in more than one INPUT FILE -- refusing to "
              f"write a merged file with ambiguous pids. First few: "
              f"{[(pid, sorted(paths)) for pid, paths in list(cross_file_dupes.items())[:10]]}", file=sys.stderr)
        sys.exit(1)

    with open(args.out, "w") as f:
        json.dump(merged, f, indent=2)
    total = {part: len(merged[part]) for part in ("train", "val", "test")}
    print(f"[DONE] merged {len(args.inputs)} file(s) -> {args.out}: {total}", flush=True)


if __name__ == "__main__":
    main()
