"""
Scores probe_results.jsonl produced by run_task_recognition_probe.py.

Reports, per input condition (full_clip / past_window / single_frame):
  - macro accuracy (mean of per-category recall -- avoids large categories
    like basic_pick_place dominating the number)
  - micro accuracy (plain overall accuracy, reported alongside for reference)
  - parse-fail rate (how often the model didn't answer in the requested
    "Final answer: <LETTER>" format -- a high rate here means the *prompt*
    needs work, independent of task-recognition ability)
  - confusion matrix (which categories get mixed up with which)
"""
import argparse
import json
from collections import defaultdict

from task_categories import CATEGORIES


def load_records(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def score_condition(records):
    per_cat_total = defaultdict(int)
    per_cat_correct = defaultdict(int)
    confusion = defaultdict(lambda: defaultdict(int))
    n_total, n_correct, n_error, n_unparsed = 0, 0, 0, 0

    for r in records:
        if "error" in r:
            n_error += 1
            continue
        n_total += 1
        true_cat = r["true_category"]
        pred_cat = r["pred_category"]
        per_cat_total[true_cat] += 1
        if pred_cat is None:
            n_unparsed += 1
        if r["correct"]:
            n_correct += 1
            per_cat_correct[true_cat] += 1
        confusion[true_cat][pred_cat or "<UNPARSED>"] += 1

    micro_acc = n_correct / max(n_total, 1)
    per_cat_recall = {
        c: per_cat_correct[c] / per_cat_total[c] for c in per_cat_total if per_cat_total[c] > 0
    }
    macro_acc = sum(per_cat_recall.values()) / max(len(per_cat_recall), 1)
    parse_fail_rate = n_unparsed / max(n_total, 1)

    return {
        "n_total": n_total,
        "n_error": n_error,
        "micro_acc": micro_acc,
        "macro_acc": macro_acc,
        "parse_fail_rate": parse_fail_rate,
        "per_cat_recall": per_cat_recall,
        "confusion": confusion,
    }


def print_report(condition, stats):
    print(f"\n{'='*70}\nCondition: {condition}\n{'='*70}")
    print(f"n={stats['n_total']}  errors={stats['n_error']}  parse_fail_rate={stats['parse_fail_rate']:.3f}")
    print(f"micro_acc={stats['micro_acc']:.3f}   macro_acc={stats['macro_acc']:.3f}")
    print("\nPer-category recall:")
    for c in CATEGORIES:
        r = stats["per_cat_recall"].get(c)
        print(f"  {c:55s} {r:.3f}" if r is not None else f"  {c:55s}  (no samples)")

    print("\nTop confusions (true -> most common wrong prediction):")
    for c in CATEGORIES:
        row = stats["confusion"].get(c, {})
        wrong = {k: v for k, v in row.items() if k != c}
        if wrong:
            top = max(wrong.items(), key=lambda kv: kv[1])
            print(f"  {c:55s} -> {top[0]:55s} ({top[1]} times)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="probe_results.jsonl")
    args = ap.parse_args()

    records = load_records(args.results)
    by_condition = defaultdict(list)
    for r in records:
        by_condition[r["condition"]].append(r)

    for condition in ["full_clip", "past_window", "single_frame"]:
        if condition in by_condition:
            stats = score_condition(by_condition[condition])
            print_report(condition, stats)


if __name__ == "__main__":
    main()
