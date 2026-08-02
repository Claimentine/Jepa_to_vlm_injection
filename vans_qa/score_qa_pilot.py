"""
Scores the VANS QA pilot with the statistics this small N actually
supports: accuracy + Wilson confidence interval per condition, and a paired
McNemar comparison between two conditions (same test items -- see
README_pilot_caveats.md for why paired beats comparing independent points).
"""
import argparse
import json
import math


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    denom = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denom
    half = (z * math.sqrt((p * (1 - p) + z ** 2 / (4 * n)) / n)) / denom
    return (p, max(0.0, center - half), min(1.0, center + half))


def load(path):
    recs = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if "error" not in r:
                recs[r["pid"]] = r
    return recs


def mcnemar_paired(a_recs, b_recs, a_name, b_name):
    common = set(a_recs) & set(b_recs)
    a_right_b_wrong = sum(1 for pid in common if a_recs[pid]["correct"] and not b_recs[pid]["correct"])
    b_right_a_wrong = sum(1 for pid in common if b_recs[pid]["correct"] and not a_recs[pid]["correct"])
    both_right = sum(1 for pid in common if a_recs[pid]["correct"] and b_recs[pid]["correct"])
    both_wrong = sum(1 for pid in common if not a_recs[pid]["correct"] and not b_recs[pid]["correct"])
    n_disc = a_right_b_wrong + b_right_a_wrong
    if n_disc > 0:
        chi2 = (abs(a_right_b_wrong - b_right_a_wrong) - 1) ** 2 / n_disc  # continuity-corrected
    else:
        chi2 = 0.0
    print(f"\nPaired comparison ({a_name} vs {b_name}, n_common={len(common)}):")
    print(f"  both correct:        {both_right}")
    print(f"  both wrong:          {both_wrong}")
    print(f"  {a_name} right, {b_name} wrong: {a_right_b_wrong}")
    print(f"  {b_name} right, {a_name} wrong: {b_right_a_wrong}")
    print(f"  McNemar chi2 (continuity-corrected): {chi2:.3f}  (>=3.84 ~ p<0.05, but n={n_disc} discordant pairs is small -- read this as a hint, not a verdict)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", required=True, help="one or more qa_eval_*.jsonl files")
    args = ap.parse_args()

    all_recs = {}
    for path in args.results:
        name = path.split("qa_eval_")[-1].replace(".jsonl", "")
        recs = load(path)
        all_recs[name] = recs
        n = len(recs)
        k = sum(1 for r in recs.values() if r["correct"])
        p, lo, hi = wilson_ci(k, n)
        n_unparsed = sum(1 for r in recs.values() if r.get("pred_letter") is None)
        print(f"{name:10s}  n={n:3d}  acc={p:.3f}  95% CI=[{lo:.3f}, {hi:.3f}]  unparsed={n_unparsed}")

    names = list(all_recs.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            mcnemar_paired(all_recs[names[i]], all_recs[names[j]], names[i], names[j])


if __name__ == "__main__":
    main()
