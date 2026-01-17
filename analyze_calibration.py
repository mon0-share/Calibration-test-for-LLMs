#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 15 16:49:56 2026

@author: christopher
"""

"""
analyze_calibration.py

Reads:
  - answers.jsonl         (your model outputs; one JSON object per line)
  - correct_answers.txt   (ground-truth: "<id> <A|B|C|D>" per line)

Computes:
  - pooled Brier score (over all graded rows)
  - irrational counts per run (p_correct < 0.25 by default)
  - top 5 most miscalibrated rows (max |p_correct - y|)
  - top 5 runs with the most irrational answers
  - top 5 question IDs with the most irrational answers (across runs)

Usage:
  python analyze_calibration.py --answers answers.jsonl --key correct_answers.txt
"""

import argparse, json
from collections import Counter, defaultdict

CHOICES = {"A", "B", "C", "D"}

def load_key(path):
    key = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    qid = int(parts[0])
                except ValueError:
                    continue
                ans = parts[1].upper()
                if ans in CHOICES:
                    key[qid] = ans
    return key

def bucket_idx(p):
    if 0.25 <= p < 0.50: return 0
    if 0.50 <= p < 0.75: return 1
    if 0.75 <= p <= 1.00: return 2
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", default="answers.jsonl")
    ap.add_argument("--key", default="correct_answers.txt")
    ap.add_argument("--irr-thr", type=float, default=0.25)
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args()

    key = load_key(args.key)
    if not key:
        raise SystemExit(f"No key entries loaded from {args.key}")

    rows = []  # (run, qid, p, y)
    stats = Counter()

    with open(args.answers, "r", encoding="utf-8") as f:
        for raw in f:
            s = raw.strip()
            if not s:
                continue
            try:
                o = json.loads(s)
            except Exception:
                stats["bad_json"] += 1
                continue
            if isinstance(o, dict) and o.get("parse_error"):
                stats["parse_error_lines"] += 1
                continue
            try:
                run = int(o["run"]); qid = int(o["id"])
                ans = str(o["answer"]).strip().upper()
                p = float(o["p_correct"])
            except Exception:
                stats["bad_fields"] += 1
                continue
            if qid not in key:
                stats["no_key"] += 1
                continue
            if ans not in CHOICES or not (0.0 <= p <= 1.0):
                stats["bad_value"] += 1
                continue

            y = 1 if ans == key[qid] else 0
            rows.append((run, qid, p, y))
            stats["ok"] += 1

    if not rows:
        raise SystemExit(f"No usable rows. Stats: {dict(stats)}")

    n = len(rows)
    pooled_brier = sum((p - y) ** 2 for _, _, p, y in rows) / n
    pooled_acc = sum(y for _, _, _, y in rows) / n

    n_by_run = defaultdict(int)
    irr_by_run = defaultdict(int)
    irr_by_qid = Counter()

    mis_sum = defaultdict(float)  # mean |p-y| per qid
    mis_n = defaultdict(int)

    # Buckets: [0.25,0.5), [0.5,0.75), [0.75,1]
    b_n = [0, 0, 0]
    b_p = [0.0, 0.0, 0.0]
    b_y = [0.0, 0.0, 0.0]
    b_outside = 0

    for run, qid, p, y in rows:
        n_by_run[run] += 1
        if p < args.irr_thr:
            irr_by_run[run] += 1
            irr_by_qid[qid] += 1
        mis_sum[qid] += abs(p - y)
        mis_n[qid] += 1

        bi = bucket_idx(p)
        if bi is None:
            b_outside += 1
        else:
            b_n[bi] += 1
            b_p[bi] += p
            b_y[bi] += y

    top_runs = sorted(irr_by_run.items(), key=lambda kv: (-kv[1], kv[0]))[:args.top]
    top_irr_qids = irr_by_qid.most_common(args.top)

    mis_by_qid = [(qid, mis_sum[qid] / mis_n[qid], mis_n[qid]) for qid in mis_n]
    top_miscal_qids = sorted(mis_by_qid, key=lambda t: t[1], reverse=True)[:args.top]

    extras = {k: v for k, v in stats.items() if k != "ok" and v}
    print(f"OK rows: {stats['ok']}  (other parse stats: {extras})")
    print(f"Pooled accuracy: {pooled_acc:.4f}")
    print(f"Pooled Brier:    {pooled_brier:.6f}")

    print(f"\nCalibration buckets (pooled):")
    labels = ["[0.25,0.5)", "[0.5,0.75)", "[0.75,1]"]
    for i, lab in enumerate(labels):
        if b_n[i] == 0:
            print(f"  {lab}: n=0")
        else:
            mean_p = b_p[i] / b_n[i]
            acc = b_y[i] / b_n[i]
            gap = acc - mean_p
            print(f"  {lab}: n={b_n[i]}  mean_p={mean_p:.3f}  acc={acc:.3f}  gap(acc-mean_p)={gap:+.3f}")
    if b_outside:
        print(f"  (outside these buckets): n={b_outside}")

    print(f"\nIrrational (p < {args.irr_thr}) by run:")
    for run in sorted(n_by_run):
        irr = irr_by_run.get(run, 0)
        tot = n_by_run[run]
        print(f"  run {run:>3}: {irr:>3}/{tot} ({irr/tot:.3%})")

    print(f"\nTop {args.top} runs with most irrational answers:")
    if not top_runs:
        print("  (none)")
    for run, cnt in top_runs:
        tot = n_by_run[run]
        print(f"  run {run:>3}: {cnt}/{tot} ({cnt/tot:.3%})")

    print(f"\nTop {args.top} question IDs with most irrational answers:")
    if not top_irr_qids:
        print("  (none)")
    for qid, cnt in top_irr_qids:
        denom = mis_n.get(qid, 0) or 1
        print(f"  id {qid:>2}: {cnt}/{denom} ({(cnt/denom):.3%})")

    print(f"\nTop {args.top} most miscalibrated question IDs (mean |p - y| across runs):")
    for qid, mean_abs_mis, denom in top_miscal_qids:
        print(f"  id {qid:>2}: mean|p-y|={mean_abs_mis:.3f} over {denom} answers")

if __name__ == "__main__":
    main()

