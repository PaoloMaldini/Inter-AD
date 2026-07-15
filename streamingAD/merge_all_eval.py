#!/usr/bin/env python3
"""
merge_all_eval.py — Load ALL experiment JSONs, compute metrics as one unified dataset,
                    output a single CSV + aggregate summary.

Usage:
    conda activate videollava
    python streamingAD/merge_all_eval.py \
        --experiment-dir experiment_results/ \
        --output-csv eval_results/all_movies_unified.csv \
        --skip-llm
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".hf_cache"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(PROJECT_ROOT / ".hf_cache" / "hub"))
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(PROJECT_ROOT / ".hf_cache" / "sentence_transformers"))
os.environ.setdefault("PIP_CACHE_DIR", str(PROJECT_ROOT / ".pip_cache"))
os.environ.setdefault("TORCH_HOME", str(PROJECT_ROOT / ".torch_cache"))

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from interactive_experiment import load_experiment_result
from run_eval import (
    EmbeddingModel, LLMJudge,
    compute_ica, compute_sss, compute_idd, compute_sec, compute_se,
    compute_transition_smoothness, compute_ndcg,
    compute_isr_judge, compute_ifr_judge, compute_user_alignment,
    compute_preference_accuracy, compute_mse,
)


def main():
    parser = argparse.ArgumentParser(description="Merge all experiments → single CSV + unified summary")
    parser.add_argument("--experiment-dir", required=True, help="Directory with experiment JSONs")
    parser.add_argument("--output-csv", required=True, help="Output unified CSV path")
    parser.add_argument("--summary-csv", default=None, help="Output one-row aggregate summary CSV path")
    parser.add_argument("--llm-model", default=None, help="Path to Qwen LLM model")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--skip-llm", action="store_true")
    args = parser.parse_args()

    exp_dir = Path(args.experiment_dir)
    json_files = sorted(exp_dir.glob("*_experiment_*.json"))
    print(f"Found {len(json_files)} experiment JSON files")

    # ── Load all records ────────────────────────────────────────────────
    all_records = []
    for jf in json_files:
        try:
            result = load_experiment_result(jf)
            all_records.extend(result.insertion_records)
            print(f"  {jf.name}: {len(result.insertion_records)} records  (movie: {result.movie_title})")
        except Exception as e:
            print(f"  SKIP {jf.name}: {e}")

    print(f"\nTotal records: {len(all_records)}")
    if not all_records:
        print("No records found.")
        return

    # ── Load models ─────────────────────────────────────────────────────
    embedder = EmbeddingModel(
        model_name=str(PROJECT_ROOT / ".hf_cache" / "sentence_transformers" / "all-MiniLM-L6-v2"),
        device=f"cuda:{args.gpu}",
    )

    judge = None
    if not args.skip_llm and args.llm_model:
        judge = LLMJudge(model_path=args.llm_model, gpu=args.gpu)

    # ── Compute metrics per record ──────────────────────────────────────
    rows: List[Dict[str, Any]] = []
    after_texts = [r.text_after for r in all_records]

    for rec in tqdm(all_records, desc="Computing metrics"):
        new_instr = rec.instruction_text
        active_str = " | ".join(
            f"[{a.get('category', '')}] {a.get('template', '')}"
            for a in getattr(rec, 'all_active_instructions', [])
        ) if hasattr(rec, 'all_active_instructions') and rec.all_active_instructions else ""

        row = {
            "insertion_id": rec.insertion_id,
            "movie_title": rec.movie_title,
            "segment_idx": rec.segment_idx,
            "timestamp_sec": rec.insert_timestamp_sec,
            "category_id": rec.category_id,
            "category_name": rec.category_name,
            "instruction_new": rec.instruction_text,
            "instruction_before": getattr(rec, 'instruction_before', ''),
            "instruction_after": getattr(rec, 'instruction_after', ''),
            "all_active_instructions": active_str,
            "instruction_lang": rec.instruction_language,
            "active_instr_count": rec.active_instructions_count,
        }

        # Embedding metrics
        ica = compute_ica(embedder, new_instr, rec.text_after)
        sss = compute_sss(embedder, rec.text_before, rec.text_after)
        row["ICA"] = round(ica, 4)
        row["SSS"] = round(sss, 4)
        row["IDD"] = round(compute_idd(ica, sss), 4)
        row["SEC"] = round(compute_sec(embedder, rec.text_before, rec.text_after), 4)
        row["SE"] = round(compute_se(embedder, new_instr, rec.text_after), 4)

        # Timing
        row["Latency_Before_sec"] = rec.latency_before_sec
        row["Latency_After_sec"] = rec.latency_after_sec
        row["TTFF_Before_sec"] = rec.ttff_before_sec
        row["TTFF_After_sec"] = rec.ttff_after_sec

        # LLM metrics
        if judge is not None:
            isr_score, isr_r = compute_isr_judge(judge, new_instr, rec.text_after)
            row["ISR"] = isr_score
            row["ISR_rationale"] = isr_r[:200]

            ifr_score, ifr_r = compute_ifr_judge(judge, new_instr, rec.text_after)
            row["IFR"] = ifr_score
            row["IFR_rationale"] = ifr_r[:200]

            ua_score, ua_r = compute_user_alignment(judge, new_instr, rec.text_after)
            row["User_Alignment"] = ua_score
            row["User_Alignment_rationale"] = ua_r[:200]

            pref_choice, pref_r = compute_preference_accuracy(judge, new_instr, rec.text_before, rec.text_after)
            row["Preference_Accuracy"] = pref_choice
            row["Preference_Accuracy_rationale"] = pref_r[:200]

            row["MSE"] = round(compute_mse(float(ua_score), ica), 4)
        else:
            for col in ["ISR", "IFR", "User_Alignment", "Preference_Accuracy", "MSE",
                        "ISR_rationale", "IFR_rationale",
                        "User_Alignment_rationale", "Preference_Accuracy_rationale"]:
                row[col] = ""

        row["text_before"] = rec.text_before
        row["text_after"] = rec.text_after
        row["ref_ad"] = rec.ref_ad

        rows.append(row)

    # ── Global metrics ──────────────────────────────────────────────────
    ts_score = compute_transition_smoothness(embedder, after_texts)
    idd_scores = [r["IDD"] for r in rows]
    ndcg_score = compute_ndcg(idd_scores)

    for row in rows:
        row["Transition_Smoothness"] = round(ts_score, 4)
        row["nDCG"] = round(ndcg_score, 4)

    # ── Aggregate summary (one row for ALL data) ────────────────────────
    def agg_stats(vals):
        v = [x for x in vals if isinstance(x, (int, float))]
        if not v:
            return 0, 0
        return round(float(np.mean(v)), 4), round(float(np.std(v)), 4)

    summary: Dict[str, Any] = {
        "total_records": len(rows),
        "total_movies": len(set(r["movie_title"] for r in rows)),
    }
    for col in ["ICA", "SSS", "IDD", "SEC", "SE",
                "Latency_Before_sec", "Latency_After_sec",
                "TTFF_Before_sec", "TTFF_After_sec",
                "Transition_Smoothness", "nDCG"]:
        m, s = agg_stats([r[col] for r in rows])
        summary[f"{col}_mean"] = m
        summary[f"{col}_std"] = s

    if judge is not None:
        isr_vals = [r["ISR"] for r in rows if isinstance(r.get("ISR"), (int, float))]
        summary["ISR_pass_rate"] = round(sum(1 for v in isr_vals if v >= 2) / max(len(isr_vals), 1), 4)

        ifr_vals = [r["IFR"] for r in rows if isinstance(r.get("IFR"), (int, float))]
        summary["IFR_mean"] = round(float(np.mean(ifr_vals)), 4) if ifr_vals else 0

        ua_vals = [r["User_Alignment"] for r in rows if isinstance(r.get("User_Alignment"), (int, float))]
        summary["User_Alignment_mean"] = round(float(np.mean(ua_vals)), 4) if ua_vals else 0

        pref_vals = [r["Preference_Accuracy"] for r in rows if r.get("Preference_Accuracy") in ("after", "before", "tie")]
        summary["Preference_After_pct"] = round(
            sum(1 for v in pref_vals if v == "after") / max(len(pref_vals), 1), 4
        ) if pref_vals else 0

        mse_vals = [r["MSE"] for r in rows if isinstance(r.get("MSE"), (int, float))]
        summary["MSE_mean"] = round(float(np.mean(mse_vals)), 4) if mse_vals else 0

    # ── Per-category summary ────────────────────────────────────────────
    cat_summary: Dict[str, Dict] = {}
    for r in rows:
        cat = r["category_name"]
        if cat not in cat_summary:
            cat_summary[cat] = {"count": 0, "ICA_sum": 0, "SSS_sum": 0, "IDD_sum": 0}
        cat_summary[cat]["count"] += 1
        cat_summary[cat]["ICA_sum"] += r["ICA"]
        cat_summary[cat]["SSS_sum"] += r["SSS"]
        cat_summary[cat]["IDD_sum"] += r["IDD"]

    # ── Write CSV ───────────────────────────────────────────────────────
    columns = [
        "insertion_id", "movie_title", "segment_idx", "timestamp_sec",
        "category_id", "category_name",
        "instruction_new", "instruction_before", "instruction_after",
        "all_active_instructions",
        "instruction_lang", "active_instr_count",
        "ICA", "SSS", "IDD", "SEC", "SE",
        "Latency_Before_sec", "Latency_After_sec",
        "TTFF_Before_sec", "TTFF_After_sec",
        "Transition_Smoothness",
        "ISR", "IFR", "User_Alignment", "Preference_Accuracy",
        "MSE", "nDCG",
        "text_before", "text_after", "ref_ad",
    ]

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_csv = Path(args.summary_csv) if args.summary_csv else output_csv.with_name(f"{output_csv.stem}_summary.csv")
    summary_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nUnified CSV: {output_csv}")

    summary_row = {"scope": "all_movies_all_records", **summary}
    summary_columns = [
        "scope",
        "total_movies", "total_records",
        "ICA_mean", "ICA_std",
        "SSS_mean", "SSS_std",
        "IDD_mean", "IDD_std",
        "SEC_mean", "SEC_std",
        "SE_mean", "SE_std",
        "Latency_Before_sec_mean", "Latency_Before_sec_std",
        "Latency_After_sec_mean", "Latency_After_sec_std",
        "TTFF_Before_sec_mean", "TTFF_Before_sec_std",
        "TTFF_After_sec_mean", "TTFF_After_sec_std",
        "Transition_Smoothness_mean", "Transition_Smoothness_std",
        "nDCG_mean", "nDCG_std",
        "ISR_pass_rate",
        "IFR_mean",
        "User_Alignment_mean",
        "Preference_After_pct",
        "MSE_mean",
    ]
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerow(summary_row)

    print(f"Summary CSV: {summary_csv}")

    # ── Print aggregate summary ─────────────────────────────────────────
    print(f"\n{'#'*60}")
    print(f"UNIFIED AGGREGATE SUMMARY (ALL {summary['total_records']} records, {summary['total_movies']} movies)")
    print(f"{'#'*60}")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    print(f"\n{'='*60}")
    print("PER-CATEGORY BREAKDOWN")
    print(f"{'='*60}")
    print(f"{'Category':<25s} {'Count':>5s} {'ICA':>8s} {'SSS':>8s} {'IDD':>8s}")
    print("-" * 56)
    for cat, data in sorted(cat_summary.items(), key=lambda x: -x[1]["IDD_sum"] / x[1]["count"]):
        n = data["count"]
        print(f"{cat:<25s} {n:>5d} {data['ICA_sum']/n:>8.4f} {data['SSS_sum']/n:>8.4f} {data['IDD_sum']/n:>8.4f}")

    print(f"\n{'='*60}")
    print("PER-MOVIE BREAKDOWN")
    print(f"{'='*60}")
    print(f"{'Movie':<35s} {'Rec':>4s} {'IDD':>8s} {'ICA':>8s} {'SSS':>8s}")
    print("-" * 61)
    movie_groups: Dict[str, List] = {}
    for r in rows:
        movie_groups.setdefault(r["movie_title"], []).append(r)
    for movie, recs in sorted(movie_groups.items(), key=lambda x: -np.mean([r["IDD"] for r in x[1]])):
        n = len(recs)
        print(f"{movie:<35s} {n:>4d} {np.mean([r['IDD']for r in recs]):>8.4f} {np.mean([r['ICA']for r in recs]):>8.4f} {np.mean([r['SSS']for f in recs]):>8.4f}")


if __name__ == "__main__":
    main()
