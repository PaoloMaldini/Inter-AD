#!/usr/bin/env python3
"""
merge_all_metrics.py — Combine traditional 7 metrics + new LLM metrics into one big table.

Reads from a run directory:
  - Traditional: all_eval_*.csv (Audio Overlap, Redundancy, Depth & Density, CRITIC,
                    BertScore, CIDEr, SPICE, R@k/N)
  - LLM-based:   eval_local_*.json (LLM-AD-eval, ISR, Decoupled)
Outputs a comprehensive CSV to /mnt/disk1new/ylz/newAD/eval_result/

Usage:
    python streamingAD/merge_all_metrics.py --run-dir batch_ad_output/run_20260530_020118
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
EVAL_RESULT_DIR = PROJECT_ROOT / "eval_result"
OUTPUT_BASE = PROJECT_ROOT / "batch_ad_output"

# ── Metric definitions ──────────────────────────────────────
TRADITIONAL_METRICS = [
    ("Audio Overlap", 0, 1, "↑"),
    ("Redundancy", 0, 1, "↓"),
    ("Depth & Density", 0, 1, "↑"),
    ("CRITIC", 0, 1, "↑"),
    ("BertScore", 0, 1, "↑"),
    ("CIDEr", 0, 1, "↑"),
    ("SPICE", 0, 1, "↑"),
    ("R@k/N", 0, 1, "↑"),
]

LLM_METRICS = [
    ("LLM-AD-eval", 1, 5, "↑"),
    ("ISR", 0, 1, "↑"),
    ("Decoupled-Style", 1, 5, "↑"),
    ("Decoupled-Content", 1, 5, "↑"),
    ("Decoupled-Overall", 1, 5, "↑"),
]

PER_MOVIE_COLS = [
    "movie", "imdbid", "num_gaps", "num_generated",
    "video_duration_sec", "preprocess_time_sec",
    "inference_total_time_sec", "total_time_sec", "time_per_video_sec",
]


def _find_most_recent(paths: List[Path]) -> Optional[Path]:
    return sorted(paths, key=lambda p: p.stat().st_mtime, reverse=True)[0] if paths else None


def _load_traditional(run_dir: Path) -> Optional[Dict[str, Dict[str, float]]]:
    """Load per-movie scores from all_eval_*.csv."""
    csv_files = sorted(run_dir.glob("all_eval_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    summary_csv = sorted(run_dir.glob("all_eval_*.summary.csv"), key=lambda p: p.stat().st_mtime, reverse=True)

    per_movie: Dict[str, Dict[str, float]] = {}
    if csv_files:
        with open(csv_files[0], encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                movie = row.get("movie", "")
                if not movie:
                    continue
                entry: Dict[str, float] = {}
                for col in PER_MOVIE_COLS:
                    val = row.get(col, 0)
                    try:
                        entry[col] = float(val)
                    except (ValueError, TypeError):
                        entry[col] = 0
                for name, _, _, _ in TRADITIONAL_METRICS:
                    val = row.get(name, "")
                    try:
                        entry[name] = float(val)
                    except (ValueError, TypeError):
                        entry[name] = 0.0
                per_movie[movie] = entry

    summary: Dict[str, float] = {}
    if summary_csv:
        with open(summary_csv[0], encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                metric = row.get("metric", "")
                try:
                    summary[metric] = float(row.get("mean_score", 0))
                except (ValueError, TypeError):
                    pass
    return per_movie if per_movie else None, summary if summary else None


def _load_llm_results(run_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Load LLM eval JSON results. Returns dict keyed by metric_name -> per_movie data."""
    results: Dict[str, Dict[str, Dict]] = {}
    for json_file in sorted(run_dir.glob("eval_local_*.json")):
        name_part = json_file.stem  # e.g. eval_local_llm_ad_20260531_123456
        with open(json_file, encoding="utf-8") as f:
            data = json.load(f)

        for entry in data:
            movie = entry.get("movie", "")
            metric = entry.get("metric", "")
            if not movie:
                continue

            if metric not in results:
                results[metric] = {}

            if metric == "LLM-AD-eval":
                results[metric][movie] = {
                    "score": entry.get("score", 0),
                    "num_samples": entry.get("num_samples", 0),
                }
            elif metric == "ISR":
                results[metric][movie] = {
                    "score": entry.get("score", 0),
                    "num_samples": entry.get("num_samples", 0),
                }
            elif metric == "Decoupled-Eval":
                results[metric][movie] = {
                    "style_score": entry.get("style_score", 0),
                    "content_score": entry.get("content_score", 0),
                    "overall_score": entry.get("overall_score", 0),
                    "num_samples": entry.get("num_samples", 0),
                }
    return results


def _latest_llm_timestamps(run_dir: Path) -> Dict[str, str]:
    """Get the latest timestamp for each LLM metric."""
    ts: Dict[str, str] = {}
    for json_file in sorted(run_dir.glob("eval_local_*.json")):
        name = json_file.stem
        for key in ["llm_ad", "isr", "decoupled"]:
            if key in name:
                parts = name.rsplit("_", 2)
                if len(parts) >= 2:
                    ts[key] = parts[-2] + "_" + parts[-1] if "_" in name else name.rsplit("_", 1)[-1]
    return ts


def main():
    parser = argparse.ArgumentParser(description="Merge all AD evaluation metrics into a big table")
    parser.add_argument("--run-dir", type=str, required=True,
                        help="Path to the run directory")
    parser.add_argument("--output", type=str, default="",
                        help="Output directory (default: /mnt/.../eval_result/)")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"Run dir not found: {run_dir}")
        sys.exit(1)

    run_name = run_dir.name
    out_dir = Path(args.output) if args.output else EVAL_RESULT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Run dir:  {run_dir}")
    print(f"Out dir:  {out_dir}")

    # ── Load traditional metrics ──
    trad_per_movie, trad_summary = _load_traditional(run_dir)
    if trad_per_movie:
        print(f"Traditional metrics: {len(trad_per_movie)} movies, {len(trad_summary)} metrics")
    else:
        print("No traditional eval CSV found (run run_all_eval.py first)")

    # ── Load LLM metrics ──
    llm_results = _load_llm_results(run_dir)
    llm_ts = _latest_llm_timestamps(run_dir)
    if llm_results:
        print(f"LLM metrics: {list(llm_results.keys())}")
    else:
        print("No LLM eval results found (run eval_local_llm.py first)")

    # ── Build master movie list ──
    all_movies: set = set()
    if trad_per_movie:
        all_movies.update(trad_per_movie.keys())
    for metric_data in llm_results.values():
        all_movies.update(metric_data.keys())
    all_movies = sorted(all_movies)

    if not all_movies:
        print("No movies found!")
        sys.exit(1)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── Per-movie CSV ──
    per_movie_path = out_dir / f"{run_name}_all_metrics_{ts}.csv"
    with open(per_movie_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)

        header = ["movie"]
        for name, lo, hi, direc in TRADITIONAL_METRICS:
            header.append(f"{name} [{lo}-{hi}] {direc}")
        for name, lo, hi, direc in LLM_METRICS:
            header.append(f"{name} [{lo}-{hi}] {direc}")
        header.extend(["num_gaps", "total_time_sec", "time_per_video_sec"])
        writer.writerow(header)

        for movie in all_movies:
            row = [movie]

            t = trad_per_movie.get(movie, {}) if trad_per_movie else {}
            for name, _, _, _ in TRADITIONAL_METRICS:
                row.append(round(t.get(name, 0), 4))

            l_ad = llm_results.get("LLM-AD-eval", {}).get(movie, {})
            l_isr = llm_results.get("ISR", {}).get(movie, {})
            l_dec = llm_results.get("Decoupled-Eval", {}).get(movie, {})

            row.append(round(l_ad.get("score", 0), 2))
            row.append(round(l_isr.get("score", 0), 2))
            row.append(round(l_dec.get("style_score", 0), 2))
            row.append(round(l_dec.get("content_score", 0), 2))
            row.append(round(l_dec.get("overall_score", 0), 2))

            row.append(int(t.get("num_gaps", 0)))
            row.append(round(t.get("total_time_sec", 0), 1))
            row.append(round(t.get("time_per_video_sec", 0), 4))

            writer.writerow(row)

    # ── Summary CSV ──
    summary_path = out_dir / f"{run_name}_all_metrics_{ts}.summary.csv"
    with open(summary_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "range", "direction", "mean_score", "num_movies"])

        for name, lo, hi, direc in TRADITIONAL_METRICS:
            val = trad_summary.get(name, 0) if trad_summary else 0
            writer.writerow([name, f"[{lo}, {hi}]", direc, round(val, 4), len(all_movies)])

        # Aggregate LLM metrics
        for metric_key, label, lo, hi, direc in [
            ("LLM-AD-eval", "LLM-AD-eval", 1, 5, "↑"),
            ("ISR", "ISR", 0, 1, "↑"),
        ]:
            scores = [v.get("score", 0) for v in llm_results.get(metric_key, {}).values()]
            avg = round(sum(scores) / len(scores), 4) if scores else 0
            writer.writerow([label, f"[{lo}, {hi}]", direc, avg, len(scores)])

        for key, label in [("style_score", "Decoupled-Style"), ("content_score", "Decoupled-Content"), ("overall_score", "Decoupled-Overall")]:
            scores = [v.get(key, 0) for v in llm_results.get("Decoupled-Eval", {}).values()]
            avg = round(sum(scores) / len(scores), 4) if scores else 0
            writer.writerow([label, f"[1, 5]", "↑", avg, len(scores)])

        # Timing aggregates
        if trad_per_movie:
            for tc, tc_label in [
                ("preprocess_time_sec", "preprocess (avg s)"),
                ("inference_total_time_sec", "inference (avg s)"),
                ("total_time_sec", "total (avg s)"),
                ("time_per_video_sec", "time_per_video_sec"),
            ]:
                vals = [v.get(tc, 0) for v in trad_per_movie.values() if v.get(tc, 0) > 0]
                if vals:
                    writer.writerow([tc_label, "", "", round(sum(vals) / len(vals), 2), len(vals)])

        total_gaps = sum(int(t.get("num_gaps", 0)) for t in trad_per_movie.values()) if trad_per_movie else 0
        writer.writerow(["total_gaps", "", "", total_gaps, "sum"])

    print(f"\nPer-movie table: {per_movie_path}")
    print(f"Summary table:   {summary_path}")

    # ── Pretty print summary ──
    print(f"\n{'─'*70}")
    print(f"  {run_name} — Full Evaluation Summary")
    print(f"{'─'*70}")
    print(f"  {'Metric':<30s} {'Score':>8s}  {'Range':>12s}")
    print(f"  {'─'*30} {'─'*8}  {'─'*12}")

    for name, lo, hi, direc in TRADITIONAL_METRICS:
        score = trad_summary.get(name, 0) if trad_summary else 0
        print(f"  {name:<30s} {score:>8.4f}  [{lo}-{hi}] {direc}")

    for metric_key, label, lo, hi, direc in [("LLM-AD-eval", "LLM-AD-eval", 1, 5, "↑"), ("ISR", "ISR", 0, 1, "↑")]:
        scores = [v.get("score", 0) for v in llm_results.get(metric_key, {}).values()]
        avg = round(sum(scores) / len(scores), 4) if scores else 0
        print(f"  {label:<30s} {avg:>8.4f}  [{lo}-{hi}] {direc}")

    for key, label in [("style_score", "Decoupled-Style"), ("content_score", "Decoupled-Content"), ("overall_score", "Decoupled-Overall")]:
        scores = [v.get(key, 0) for v in llm_results.get("Decoupled-Eval", {}).values()]
        avg = round(sum(scores) / len(scores), 4) if scores else 0
        print(f"  {label:<30s} {avg:>8.4f}  [1-5] ↑")
    print(f"{'─'*70}")


if __name__ == "__main__":
    main()
