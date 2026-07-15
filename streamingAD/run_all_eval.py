#!/usr/bin/env python3
"""
run_all_eval.py — Evaluation + CSV aggregation for a completed AD generation run.

Reads all _ad_output.json and _ref.json files from a run directory,
runs 8 metrics for each movie, saves _eval.json, and aggregates to CSV.

Usage:
    conda activate videollava
    python streamingAD/run_all_eval.py
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# ── Config — 改这里 ───────────────────────────────────────────
OUTPUT_BASE: str = "/mnt/disk1new/ylz/newAD/batch_ad_output"
RUN_DIR: str = "/mnt/disk1new/ylz/newAD/batch_ad_output/focusedad_run_20260531_021518"
# ──────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))


def _find_latest_run_dir() -> Path | None:
    base = Path(OUTPUT_BASE)
    run_dirs = sorted(
        [d for d in base.iterdir() if d.is_dir() and (d.name.startswith("run_") or d.name.startswith("focusedad_run_"))],
        reverse=True,
    )
    return run_dirs[0] if run_dirs else None


def eval_one_movie(ad_file: Path, ref_file: Path, eval_file: Path) -> Dict[str, Any] | None:
    from eval_ad import (
        AdEntry,
        eval_audio_overlap,
        eval_redundancy,
        eval_depth_density,
        eval_critic,
        eval_bertscore,
        eval_cider,
        eval_spice,
        eval_retrieval_r_at_k,
    )

    if not ad_file.exists():
        print(f"  SKIP: no ad_output file")
        return None

    with ad_file.open() as f:
        ad_data = json.load(f)

    entries = ad_data.get('ad_entries', [])
    if not entries:
        print(f"  SKIP: no entries")
        return None

    movie_name = ad_data.get('movie', ad_file.stem.replace('_ad_output', ''))
    imdbid = ad_data.get('imdbid', '')

    ad_entries = [
        AdEntry(
            gap_id=e['gap_id'],
            ad_text=e['ad_text'],
            gap_duration_sec=e['gap_duration_sec'],
            scene_index=e.get('scene_index', ''),
            location=e.get('location', ''),
            characters=e.get('characters', []),
            context_before=e.get('context_before', []),
            context_after=e.get('context_after', []),
        )
        for e in entries
    ]

    has_ref = ref_file.exists()
    ref_path = ref_file if has_ref else None

    print(f"    {len(entries)} ADs, ref={'yes' if has_ref else 'no'}")

    eval_results: Dict[str, Any] = {
        "movie": movie_name,
        "imdbid": imdbid,
        "num_gaps_total": ad_data.get('total_gaps', len(entries)),
        "num_generated": ad_data.get('generated_count', len(entries)),
        "video_duration_sec": ad_data.get('video_duration_sec', 0),
        "preprocess_time_sec": ad_data.get('preprocess_time_sec', 0),
        "inference_total_time_sec": ad_data.get('inference_total_time_sec', 0),
        "total_time_sec": ad_data.get('total_time_sec', 0),
        "time_per_video_sec": ad_data.get('time_per_video_sec', 0),
    }

    r = eval_audio_overlap(ad_entries)
    eval_results["Audio Overlap"] = {"score": r.score, "details": r.details}

    r = eval_redundancy(ad_entries)
    eval_results["Redundancy"] = {"score": r.score, "details": r.details}

    r = eval_depth_density(ad_entries)
    eval_results["Depth & Density"] = {"score": r.score, "details": r.details}

    r = eval_critic(ad_entries, use_ner=True)
    eval_results["CRITIC"] = {"score": r.score, "details": r.details}

    if has_ref:
        bs = eval_bertscore(ad_entries, ref_path=ref_path, use_context_as_ref=False)
        cd = eval_cider(ad_entries, ref_path=ref_path)
        sp = eval_spice(ad_entries, ref_path=ref_path)
        rk = eval_retrieval_r_at_k(ad_entries, ref_path=ref_path)
    else:
        bs = eval_bertscore(ad_entries, ref_path=None, use_context_as_ref=True)
        cd = eval_cider(ad_entries, ref_path=None)
        sp = eval_spice(ad_entries, ref_path=None)
        rk = eval_retrieval_r_at_k(ad_entries, ref_path=None)

    eval_results["BertScore"] = {"score": bs.score, "details": bs.details}
    eval_results["CIDEr"] = {"score": cd.score, "details": cd.details}
    eval_results["SPICE"] = {"score": sp.score, "details": sp.details}
    eval_results["R@k/N"] = {"score": rk.score, "details": rk.details}

    with eval_file.open('w', encoding='utf-8') as f:
        json.dump(eval_results, f, ensure_ascii=False, indent=2)

    return eval_results


def aggregate_to_csv(all_evals: List[Dict[str, Any]], csv_path: Path, run_ts: str) -> None:
    metrics = [
        "Audio Overlap", "Redundancy", "Depth & Density", "CRITIC",
        "BertScore", "CIDEr", "SPICE", "R@k/N",
    ]
    timing_cols = [
        "video_duration_sec", "preprocess_time_sec",
        "inference_total_time_sec", "total_time_sec", "time_per_video_sec",
    ]
    with csv_path.open('w', encoding='utf-8', newline='') as f:
        import csv
        writer = csv.writer(f)
        header = ["movie", "imdbid", "run_timestamp", "num_gaps", "num_generated"] + timing_cols + metrics
        writer.writerow(header)
        for ev in all_evals:
            row = [
                ev.get("movie", ""),
                ev.get("imdbid", ""),
                run_ts,
                ev.get("num_gaps_total", 0),
                ev.get("num_generated", 0),
            ]
            for tc in timing_cols:
                row.append(ev.get(tc, 0))
            for m in metrics:
                mr = ev.get(m, {})
                score = mr.get("score", "") if isinstance(mr, dict) else ""
                row.append(score)
            writer.writerow(row)

    agg_path = csv_path.with_suffix(".summary.csv")
    with agg_path.open('w', encoding='utf-8', newline='') as f:
        import csv
        writer = csv.writer(f)
        writer.writerow(["metric", "mean_score", "num_movies", "notes"])
        for m in metrics:
            valid = [
                ev[m]["score"]
                for ev in all_evals
                if m in ev and isinstance(ev[m], dict) and "score" in ev[m]
                and ev[m]["score"] is not None
            ]
            if valid:
                writer.writerow([m, round(sum(valid) / len(valid), 4), len(valid), ""])
        # Timing aggregates
        for tc in ["preprocess_time_sec", "inference_total_time_sec", "total_time_sec", "time_per_video_sec"]:
            valid_t = [ev.get(tc, 0) for ev in all_evals if ev.get(tc, 0) > 0]
            if valid_t:
                writer.writerow([tc, round(sum(valid_t) / len(valid_t), 2), len(valid_t), "mean"])
        total_gaps = sum(ev.get("num_gaps_total", 0) for ev in all_evals)
        total_generated = sum(ev.get("num_generated", 0) for ev in all_evals)
        writer.writerow(["total_gaps", total_gaps, "sum", ""])
        writer.writerow(["total_generated", total_generated, "sum", ""])

    print(f"\nPer-movie CSV: {csv_path}")
    print(f"Summary CSV:   {agg_path}")


def main():
    sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, "reconfigure") else None

    run_dir: Path
    if RUN_DIR.strip():
        run_dir = Path(RUN_DIR.strip())
    else:
        found = _find_latest_run_dir()
        if found is None:
            print("No run dir found. Set RUN_DIR in the script or make sure batch_ad_output has run_* dirs.")
            sys.exit(1)
        run_dir = found

    if not run_dir.exists():
        print(f"Run dir not found: {run_dir}")
        sys.exit(1)

    eval_log = run_dir / "eval.log"
    log_fh = eval_log.open('w', encoding='utf-8', buffering=1)

    class Tee:
        def __init__(self, fh):
            self.fh = fh
            self._orig = sys.stdout
        def write(self, s):
            self._orig.write(s)
            self.fh.write(s)
        def flush(self):
            self._orig.flush()
            self.fh.flush()

    sys.stdout = Tee(log_fh)

    print(f"Run dir: {run_dir}\n")

    ad_files = sorted(run_dir.glob("*_ad_output.json"))
    print(f"Found {len(ad_files)} AD output files")

    all_evals: List[Dict[str, Any]] = []
    success = 0
    fail = 0

    for ad_file in ad_files:
        movie_name = ad_file.stem.replace('_ad_output', '')
        ref_file = run_dir / f"{movie_name}_ref.json"
        eval_file = run_dir / f"{movie_name}_eval.json"

        print(f"  [{movie_name}]")
        try:
            result = eval_one_movie(ad_file, ref_file, eval_file)
            if result:
                all_evals.append(result)
                success += 1
            else:
                fail += 1
        except Exception as e:
            print(f"    ❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
            fail += 1

    print(f"\nEval complete: {success} success, {fail} fail")

    if all_evals:
        run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = run_dir / f"all_eval_{run_ts}.csv"
        aggregate_to_csv(all_evals, csv_path, run_ts)

    print(f"\nOutput: {run_dir}")


if __name__ == "__main__":
    main()
