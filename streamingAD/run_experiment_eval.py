#!/usr/bin/env python3
"""
run_experiment_eval.py — Evaluate interactive AD experiment results.

Compares baseline (no instructions) vs instructed (with instructions) AD outputs.
For each insertion point, computes metrics comparing segments BEFORE vs AFTER
the instruction, measuring whether the instruction took effect and persisted.

Metrics per insertion:
  - SSS (Semantic Shift Score): cosine distance between baseline and instructed ADs
  - ICA (Instruction-Content Alignment): similarity between instruction and instructed AD
  - Style Consistency: similarity of post-instruction ADs among themselves
  - Persistence Decay: whether effect diminishes over time after insertion

Aggregate metrics per movie:
  - Mean SSS, ICA, Style Consistency across all insertions
  - Overall instruction effectiveness score

Usage:
    conda activate videollava
    python streamingAD/run_experiment_eval.py \
        --baseline-dir /mnt/disk1new/ylz/newAD/batch_ad_output/run_20260601_streamad_full \
        --instructed-dir experiment_results/ \
        --output-dir experiment_eval_results/ \
        --gpu 0
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".hf_cache"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(PROJECT_ROOT / ".hf_cache" / "hub"))
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(PROJECT_ROOT / ".hf_cache" / "sentence_transformers"))

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_eval import EmbeddingModel


# ═══════════════════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_ad_output(json_path: Path) -> Dict[str, Any]:
    """Load an _ad_output.json file."""
    with json_path.open(encoding="utf-8") as f:
        return json.load(f)


def match_baseline_to_instructed(
    baseline_dir: Path,
    instructed_dir: Path,
) -> List[Tuple[str, Path, Path]]:
    """Match baseline and instructed AD outputs by movie title.

    Returns list of (movie_title, baseline_path, instructed_path).
    """
    # Index baseline files
    baseline_map: Dict[str, Path] = {}
    for f in sorted(baseline_dir.glob("*_ad_output.json")):
        data = load_ad_output(f)
        title = data.get("movie", data.get("movie_title", f.stem.replace("_ad_output", "")))
        baseline_map[title.lower().strip()] = f

    # Match instructed files
    matched: List[Tuple[str, Path, Path]] = []
    for f in sorted(instructed_dir.glob("*_ad_output.json")):
        data = load_ad_output(f)
        title = data.get("movie", data.get("movie_title", f.stem.replace("_ad_output", "")))
        key = title.lower().strip()
        if key in baseline_map:
            matched.append((title, baseline_map[key], f))
        else:
            print(f"  [warn] No baseline match for '{title}'")

    return matched


# ═══════════════════════════════════════════════════════════════════════════════
# Per-Insertion Metrics
# ═══════════════════════════════════════════════════════════════════════════════

def compute_insertion_metrics(
    embedder: EmbeddingModel,
    baseline_entries: List[Dict[str, Any]],
    instructed_entries: List[Dict[str, Any]],
    insertion_events: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    For each insertion point, compute metrics comparing the region after
    the insertion in baseline vs instructed.

    An insertion at segment_idx S means:
    - BEFORE region: segments [0, S)  (should be similar in both)
    - AFTER region:  segments [S, N)  (should differ if instruction works)
    """
    if not insertion_events:
        return []

    n_base = len(baseline_entries)
    n_inst = len(instructed_entries)
    n = min(n_base, n_inst)

    results: List[Dict[str, Any]] = []

    for event in insertion_events:
        seg_idx = event.get("segment_idx", 0)
        instr_text = event.get("instruction_text", "")
        category = event.get("category_name", "")
        insertion_id = event.get("insertion_id", 0)

        # Collect texts for AFTER region (from insertion point to end)
        base_after_texts: List[str] = []
        inst_after_texts: List[str] = []
        base_before_texts: List[str] = []
        inst_before_texts: List[str] = []

        for i in range(n):
            base_text = baseline_entries[i].get("ad_text", "")
            inst_text = instructed_entries[i].get("ad_text", "")

            if i >= seg_idx:
                if base_text.strip():
                    base_after_texts.append(base_text)
                if inst_text.strip():
                    inst_after_texts.append(inst_text)
            else:
                if base_text.strip():
                    base_before_texts.append(base_text)
                if inst_text.strip():
                    inst_before_texts.append(inst_text)

        if not inst_after_texts:
            continue

        # 1. SSS: Semantic Shift — mean cosine distance between baseline and instructed AFTER
        sss_scores: List[float] = []
        for bt, it in zip(base_after_texts, inst_after_texts):
            sim = embedder.cosine_sim(bt, it)
            sss_scores.append(1.0 - sim)  # distance = 1 - similarity
        sss = float(np.mean(sss_scores)) if sss_scores else 0.0

        # 2. ICA: Instruction-Content Alignment
        #    Mean similarity between instruction text and each instructed AD in AFTER region
        ica_scores: List[float] = []
        for it in inst_after_texts:
            sim = embedder.cosine_sim(instr_text, it)
            ica_scores.append(max(0.0, sim))
        ica = float(np.mean(ica_scores)) if ica_scores else 0.0

        # 3. Style Consistency: mean pairwise similarity among instructed AFTER texts
        style_consistency = 0.0
        if len(inst_after_texts) >= 2:
            embs = embedder.encode(inst_after_texts)
            sims: List[float] = []
            for i in range(len(embs)):
                for j in range(i + 1, len(embs)):
                    sims.append(float(np.dot(embs[i], embs[j])))
            style_consistency = float(np.mean(sims)) if sims else 0.0

        # 4. Persistence: compare early vs late AFTER segments
        #    If instruction persists, both early and late should differ from baseline similarly
        persistence_early = 0.0
        persistence_late = 0.0
        half = len(sss_scores) // 2
        if half > 0:
            persistence_early = float(np.mean(sss_scores[:half]))
            persistence_late = float(np.mean(sss_scores[half:]))

        # 5. Before-region similarity (should be ~0 if instruction hasn't affected pre-insertion)
        before_shift = 0.0
        if base_before_texts and inst_before_texts:
            before_sims: List[float] = []
            for bt, it in zip(base_before_texts[-5:], inst_before_texts[-5:]):
                before_sims.append(1.0 - embedder.cosine_sim(bt, it))
            before_shift = float(np.mean(before_sims)) if before_sims else 0.0

        results.append({
            "insertion_id": insertion_id,
            "segment_idx": seg_idx,
            "timestamp_sec": event.get("timestamp_sec", 0),
            "category": category,
            "instruction": instr_text,
            "num_after_segments": len(sss_scores),
            "SSS": round(sss, 4),
            "ICA": round(ica, 4),
            "Style_Consistency": round(style_consistency, 4),
            "Persistence_Early": round(persistence_early, 4),
            "Persistence_Late": round(persistence_late, 4),
            "Before_Shift": round(before_shift, 4),
        })

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Movie-Level Evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def eval_one_movie(
    embedder: EmbeddingModel,
    movie_title: str,
    baseline_path: Path,
    instructed_path: Path,
    output_dir: Path,
) -> Optional[Dict[str, Any]]:
    """Evaluate one movie: baseline vs instructed."""
    baseline_data = load_ad_output(baseline_path)
    instructed_data = load_ad_output(instructed_path)

    baseline_entries = baseline_data.get("ad_entries", [])
    instructed_entries = instructed_data.get("ad_entries", [])
    insertion_events = instructed_data.get("insertion_events", [])

    if not instructed_entries:
        print(f"  [skip] No instructed AD entries")
        return None

    n = min(len(baseline_entries), len(instructed_entries))

    # Global metrics: full-movie SSS (baseline vs instructed, all segments)
    all_base = [e.get("ad_text", "") for e in baseline_entries[:n]]
    all_inst = [e.get("ad_text", "") for e in instructed_entries[:n]]

    global_sss_scores: List[float] = []
    for bt, it in zip(all_base, all_inst):
        if bt.strip() and it.strip():
            global_sss_scores.append(1.0 - embedder.cosine_sim(bt, it))
    global_sss = float(np.mean(global_sss_scores)) if global_sss_scores else 0.0

    # Per-insertion metrics
    insertion_metrics = compute_insertion_metrics(
        embedder, baseline_entries, instructed_entries, insertion_events,
    )

    # Aggregate insertion metrics
    agg: Dict[str, Any] = {
        "movie": movie_title,
        "num_segments_baseline": len(baseline_entries),
        "num_segments_instructed": len(instructed_entries),
        "num_segments_compared": n,
        "num_insertions": len(insertion_events),
        "Global_SSS": round(global_sss, 4),
    }

    metric_keys = ["SSS", "ICA", "Style_Consistency", "Persistence_Early",
                    "Persistence_Late", "Before_Shift"]
    for mk in metric_keys:
        vals = [r[mk] for r in insertion_metrics if r.get(mk) is not None]
        agg[f"mean_{mk}"] = round(float(np.mean(vals)), 4) if vals else None
        agg[f"std_{mk}"] = round(float(np.std(vals)), 4) if vals else None

    # Save per-movie result
    movie_output = output_dir / f"{movie_title.replace(' ', '_')}_experiment_eval.json"
    with movie_output.open("w", encoding="utf-8") as f:
        json.dump({
            "movie": movie_title,
            "aggregate": agg,
            "per_insertion": insertion_metrics,
            "insertion_events": insertion_events,
        }, f, ensure_ascii=False, indent=2)

    return agg


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Evaluate interactive AD experiments")
    parser.add_argument("--baseline-dir", required=True,
                        help="Directory with baseline _ad_output.json files")
    parser.add_argument("--instructed-dir", required=True,
                        help="Directory with instructed _ad_output.json files")
    parser.add_argument("--output-dir", default="experiment_eval_results/",
                        help="Output directory")
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2",
                        help="Embedding model name")
    parser.add_argument("--device", default="cuda", help="Device for embedding model")
    args = parser.parse_args()

    baseline_dir = Path(args.baseline_dir)
    instructed_dir = Path(args.instructed_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("EXPERIMENT EVALUATION: Baseline vs Instructed")
    print("=" * 60)

    # Match movies
    matched = match_baseline_to_instructed(baseline_dir, instructed_dir)
    print(f"\nMatched {len(matched)} movies")
    if not matched:
        print("No movies matched. Check directories.")
        return

    # Load embedding model
    print(f"\nLoading embedding model: {args.embedding_model}")
    embedder = EmbeddingModel(model_name=args.embedding_model, device=args.device)

    # Evaluate each movie
    all_results: List[Dict[str, Any]] = []
    for title, base_path, inst_path in tqdm(matched, desc="Movies"):
        print(f"\n{'─'*50}")
        print(f"  {title}")
        try:
            result = eval_one_movie(embedder, title, base_path, inst_path, output_dir)
            if result:
                all_results.append(result)
                print(f"  SSS={result.get('mean_SSS', 'N/A')}, "
                      f"ICA={result.get('mean_ICA', 'N/A')}, "
                      f"Insertions={result['num_insertions']}")
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Aggregate CSV
    if all_results:
        csv_path = output_dir / f"experiment_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        metric_keys = ["SSS", "ICA", "Style_Consistency", "Persistence_Early",
                        "Persistence_Late", "Before_Shift"]

        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            header = ["movie", "num_segments_compared", "num_insertions", "Global_SSS"]
            for mk in metric_keys:
                header.extend([f"mean_{mk}", f"std_{mk}"])
            writer.writerow(header)

            for r in all_results:
                row = [r["movie"], r["num_segments_compared"], r["num_insertions"],
                       r["Global_SSS"]]
                for mk in metric_keys:
                    row.append(r.get(f"mean_{mk}", ""))
                    row.append(r.get(f"std_{mk}", ""))
                writer.writerow(row)

        # Summary
        summary_path = csv_path.with_suffix(".summary.csv")
        with summary_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "mean", "std", "num_movies"])

            global_sss_vals = [r["Global_SSS"] for r in all_results]
            writer.writerow(["Global_SSS",
                             round(float(np.mean(global_sss_vals)), 4),
                             round(float(np.std(global_sss_vals)), 4),
                             len(global_sss_vals)])

            for mk in metric_keys:
                vals = [r[f"mean_{mk}"] for r in all_results
                        if r.get(f"mean_{mk}") is not None]
                if vals:
                    writer.writerow([f"mean_{mk}",
                                     round(float(np.mean(vals)), 4),
                                     round(float(np.std(vals)), 4),
                                     len(vals)])

        print(f"\n{'='*60}")
        print(f"RESULTS SAVED:")
        print(f"  Per-movie:  {output_dir}/")
        print(f"  CSV:        {csv_path}")
        print(f"  Summary:    {summary_path}")
        print(f"{'='*60}")

        print(f"\nSummary:")
        print(f"  Global SSS (full-movie shift): {float(np.mean(global_sss_vals)):.4f} "
              f"± {float(np.std(global_sss_vals)):.4f}")
        for mk in metric_keys:
            vals = [r[f"mean_{mk}"] for r in all_results
                    if r.get(f"mean_{mk}") is not None]
            if vals:
                print(f"  {mk:25s}: {float(np.mean(vals)):.4f} ± {float(np.std(vals)):.4f}")


if __name__ == "__main__":
    main()
