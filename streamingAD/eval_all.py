#!/usr/bin/env python3
"""
eval_all.py — Unified no-GT evaluation for Audio Description generation.

Metrics (all reference-free):
  Traditional:  Audio Overlap, Redundancy, Depth & Density, CRITIC, BertScore (context)
  LLM Judge:    ISR (Instruction Success Rate), Decoupled-Eval (Style + Content)

Usage:
    # 1. Edit the config section below (RUN_DIR, LLM_MODEL_PATH, GPU)
    # 2. Run:
    conda activate focusedad
    python streamingAD/eval_all.py

    # Or override via command line:
    python streamingAD/eval_all.py \
        --run-dir batch_ad_output/focusedad_run_20260531_021518 \
        --model /mnt/disk5new/gcc/models/Qwen2.5-7B-Instruct \
        --gpu 1
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))


# ═══════════════════════════════════════════════════════════════════
# CONFIG — 改这里 ─────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════

RUN_DIR: str = "/mnt/disk1new/ylz/newAD/batch_ad_output/focusedad_run_20260531_021518"
LLM_MODEL_PATH: str = "/mnt/disk5new/gcc/models/Qwen2.5-7B-Instruct"
GPU: int = 1
OUTPUT_DIR: str = "/mnt/disk1new/ylz/newAD/eval_result"

# ═══════════════════════════════════════════════════════════════════


TRADITIONAL_METRICS = [
    ("Audio Overlap",    0, 1, "higher"),
    ("Redundancy",       0, 1, "lower"),
    ("Depth & Density",  0, 1, "higher"),
    ("CRITIC",           0, 1, "higher"),
    ("BertScore",        0, 1, "higher"),
]

LLM_METRICS = [
    ("ISR",               0, 1, "higher"),
    ("Decoupled-Style",   1, 5, "higher"),
    ("Decoupled-Content", 1, 5, "higher"),
    ("Decoupled-Overall", 1, 5, "higher"),
]

TIMING_COLS = [
    "num_gaps", "num_generated", "video_duration_sec",
    "preprocess_time_sec", "inference_total_time_sec",
    "total_time_sec", "time_per_video_sec",
]


# ═══════════════════════════════════════════════════════════════════
# LLM Judge
# ═══════════════════════════════════════════════════════════════════

class LocalLLMJudge:
    def __init__(self, model_path: str, gpu: int = 0):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"
        print(f"[LLM Judge] Loading {model_path} on {self.device} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(self.device)
        self.model.eval()
        print(f"[LLM Judge] Model loaded.")

    def generate(self, prompt: str, max_new_tokens: int = 256, temperature: float = 0.0) -> str:
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature > 0 else 1.0,
                do_sample=(temperature > 0),
            )
        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
        )
        return response.strip()

    def score_1_to_5(self, prompt: str) -> Tuple[int, str]:
        raw = self.generate(prompt, max_new_tokens=128, temperature=0.0)
        score = 3
        for pattern in [r'\bscore\s*[:=]?\s*(\d)', r'\brating\s*[:=]?\s*(\d)', r'\b(\d)\s*/\s*5']:
            m = re.search(pattern, raw, re.IGNORECASE)
            if m:
                score = int(m.group(1))
                break
        return min(max(score, 1), 5), raw


# ═══════════════════════════════════════════════════════════════════
# LLM Metrics
# ═══════════════════════════════════════════════════════════════════

def eval_isr(
    entries: List[Dict[str, Any]],
    judge: LocalLLMJudge,
    instruction: Optional[str] = None,
) -> Dict[str, Any]:
    DEFAULT_INSTRUCTION = (
        "Describe what is happening in this clip concisely. "
        "Focus on visible actions, movements, and expressions. "
        "If character names are mentioned in the context, use them "
        "(e.g. 'Don Vito Corleone walks...' not 'A man walks...'). "
        "Do not quote dialogue."
    )
    instr = (instruction or DEFAULT_INSTRUCTION).strip()

    PROMPT = """You are evaluating whether a generated Audio Description (AD) 
follows a specific instruction.

INSTRUCTION: {instruction}

GENERATED AD: {ad_text}

Score 1 (fail) or 2 (pass):
  2 = The AD fully follows the instruction
  1 = The AD violates the instruction (e.g., uses names when told not to, 
      describes off-screen content, uses dialogue, etc.)

Output format: Score: X  (one line, then brief reason)
"""
    scores = []
    for entry in tqdm(entries, desc="  ISR", leave=False):
        ad_text = entry.get("ad_text", "")
        if not ad_text:
            continue
        s, _ = judge.score_1_to_5(PROMPT.format(instruction=instr, ad_text=ad_text))
        scores.append(1.0 if s >= 2 else 0.0)

    rate = round(sum(scores) / len(scores), 4) if scores else 0
    return {"score": rate, "num_samples": len(scores)}


def eval_decoupled(
    entries: List[Dict[str, Any]],
    judge: LocalLLMJudge,
) -> Dict[str, Any]:
    STYLE_PROMPT = """You are evaluating the STYLE/TONE of an Audio Description (AD) sentence.
Rate ONLY the style aspects: fluency, vividness, appropriate AD tone (concise, 
present-tense, visual-only descriptions).

AD TEXT: {ad_text}

Rate style 1-5:
  5 = Excellent AD style: vivid, concise, fluent, proper present-tense
  4 = Good style, minor issues
  3 = Acceptable
  2 = Poor style (e.g., too wordy, past tense, evaluative language)
  1 = Very poor (unreadable, wrong format)

Output format: Score: X  (one line, then brief reason)
"""

    CONTENT_PROMPT = """You are evaluating the FACTUAL ACCURACY of an Audio Description (AD) sentence.
Given the surrounding dialogue context, rate whether the described visual actions/objects 
are plausible for this moment in the film.

CONTEXT BEFORE: {context_before}
AD TEXT:        {ad_text}
CONTEXT AFTER:  {context_after}

Rate factual plausibility 1-5:
  5 = Fully plausible, matches context perfectly
  4 = Likely correct, minor uncertainty
  3 = Uncertain / neutral
  2 = Likely wrong or implausible
  1 = Clearly hallucinated or contradictory

Output format: Score: X  (one line, then brief reason)
"""

    style_scores = []
    content_scores = []
    for entry in tqdm(entries, desc="  Decoupled", leave=False):
        ad_text = entry.get("ad_text", "")
        if not ad_text:
            continue
        cb = str(entry.get("context_before", ""))[:200]
        ca = str(entry.get("context_after", ""))[:200]

        ss, _ = judge.score_1_to_5(STYLE_PROMPT.format(ad_text=ad_text))
        cs, _ = judge.score_1_to_5(CONTENT_PROMPT.format(
            context_before=cb, ad_text=ad_text, context_after=ca,
        ))
        style_scores.append(ss)
        content_scores.append(cs)

    avg_s = round(sum(style_scores) / len(style_scores), 3) if style_scores else 0
    avg_c = round(sum(content_scores) / len(content_scores), 3) if content_scores else 0
    return {
        "style_score": avg_s,
        "content_score": avg_c,
        "overall_score": round((avg_s + avg_c) / 2, 3),
        "num_samples": len(style_scores),
    }


# ═══════════════════════════════════════════════════════════════════
# Traditional metrics (imported from eval_ad)
# ═══════════════════════════════════════════════════════════════════

def _build_ad_entries(raw_entries: List[Dict[str, Any]]) -> List:
    from eval_ad import AdEntry
    return [
        AdEntry(
            gap_id=e.get("gap_id", 0),
            ad_text=e.get("ad_text", ""),
            gap_duration_sec=float(e.get("gap_duration_sec", 0)),
            scene_index=str(e.get("scene_index", "")),
            location=str(e.get("location", "")),
            characters=[str(c).strip() for c in e.get("characters", []) if str(c).strip()],
            context_before=[str(s) for s in e.get("context_before", [])],
            context_after=[str(s) for s in e.get("context_after", [])],
        )
        for e in raw_entries
    ]


def compute_traditional(ad_entries: List) -> Dict[str, Any]:
    from eval_ad import (
        eval_audio_overlap,
        eval_redundancy,
        eval_depth_density,
        eval_critic,
        eval_bertscore,
    )
    results: Dict[str, Any] = {}
    tasks = [
        (eval_audio_overlap, "Audio Overlap"),
        (eval_redundancy, "Redundancy"),
        (eval_depth_density, "Depth & Density"),
        (eval_critic, "CRITIC"),
    ]
    for func, name in tqdm(tasks, desc="  Traditional", leave=False):
        r = func(ad_entries)
        results[name] = {"score": r.score, "details": r.details}

    r = eval_bertscore(ad_entries, ref_path=None, use_context_as_ref=True)
    results["BertScore"] = {"score": r.score, "details": r.details}
    return results


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def find_run_dir(run_dir_arg: str) -> Path:
    if run_dir_arg.strip():
        return Path(run_dir_arg.strip())
    base = PROJECT_ROOT / "batch_ad_output"
    run_dirs = sorted(
        [d for d in base.iterdir()
         if d.is_dir() and (d.name.startswith("run_") or d.name.startswith("focusedad_run_"))],
        reverse=True,
    )
    if run_dirs:
        return run_dirs[0]
    print("No run dir found. Set RUN_DIR in script or use --run-dir.")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Unified no-GT AD evaluation")
    parser.add_argument("--run-dir", type=str, default=RUN_DIR,
                        help="Path to run directory")
    parser.add_argument("--model", type=str, default=LLM_MODEL_PATH,
                        help="Path to LLM model for judging")
    parser.add_argument("--gpu", type=int, default=GPU,
                        help="GPU device ID")
    parser.add_argument("--output", type=str, default=OUTPUT_DIR,
                        help="Output directory for results")
    parser.add_argument("--skip-llm", action="store_true",
                        help="Skip LLM-based metrics (ISR, Decoupled)")
    parser.add_argument("--movie", type=str, default="",
                        help="Only evaluate this movie (partial name match)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from previous eval (skip already-evaluated movies)")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"Run dir not found: {run_dir}")
        sys.exit(1)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = run_dir.name

    # ── Resume: load previous results ───────────────────────
    resumed_rows: List[Dict[str, Any]] = []
    skip_movies: set = set()
    if args.resume:
        prev_jsons = sorted(out_dir.glob(f"{run_name}_eval_*.json"), reverse=True)
        if prev_jsons:
            print(f"[Resume] Found previous eval: {prev_jsons[0].name}")
            with open(prev_jsons[0], encoding="utf-8") as f:
                prev_data = json.load(f)
            for r in prev_data:
                skip_movies.add(r["movie"])
                resumed_rows.append(r)
            print(f"[Resume] Skipping {len(skip_movies)} already-evaluated movies")
        else:
            print("[Resume] No previous eval found, starting fresh")

    print(f"{'='*60}")
    print(f"Run dir:   {run_dir}")
    print(f"Output:    {out_dir}")
    print(f"LLM model: {args.model} (GPU {args.gpu})")
    print(f"LLM skip:  {args.skip_llm}")
    print(f"{'='*60}\n")

    ad_files = sorted(run_dir.glob("*_ad_output.json"))
    if not ad_files:
        print("No _ad_output.json files found.")
        sys.exit(1)
    print(f"Found {len(ad_files)} movies\n")

    judge = None
    if not args.skip_llm:
        judge = LocalLLMJudge(model_path=args.model, gpu=args.gpu)

    all_rows: List[Dict[str, Any]] = []

    movie_filter = args.movie.strip().lower()

    for idx, ad_file in enumerate(ad_files):
        movie = ad_file.stem.replace("_ad_output", "")

        if movie_filter and movie_filter not in movie.lower():
            continue

        if movie in skip_movies:
            continue

        print(f"[{idx+1}/{len(ad_files)}] {movie}")

        with open(ad_file, encoding="utf-8") as f:
            data = json.load(f)

        raw_entries = data.get("ad_entries", [])
        if not raw_entries:
            print(f"  SKIP: no entries")
            continue

        print(f"  {len(raw_entries)} entries")

        row: Dict[str, Any] = {
            "movie": movie,
            "imdbid": data.get("imdbid", ""),
        }
        for tc in TIMING_COLS:
            row[tc] = data.get(tc, 0)
        row["num_gaps"] = data.get("total_gaps", len(raw_entries))
        row["num_generated"] = data.get("generated_count", len(raw_entries))

        # ── Traditional metrics ──
        t0 = time.time()
        ad_entries = _build_ad_entries(raw_entries)
        trad = compute_traditional(ad_entries)
        for name in [m[0] for m in TRADITIONAL_METRICS]:
            row[name] = trad.get(name, {}).get("score", 0)
        t1 = time.time()
        print(f"  Traditional: {t1-t0:.1f}s")

        # ── LLM metrics ──
        if judge is not None:
            t0 = time.time()
            isr = eval_isr(raw_entries, judge)
            row["ISR"] = isr["score"]

            dec = eval_decoupled(raw_entries, judge)
            row["Decoupled-Style"] = dec["style_score"]
            row["Decoupled-Content"] = dec["content_score"]
            row["Decoupled-Overall"] = dec["overall_score"]
            t1 = time.time()
            print(f"  LLM: {t1-t0:.1f}s  ISR={isr['score']:.3f}  Decoupled=(s={dec['style_score']}, c={dec['content_score']})")

        for name in [m[0] for m in TRADITIONAL_METRICS]:
            print(f"    {name}: {row.get(name, 0):.4f}")
        if judge is not None:
            for name in [m[0] for m in LLM_METRICS]:
                print(f"    {name}: {row.get(name, 0):.4f}")

        all_rows.append(row)

    # ── Merge resumed + new rows ─────────────────────────────
    all_rows = resumed_rows + all_rows

    if not all_rows:
        print("No data to save.")
        sys.exit(1)

    # ── Per-movie CSV ──
    per_csv = out_dir / f"{run_name}_eval_{ts}.csv"
    metric_names = [m[0] for m in TRADITIONAL_METRICS]
    if judge is not None:
        metric_names += [m[0] for m in LLM_METRICS]

    with open(per_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        header = ["movie", "imdbid"] + TIMING_COLS + metric_names
        writer.writerow(header)
        for row in all_rows:
            writer.writerow([
                row.get("movie", ""),
                row.get("imdbid", ""),
            ] + [row.get(tc, 0) for tc in TIMING_COLS]
              + [row.get(m, 0) for m in metric_names])

    # ── Summary CSV ──
    sum_csv = out_dir / f"{run_name}_eval_{ts}.summary.csv"
    with open(sum_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "range", "direction", "mean_score", "num_movies", "std"])

        all_defs = TRADITIONAL_METRICS + LLM_METRICS
        for name, lo, hi, direc in all_defs:
            vals = [row.get(name, 0) for row in all_rows if row.get(name, 0) != 0]
            if vals:
                mean_v = round(np.mean(vals), 4)
                std_v = round(np.std(vals), 4)
            else:
                mean_v, std_v = 0, 0
            writer.writerow([name, f"[{lo}, {hi}]", direc, mean_v, len(vals), std_v])

        for tc in ["preprocess_time_sec", "inference_total_time_sec", "total_time_sec", "time_per_video_sec"]:
            vals = [row.get(tc, 0) for row in all_rows]
            if vals:
                writer.writerow([tc, "", "", round(np.mean(vals), 2), len(vals), round(np.std(vals), 2)])

        total_gaps = sum(row.get("num_gaps", 0) for row in all_rows)
        total_gen = sum(row.get("num_generated", 0) for row in all_rows)
        writer.writerow(["total_gaps (sum)", "", "", total_gaps, "", ""])
        writer.writerow(["total_generated (sum)", "", "", total_gen, "", ""])

    # ── Detail JSON ──
    detail_json = out_dir / f"{run_name}_eval_{ts}.json"
    with open(detail_json, "w", encoding="utf-8") as f:
        json.dump(all_rows, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"Output saved to {out_dir}/")
    print(f"  Per-movie:  {per_csv.name}")
    print(f"  Summary:    {sum_csv.name}")
    print(f"  Detail:     {detail_json.name}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
