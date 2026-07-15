#!/usr/bin/env python3
"""
eval_gt_all.py — GT-based + LLM + CRITIC + Time evaluation for Audio Description.

Metrics:
  GT-based:    CIDEr, BLEU-1, BLEU-4, ROUGE-L, METEOR, SPICE, BERTScore (vs GT)
  LLM Judge:   ISR, Decoupled-Style, Decoupled-Content, Decoupled-Overall
  CRITIC:      character co-reference accuracy
  Time:        preprocess, inference, total, time_per_video

Usage:
    conda activate videollava && python streamingAD/eval_gt_all.py \
        --run-dir compare/didemo \
        --model /mnt/disk5new/gcc/models/Qwen2.5-7B-Instruct \
        --gpu 0

    # Skip LLM metrics:
    python streamingAD/eval_gt_all.py --run-dir compare/didemo --skip-llm

    # Resume from previous:
    python streamingAD/eval_gt_all.py --run-dir compare/didemo --resume
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
# Metric definitions
# ═══════════════════════════════════════════════════════════════════

GT_METRICS = [
    ("CIDEr",        0, 100, "higher"),  # scale varies
    ("BLEU-1",       0, 1,   "higher"),
    ("BLEU-4",       0, 1,   "higher"),
    ("ROUGE-L",      0, 1,   "higher"),
    ("METEOR",       0, 1,   "higher"),
    ("SPICE",        0, 1,   "higher"),
    ("BERTScore",    0, 1,   "higher"),
    ("R@1",          0, 100, "higher"),
    ("R@5",          0, 100, "higher"),
]

LLM_METRICS = [
    ("ISR",               0, 1, "higher"),
    ("Decoupled-Style",   1, 5, "higher"),
    ("Decoupled-Content", 1, 5, "higher"),
    ("Decoupled-Overall", 1, 5, "higher"),
]

OTHER_METRICS = [
    ("CRITIC",           0, 1, "higher"),
]

TIMING_COLS = [
    "num_gaps", "num_generated", "video_duration_sec",
    "preprocess_time_sec", "inference_total_time_sec",
    "total_time_sec", "time_per_video_sec",
]


# ═══════════════════════════════════════════════════════════════════
# GT-based metrics using pycocoevalcap
# ═══════════════════════════════════════════════════════════════════

def compute_pycoco_metrics(
    candidates: Dict[int, str],
    references: Dict[int, List[str]],
) -> Dict[str, float]:
    """Compute BLEU, ROUGE-L, METEOR, CIDEr, SPICE using pycocoevalcap."""
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.rouge.rouge import Rouge
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.spice.spice import Spice

    # pycocoevalcap expects string keys
    gts = {str(k): v for k, v in references.items()}
    res = {str(k): [v] for k, v in candidates.items() if k in references}

    results = {}

    # BLEU (1-4)
    try:
        bleu_scorer = Bleu(4)
        bleu_scores, _ = bleu_scorer.compute_score(gts, res)
        results["BLEU-1"] = float(bleu_scores[0])
        results["BLEU-4"] = float(bleu_scores[3])
    except Exception as e:
        print(f"    [WARN] BLEU failed: {e}")
        results["BLEU-1"] = 0.0
        results["BLEU-4"] = 0.0

    # ROUGE-L
    try:
        rouge_scorer = Rouge()
        rouge_scores, _ = rouge_scorer.compute_score(gts, res)
        results["ROUGE-L"] = float(rouge_scores)
    except Exception as e:
        print(f"    [WARN] ROUGE-L failed: {e}")
        results["ROUGE-L"] = 0.0

    # METEOR — sanitize text (newlines break Java pipe protocol)
    try:
        from pycocoevalcap.meteor.meteor import Meteor
        _san = lambda s: s.replace('\n', ' ').replace('\r', ' ').replace('|||', '').replace('  ', ' ')
        m_gts = {k: [_san(v[0])] for k, v in gts.items()}
        m_res = {k: [_san(v[0])] for k, v in res.items()}
        meteor_scorer = Meteor()
        meteor_scores, _ = meteor_scorer.compute_score(m_gts, m_res)
        results["METEOR"] = float(meteor_scores)
    except Exception as e:
        print(f"    [WARN] METEOR failed: {e}")
        results["METEOR"] = 0.0

    # CIDEr
    try:
        cider_scorer = Cider()
        cider_scores, _ = cider_scorer.compute_score(gts, res)
        results["CIDEr"] = float(cider_scores)
    except Exception as e:
        print(f"    [WARN] CIDEr failed: {e}")
        results["CIDEr"] = 0.0

    # SPICE
    try:
        spice_scorer = Spice()
        spice_scores, _ = spice_scorer.compute_score(gts, res)
        results["SPICE"] = float(spice_scores)
    except Exception as e:
        print(f"    [WARN] SPICE failed: {e}")
        results["SPICE"] = 0.0

    return results


def compute_bertscore_vs_gt(
    candidates: List[str],
    references: List[str],
    model_path: str = "roberta-large",
) -> float:
    """Compute BERTScore (F1) between candidates and GT references."""
    from bert_score import score as bert_score_fn

    if not candidates or not references:
        return 0.0

    P, R, F1 = bert_score_fn(
        candidates, references,
        model_type=model_path,
        verbose=False,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    return float(F1.mean())


# ═══════════════════════════════════════════════════════════════════
# R@k retrieval metrics (from CondensedMovies/metric.py)
# ═══════════════════════════════════════════════════════════════════

def compute_retrieval_metrics(
    gt_texts: List[str],
    pred_texts: List[str],
) -> Dict[str, float]:
    """Compute R@1, R@5 retrieval metrics.

    For each GT text (query), rank all predicted texts by cosine similarity
    and check if the aligned prediction is in the top-k.

    Follows the v2t_metrics logic from CondensedMovies/model/metric.py:
    per-query rank computation with tie-breaking by averaging.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    n = min(len(gt_texts), len(pred_texts))
    if n < 2:
        return {"R@1": 0.0, "R@5": 0.0}

    gt_texts = gt_texts[:n]
    pred_texts = pred_texts[:n]

    # Build TF-IDF on all texts together, then split
    all_texts = list(gt_texts) + list(pred_texts)
    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
    tfidf = vectorizer.fit_transform(all_texts)
    gt_embs = tfidf[:n].toarray()
    pred_embs = tfidf[n:].toarray()

    # cosine similarity matrix (n x n), sims[i][j] = sim(gt_i, pred_j)
    sims = cosine_similarity(gt_embs, pred_embs)

    # Per-query rank: for each gt_i, rank all predictions by distance,
    # find rank of the aligned pred_i (diagonal). Break ties by averaging.
    dists = -sims
    query_ranks = []
    for i in range(n):
        row_dists = dists[i]
        sorted_dists = np.sort(row_dists)
        gt_dist = row_dists[i]
        ranks = np.where((sorted_dists - gt_dist) == 0)[0]
        rank = float(ranks.mean())  # average over tied positions
        query_ranks.append(rank)
    query_ranks = np.array(query_ranks)

    r1 = 100.0 * float(np.sum(query_ranks == 0)) / n
    r5 = 100.0 * float(np.sum(query_ranks < 5)) / n
    return {"R@1": round(r1, 2), "R@5": round(r5, 2)}


# ═══════════════════════════════════════════════════════════════════
# CRITIC metric (from eval_ad.py)
# ═══════════════════════════════════════════════════════════════════

def compute_critic(entries: list) -> Optional[float]:
    """Compute CRITIC: character co-reference accuracy."""
    from eval_ad import eval_critic as _eval_critic
    from eval_ad import AdEntry as _AdEntry

    ad_entries = []
    for e in entries:
        ad_entries.append(_AdEntry(
            gap_id=e.get("gap_id", 0),
            ad_text=e.get("ad_text", ""),
            gap_duration_sec=float(e.get("gap_duration_sec", 0)),
            scene_index=str(e.get("scene_index", "")),
            location=str(e.get("location", "")),
            characters=[str(c).strip() for c in e.get("characters", []) if str(c).strip()],
            context_before=[str(s) for s in e.get("context_before", [])],
            context_after=[str(s) for s in e.get("context_after", [])],
        ))
    result = _eval_critic(ad_entries, use_ner=True)
    return float(result.score) if result.score is not None else None


# ═══════════════════════════════════════════════════════════════════
# LLM Judge (from eval_all.py)
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
        print("[LLM Judge] Model loaded.")

    def generate(self, prompt: str, max_new_tokens: int = 256, temperature: float = 0.0) -> str:
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                temperature=temperature if temperature > 0 else None,
                do_sample=temperature > 0,
            )
        generated = outputs[0][inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()


def eval_isr(llm: LocalLLMJudge, ad_text: str, instruction: str) -> float:
    prompt = (
        f"Rate whether the following audio description successfully fulfills "
        f"the instruction on a scale of 0 to 1.\n\n"
        f"Instruction: {instruction}\n"
        f"Audio Description: {ad_text}\n\n"
        f"Answer with a single number between 0 and 1:"
    )
    try:
        response = llm.generate(prompt, max_new_tokens=16)
        match = re.search(r"0?\.\d+|1\.0*", response)
        return float(match.group()) if match else 0.0
    except Exception:
        return 0.0


def eval_decoupled(llm: LocalLLMJudge, ad_text: str, instruction: str) -> Dict[str, float]:
    prompt = (
        f"Evaluate the following audio description on three dimensions "
        f"(1-5 each):\n\n"
        f"Instruction: {instruction}\n"
        f"Audio Description: {ad_text}\n\n"
        f"Rate each dimension:\n"
        f"Style (1-5): How well does the AD style match expected AD conventions?\n"
        f"Content (1-5): How accurately does the AD describe the visual content?\n"
        f"Overall (1-5): Overall quality?\n\n"
        f"Answer format: Style=X Content=Y Overall=Z"
    )
    result = {"Style": 3.0, "Content": 3.0, "Overall": 3.0}
    try:
        response = llm.generate(prompt, max_new_tokens=64)
        for key in ["Style", "Content", "Overall"]:
            match = re.search(rf"{key}\s*=\s*(\d)", response)
            if match:
                result[key] = float(match.group(1))
    except Exception:
        pass
    return result


# ═══════════════════════════════════════════════════════════════════
# Movie-level evaluation
# ═══════════════════════════════════════════════════════════════════

def evaluate_movie(
    ad_file: Path,
    llm_judge: Optional[LocalLLMJudge] = None,
    bertscore_model_path: str = "",
) -> Optional[Dict[str, Any]]:
    """Evaluate one movie / dataset output file."""
    with open(ad_file, encoding="utf-8") as f:
        data = json.load(f)

    entries = data.get("ad_entries", [])
    if not entries:
        print(f"  [SKIP] {ad_file.name}: no entries")
        return None

    movie = ad_file.stem.replace("_ad_output", "").replace("_output", "")
    print(f"\n{'='*60}")
    print(f"  [{movie}] ({len(entries)} entries)")

    result: Dict[str, Any] = {"movie": movie}

    # ── Extract GT references ───────────────────────────────
    candidates: Dict[int, str] = {}
    references: Dict[int, List[str]] = {}
    cand_list: List[str] = []
    ref_list: List[str] = []

    for idx, e in enumerate(entries):
        ad_text = str(e.get("ad_text", "")).strip()
        gt_text = str(e.get("gt_text", e.get("gt_caption", ""))).strip()
        # Use gap_id, clip_idx, or fallback to index as unique key
        key = e.get("gap_id") or e.get("clip_idx") or idx

        # Only include entries that have both candidate and reference
        if ad_text and gt_text:
            candidates[key] = ad_text
            cand_list.append(ad_text)
            references[key] = [gt_text]
            ref_list.append(gt_text)

    has_gt = len(references) > 0
    print(f"  GT refs: {len(references)}, candidates: {len(candidates)}")

    # ── GT-based metrics ────────────────────────────────────
    if has_gt:
        print("  [GT metrics] Computing CIDEr/BLEU/ROUGE/METEOR/SPICE...")
        coco_metrics = compute_pycoco_metrics(candidates, references)
        for name, val in coco_metrics.items():
            result[name] = round(val, 4)
            print(f"    {name}: {val:.4f}")

        print("  [GT metrics] Computing BERTScore vs GT...")
        bs = compute_bertscore_vs_gt(
            cand_list,
            ref_list,
            model_path=bertscore_model_path,
        )
        result["BERTScore"] = round(bs, 4)
        print(f"    BERTScore: {bs:.4f}")

        print("  [GT metrics] Computing R@1/5 retrieval...")
        ret_metrics = compute_retrieval_metrics(ref_list, cand_list)
        result["R@1"] = ret_metrics["R@1"]
        result["R@5"] = ret_metrics["R@5"]
        print(f"    R@1: {ret_metrics['R@1']:.2f}, R@5: {ret_metrics['R@5']:.2f}")
    else:
        print("  [GT metrics] No GT references found, skipping GT-based metrics")
        for name, _, _, _ in GT_METRICS:
            result[name] = None

    # ── CRITIC ──────────────────────────────────────────────
    print("  [CRITIC] Computing...")
    try:
        critic_score = compute_critic(entries)
        result["CRITIC"] = round(critic_score, 4) if critic_score is not None else None
        print(f"    CRITIC: {critic_score if critic_score is not None else 'N/A (no valid character annotations)'}")
    except Exception as e:
        print(f"    [WARN] CRITIC failed: {e}")
        result["CRITIC"] = None

    # ── LLM metrics ────────────────────────────────────────
    if llm_judge is not None:
        print("  [LLM Judge] Computing ISR + Decoupled...")
        default_instr = (
            "Describe what is happening in this clip concisely. "
            "Focus on visible actions, movements, and expressions."
        )
        isr_scores: List[float] = []
        style_scores: List[float] = []
        content_scores: List[float] = []
        overall_scores: List[float] = []

        for e in tqdm(entries, desc="  LLM Judge", leave=False):
            ad_text = str(e.get("ad_text", "")).strip()
            if not ad_text:
                continue

            isr = eval_isr(llm_judge, ad_text, default_instr)
            isr_scores.append(isr)

            dec = eval_decoupled(llm_judge, ad_text, default_instr)
            style_scores.append(dec["Style"])
            content_scores.append(dec["Content"])
            overall_scores.append(dec["Overall"])

        result["ISR"] = round(float(np.mean(isr_scores)), 4) if isr_scores else None
        result["Decoupled-Style"] = round(float(np.mean(style_scores)), 4) if style_scores else None
        result["Decoupled-Content"] = round(float(np.mean(content_scores)), 4) if content_scores else None
        result["Decoupled-Overall"] = round(float(np.mean(overall_scores)), 4) if overall_scores else None
        print(f"    ISR: {result['ISR']}, Style: {result['Decoupled-Style']}, "
              f"Content: {result['Decoupled-Content']}, Overall: {result['Decoupled-Overall']}")
    else:
        for name, _, _, _ in LLM_METRICS:
            result[name] = None

    # ── Timing ──────────────────────────────────────────────
    for col in TIMING_COLS:
        val = data.get(col, None)
        # Handle field name variants: time_per_clip_sec -> time_per_video_sec
        if val is None and col == "time_per_video_sec":
            val = data.get("time_per_clip_sec", None)
        result[col] = val

    return result


# ═══════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════

def compute_summary(all_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Aggregate per-movie results into a summary CSV."""
    summary = []
    all_defs = GT_METRICS + LLM_METRICS + OTHER_METRICS

    for name, lo, hi, direction in all_defs:
        vals = [r[name] for r in all_rows if r.get(name) is not None]
        if vals:
            summary.append({
                "metric": name,
                "range": f"[{lo}, {hi}]",
                "direction": direction,
                "mean_score": round(float(np.mean(vals)), 4),
                "num_movies": len(vals),
                "std": round(float(np.std(vals)), 4),
            })

    # Timing
    for col in TIMING_COLS:
        vals = [r[col] for r in all_rows if r.get(col) is not None and r[col] != 0]
        if vals:
            summary.append({
                "metric": col,
                "range": "",
                "direction": "",
                "mean_score": round(float(np.mean(vals)), 2),
                "num_movies": len(vals),
                "std": round(float(np.std(vals)), 2),
            })

    return summary


def write_csv(rows: List[Dict[str, Any]], path: Path):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"  CSV: {path}")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="GT-based + LLM evaluation for AD generation")
    parser.add_argument("--run-dir", required=True,
                        help="Directory containing *_ad_output.json or *_output.json")
    parser.add_argument("--output", default="",
                        help="Output directory (default: eval_result_gt)")
    parser.add_argument("--model", default="/mnt/disk5new/gcc/models/Qwen2.5-7B-Instruct",
                        help="LLM model path for LLM judge")
    parser.add_argument("--bertscore-model", default="roberta-large",
                        help="Model for BERTScore (default: roberta-large)")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--skip-llm", action="store_true",
                        help="Skip LLM-based metrics (ISR, Decoupled)")
    parser.add_argument("--movie", type=str, default="",
                        help="Only evaluate this movie (partial name match)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from previous eval (skip already-evaluated movies)")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        print(f"Error: {run_dir} is not a directory")
        return

    out_dir = Path(args.output) if args.output else PROJECT_ROOT / "eval_result_gt"
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

    # ── Find output files ───────────────────────────────────
    ad_files = sorted(run_dir.glob("*_ad_output.json")) + sorted(run_dir.glob("*_output.json"))
    movie_filter = args.movie.lower() if args.movie else ""
    print(f"\nFound {len(ad_files)} output files in {run_dir}")

    # ── Load LLM judge ──────────────────────────────────────
    llm_judge = None
    if not args.skip_llm:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        llm_judge = LocalLLMJudge(args.model, gpu=0)

    # ── Evaluate each movie ─────────────────────────────────
    all_rows: List[Dict[str, Any]] = []

    for idx, ad_file in enumerate(ad_files):
        movie = ad_file.stem.replace("_ad_output", "").replace("_output", "")

        if movie_filter and movie_filter not in movie.lower():
            continue

        if movie in skip_movies:
            continue

        row = evaluate_movie(
            ad_file,
            llm_judge=llm_judge,
            bertscore_model_path=args.bertscore_model,
        )
        if row:
            all_rows.append(row)

    # ── Merge resumed + new rows ────────────────────────────
    all_rows = resumed_rows + all_rows

    if not all_rows:
        print("\nNo movies evaluated.")
        return

    # ── Save per-movie results ──────────────────────────────
    detail_path = out_dir / f"{run_name}_eval_{ts}.json"
    with open(detail_path, "w", encoding="utf-8") as f:
        json.dump(all_rows, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {detail_path}")

    # ── Save per-movie CSV ──────────────────────────────────
    detail_csv_path = out_dir / f"{run_name}_eval_{ts}.csv"
    write_csv(all_rows, detail_csv_path)

    # ── Summary ─────────────────────────────────────────────
    summary = compute_summary(all_rows)
    summary_csv_path = out_dir / f"{run_name}_eval_{ts}.summary.csv"
    write_csv(summary, summary_csv_path)

    print(f"\n{'='*60}")
    print("  Evaluation Summary")
    print(f"{'='*60}")
    for row in summary:
        print(f"  {row['metric']:25s} = {row['mean_score']:.4f}  (n={row['num_movies']})")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
