#!/usr/bin/env python3
"""
eval_local_llm.py — LLM-as-Judge metrics using locally deployed open-source models.

Metrics:
  1. LLM-AD-eval    — Core action/object consistency vs. GT (textual entailment)
  2. ISR             — Instruction Success Rate (format/compliance check)
  3. Decoupled-Eval  — Style tone + factual content scored independently

Usage:
    conda activate focusedad
    python streamingAD/eval_local_llm.py \
        --run-dir batch_ad_output/focusedad_run_20260531_021518 \
        --model Qwen/Qwen2.5-14B-Instruct \
        --gpu 1

Or auto-detect latest run dir:
    python streamingAD/eval_local_llm.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

OUTPUT_BASE = PROJECT_ROOT / "batch_ad_output"


def _find_latest_run_dir() -> Path | None:
    base = OUTPUT_BASE
    run_dirs = sorted(
        [d for d in base.iterdir() if d.is_dir() and (d.name.startswith("run_") or d.name.startswith("focusedad_run_"))],
        reverse=True,
    )
    return run_dirs[0] if run_dirs else None


# ── LLM Wrapper (HuggingFace) ─────────────────────────────────

class LocalLLMJudge:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-14B-Instruct", gpu: int = 0):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"
        print(f"[LLM Judge] Loading {model_name} on {self.device} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(self.device)
        self.model.eval()
        self.model_name = model_name
        print(f"[LLM Judge] Model loaded.")

    def generate(self, prompt: str, max_new_tokens: int = 256, temperature: float = 0.0) -> str:
        import torch
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
        rationale = raw
        import re
        for pattern in [r'\bscore\s*[:=]?\s*(\d)', r'\brating\s*[:=]?\s*(\d)', r'\b(\d)\s*/\s*5']:
            m = re.search(pattern, raw, re.IGNORECASE)
            if m:
                score = int(m.group(1))
                break
        return min(max(score, 1), 5), rationale


# ── Metric 1: LLM-AD-eval ────────────────────────────────────

def eval_llm_ad(
    ad_entries: List[Dict[str, Any]],
    refs: Dict[str, List[str]],
    judge: LocalLLMJudge,
) -> Dict[str, Any]:
    """
    Evaluate core action/object consistency between generated AD and GT.
    Uses textual entailment: "does the generated text describe the same action/objects?"
    """
    PROMPT_TEMPLATE = """You are an expert evaluator for Audio Description (AD) quality.
Your task is to compare a GENERATED AD sentence with a GROUND TRUTH (GT) AD sentence
for the same video clip gap.

Ignore character names, actor names.  Only evaluate whether the core ACTIONS,
MOVEMENTS, and VISUAL OBJECTS described are consistent.

GENERATED AD: {generated}
GROUND TRUTH:  {gt}

Rate on a 1-5 scale:
  5 = Actions and objects fully match
  4 = Minor differences (e.g., missing one detail)
  3 = Partial match (half correct)
  2 = Mostly wrong
  1 = Completely different or contradictory

Output format: Score: X  (one line only, then brief reason)
"""
    scores = []
    per_sample = []

    ref_gap_ids = list(refs.keys())
    for entry in ad_entries:
        gap_id = str(entry.get("gap_id", ""))
        ad_text = entry.get("ad_text", "")
        if not ad_text or gap_id not in refs:
            continue

        gt_texts = refs[gap_id]
        gt = " | ".join(gt_texts)

        prompt = PROMPT_TEMPLATE.format(generated=ad_text, gt=gt)
        score, rationale = judge.score_1_to_5(prompt)
        scores.append(score)
        per_sample.append({
            "gap_id": gap_id,
            "ad_text": ad_text,
            "gt_text": gt,
            "score": score,
            "rationale": rationale,
        })

    avg = round(sum(scores) / len(scores), 3) if scores else 0
    return {
        "metric": "LLM-AD-eval",
        "score": avg,
        "num_samples": len(scores),
        "per_sample": per_sample,
    }


# ── Metric 2: ISR (Instruction Success Rate) ──────────────────

def eval_isr(
    ad_entries: List[Dict[str, Any]],
    judge: LocalLLMJudge,
    instruction: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Check if the generated AD follows a given hard instruction.
    Default instruction: "Describe visible actions concisely, no character names."
    """
    DEFAULT_INSTRUCTION = (
        "Describe what is happening in this clip concisely. "
        "Focus on visible actions, movements, and expressions. "
        "If character names are mentioned in the context, use them "
        "(e.g. 'Don Vito Corleone walks...' not 'A man walks...'). "
        "Do not quote dialogue."
    )

    instr = (instruction or DEFAULT_INSTRUCTION).strip()

    PROMPT_TEMPLATE = """You are evaluating whether a generated Audio Description (AD) 
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
    per_sample = []

    for entry in ad_entries:
        ad_text = entry.get("ad_text", "")
        if not ad_text:
            continue

        prompt = PROMPT_TEMPLATE.format(instruction=instr, ad_text=ad_text)
        score, rationale = judge.score_1_to_5(prompt)
        isr_score = 1.0 if score >= 2 else 0.0
        scores.append(isr_score)
        per_sample.append({
            "gap_id": entry.get("gap_id", ""),
            "ad_text": ad_text,
            "passed": isr_score > 0,
            "rationale": rationale,
        })

    success_rate = round(sum(scores) / len(scores), 4) if scores else 0
    return {
        "metric": "ISR",
        "score": success_rate,
        "num_samples": len(scores),
        "per_sample": per_sample,
    }


# ── Metric 3: Decoupled Evaluation ────────────────────────────

def eval_decoupled(
    ad_entries: List[Dict[str, Any]],
    judge: LocalLLMJudge,
) -> Dict[str, Any]:
    """
    Score style (tone, fluency) and factual content independently.
    """
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
    per_sample = []

    for entry in ad_entries:
        ad_text = entry.get("ad_text", "")
        if not ad_text:
            continue

        cb = entry.get("context_before", "")
        ca = entry.get("context_after", "")

        s_score, s_rationale = judge.score_1_to_5(
            STYLE_PROMPT.format(ad_text=ad_text)
        )
        c_score, c_rationale = judge.score_1_to_5(
            CONTENT_PROMPT.format(
                context_before=str(cb)[:200],
                ad_text=ad_text,
                context_after=str(ca)[:200],
            )
        )
        style_scores.append(s_score)
        content_scores.append(c_score)
        per_sample.append({
            "gap_id": entry.get("gap_id", ""),
            "ad_text": ad_text,
            "style_score": s_score,
            "style_rationale": s_rationale,
            "content_score": c_score,
            "content_rationale": c_rationale,
        })

    avg_style = round(sum(style_scores) / len(style_scores), 3) if style_scores else 0
    avg_content = round(sum(content_scores) / len(content_scores), 3) if content_scores else 0
    return {
        "metric": "Decoupled-Eval",
        "style_score": avg_style,
        "content_score": avg_content,
        "overall_score": round((avg_style + avg_content) / 2, 3),
        "num_samples": len(style_scores),
        "per_sample": per_sample,
    }


# ── Batch runner ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LLM-as-Judge: local model evaluation")
    parser.add_argument("--run-dir", type=str, default="",
                        help="Path to run directory (default: auto-detect latest)")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-14B-Instruct",
                        help="HuggingFace model name for judging")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID")
    parser.add_argument("--instruction", type=str, default="",
                        help="Custom instruction for ISR metric")
    parser.add_argument("--metrics", type=str, default="all",
                        help="Comma-separated: llm_ad,isr,decoupled,all")
    parser.add_argument("--limit", type=int, default=0,
                        help="Max samples per metric (0 = all)")
    args = parser.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir.strip() else _find_latest_run_dir()
    if not run_dir or not run_dir.exists():
        print(f"Run dir not found: {run_dir}")
        sys.exit(1)

    metric_list = ["llm_ad", "isr", "decoupled"] if args.metrics == "all" else args.metrics.split(",")

    print(f"Run dir: {run_dir}")
    print(f"Model:   {args.model}")
    print(f"Metrics: {metric_list}\n")

    judge = LocalLLMJudge(model_name=args.model, gpu=args.gpu)

    ad_files = sorted(run_dir.glob("*_ad_output.json"))
    if not ad_files:
        print("No _ad_output.json files found.")
        sys.exit(1)
    print(f"Found {len(ad_files)} movie(s)\n")

    all_results: Dict[str, List[Dict[str, Any]]] = {m: [] for m in metric_list}

    for ad_file in ad_files:
        movie = ad_file.stem.replace("_ad_output", "")
        ref_file = run_dir / f"{movie}_ref.json"

        with open(ad_file) as f:
            data = json.load(f)
        entries = data.get("ad_entries", [])
        if args.limit > 0:
            entries = entries[:args.limit]

        refs: Dict[str, List[str]] = {}
        if ref_file.exists():
            with open(ref_file) as f:
                ref_data = json.load(f)
            refs = {str(k): [str(v)] if isinstance(v, str) else [str(x) for x in v]
                    for k, v in ref_data.items()}

        print(f"[{movie}] {len(entries)} entries")

        if "llm_ad" in metric_list:
            r = eval_llm_ad(entries, refs, judge)
            all_results["llm_ad"].append({**r, "movie": movie})
            print(f"  LLM-AD-eval: {r['score']} ({r['num_samples']} samples)")

        if "isr" in metric_list:
            r = eval_isr(entries, judge, instruction=args.instruction or None)
            all_results["isr"].append({**r, "movie": movie})
            print(f"  ISR: {r['score']:.2%} ({r['num_samples']} samples)")

        if "decoupled" in metric_list:
            r = eval_decoupled(entries, judge)
            all_results["decoupled"].append({**r, "movie": movie})
            print(f"  Decoupled: style={r['style_score']}, content={r['content_score']}")

    # ── Aggregate & save ──
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    for metric_name, movie_results in all_results.items():
        if not movie_results:
            continue

        json_path = run_dir / f"eval_local_{metric_name}_{ts}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(movie_results, f, ensure_ascii=False, indent=2)

        csv_path = run_dir / f"eval_local_{metric_name}_{ts}.csv"
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            import csv
            writer = csv.writer(f)
            if metric_name == "llm_ad":
                writer.writerow(["movie", "llm_ad_eval_score", "num_samples"])
                for mr in movie_results:
                    writer.writerow([mr["movie"], mr["score"], mr["num_samples"]])
            elif metric_name == "isr":
                writer.writerow(["movie", "isr_score", "num_samples"])
                for mr in movie_results:
                    writer.writerow([mr["movie"], mr["score"], mr["num_samples"]])
            elif metric_name == "decoupled":
                writer.writerow(["movie", "style_score", "content_score", "overall_score", "num_samples"])
                for mr in movie_results:
                    writer.writerow([mr["movie"], mr["style_score"], mr["content_score"], mr["overall_score"], mr["num_samples"]])

        print(f"\nSaved: {json_path}")
        print(f"Saved: {csv_path}")

    print(f"\nDone. Output: {run_dir}")


if __name__ == "__main__":
    main()
