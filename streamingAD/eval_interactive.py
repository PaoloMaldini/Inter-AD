#!/usr/bin/env python3
"""
eval_interactive.py — Evaluation metrics for interactive AD instruction experiments.

Computes 16+ metrics on experiment insertion records and outputs a CSV.

Metrics:
  A. Computational (embedding-based):
    1. ICA   — Instruction-Content Alignment (similarity: instruction ↔ after_text)
    2. SSS   — Semantic Shift Score (1 - sim: before_text ↔ after_text)
    3. IDD   — Instruction-Driven Divergence (α·ICA + β·SSS)
    4. SEC   — Side-Effect Control (coherence: before ↔ after, penalizing large shifts)
    5. SE    — Semantic Exploration (embedding distance: after ↔ instruction)
    6. Transition Smoothness (embedding sim: consecutive after_texts)

  B. Timing (from experiment records):
    7. Latency Before (sec)
    8. Latency After (sec)
    9. TTFF Before (sec)
   10. TTFF After (sec)

  C. LLM-Judge (Qwen-based):
   11. ISR    — Instruction Success Rate (binary pass/fail)
   12. IFR    — Instruction Following Ratio (decomposed requirements %)
   13. User-Alignment Score (1-5)
   14. Preference Accuracy (before vs after comparison)
   15. MSE    — LLM score vs. embedding alignment divergence
   16. nDCG   — Ranking quality of segments by relevance

Usage:
    python eval_interactive.py \
        --experiment-json experiment_results/movie_experiment.json \
        --output-csv eval_results/interactive_metrics.csv \
        --llm-model /mnt/disk5new/gcc/models/Qwen2.5-7B-Instruct \
        --gpu 1
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

# ── Project paths & env vars (cache locally, not on system disk) ─────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".hf_cache"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(PROJECT_ROOT / ".hf_cache" / "hub"))
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(PROJECT_ROOT / ".hf_cache" / "sentence_transformers"))
os.environ.setdefault("PIP_CACHE_DIR", str(PROJECT_ROOT / ".pip_cache"))
os.environ.setdefault("TORCH_HOME", str(PROJECT_ROOT / ".torch_cache"))

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


# ═══════════════════════════════════════════════════════════════════════════════
# Embedding Model
# ═══════════════════════════════════════════════════════════════════════════════

class EmbeddingModel:
    """Sentence embedding model for computing text similarities."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = "cuda"):
        from sentence_transformers import SentenceTransformer
        import torch
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        print(f"[embedding] Loading {model_name} on {device} ...")
        self.model = SentenceTransformer(model_name, device=device)
        print("[embedding] Model loaded.")

    def encode(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)
        return self.model.encode(texts, batch_size=batch_size, show_progress_bar=False,
                                 normalize_embeddings=True).astype(np.float32)

    def cosine_sim(self, text_a: str, text_b: str) -> float:
        emb = self.encode([text_a, text_b])
        return float(np.dot(emb[0], emb[1]))

    def cosine_sim_batch(self, texts_a: List[str], texts_b: List[str]) -> np.ndarray:
        emb_a = self.encode(texts_a)
        emb_b = self.encode(texts_b)
        return np.array([float(np.dot(a, b)) for a, b in zip(emb_a, emb_b)])


# ═══════════════════════════════════════════════════════════════════════════════
# LLM Judge
# ═══════════════════════════════════════════════════════════════════════════════

class LLMJudge:
    """Local LLM (Qwen) for judge-based metrics."""

    def __init__(self, model_path: str, gpu: int = 0):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.device = f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"
        print(f"[llm_judge] Loading {model_path} on {self.device} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
        ).to(self.device).eval()
        print("[llm_judge] Model loaded.")

    def generate(self, prompt: str, max_new_tokens: int = 256, temperature: float = 0.0) -> str:
        import torch
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                temperature=temperature if temperature > 0 else 1.0,
                do_sample=(temperature > 0),
            )
        return self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

    def score_1_to_5(self, prompt: str) -> Tuple[int, str]:
        raw = self.generate(prompt, max_new_tokens=128)
        score = 3
        for pattern in [r'\bscore\s*[:=]?\s*(\d)', r'\brating\s*[:=]?\s*(\d)', r'\b(\d)\s*/\s*5',
                        r'\b(\d)\s*out\s*of\s*5', r'[\[:]\s*(\d)\s*[\]]']:
            m = re.search(pattern, raw, re.IGNORECASE)
            if m:
                score = int(m.group(1))
                break
        return min(max(score, 1), 5), raw

    def score_binary(self, prompt: str) -> Tuple[int, str]:
        raw = self.generate(prompt, max_new_tokens=128)
        # Look for 1 or 2
        for pattern in [r'\bscore\s*[:=]?\s*(\d)', r'\b(\d)\b']:
            m = re.search(pattern, raw, re.IGNORECASE)
            if m:
                val = int(m.group(1))
                if val in (1, 2):
                    return val, raw
        # Default: look for keywords
        if any(w in raw.lower() for w in ["pass", "follows", "correct", "yes"]):
            return 2, raw
        return 1, raw


# ═══════════════════════════════════════════════════════════════════════════════
# Metric Functions
# ═══════════════════════════════════════════════════════════════════════════════

def compute_ica(embedder: EmbeddingModel, instruction: str, after_text: str) -> float:
    """Metric 1: Instruction-Content Alignment. Higher = better."""
    if not instruction.strip() or not after_text.strip():
        return 0.0
    return embedder.cosine_sim(instruction, after_text)


def compute_sss(embedder: EmbeddingModel, before_text: str, after_text: str) -> float:
    """Metric 2: Semantic Shift Score. Higher = larger change."""
    if not before_text.strip() or not after_text.strip():
        return 0.0
    return 1.0 - embedder.cosine_sim(before_text, after_text)


def compute_idd(ica: float, sss: float, alpha: float = 0.5, beta: float = 0.5) -> float:
    """Metric 3: Instruction-Driven Divergence. Higher = better."""
    return alpha * ica + beta * sss


def compute_sec(embedder: EmbeddingModel, before_text: str, after_text: str) -> float:
    """Metric 4: Side-Effect Control. Higher = fewer unintended changes."""
    if not before_text.strip() or not after_text.strip():
        return 1.0
    sim = embedder.cosine_sim(before_text, after_text)
    return max(0.0, sim)  # already 0~1


def compute_se(embedder: EmbeddingModel, instruction: str, after_text: str) -> float:
    """Metric 5: Semantic Exploration. Cosine distance from instruction to output."""
    if not instruction.strip() or not after_text.strip():
        return 0.0
    return 1.0 - embedder.cosine_sim(instruction, after_text)


def compute_transition_smoothness(embedder: EmbeddingModel, after_texts: List[str]) -> float:
    """Metric 6: Average cosine similarity between consecutive after_texts."""
    if len(after_texts) < 2:
        return 1.0
    sims = []
    for i in range(len(after_texts) - 1):
        if after_texts[i].strip() and after_texts[i + 1].strip():
            sims.append(embedder.cosine_sim(after_texts[i], after_texts[i + 1]))
    return float(np.mean(sims)) if sims else 1.0


def compute_isr_judge(judge: LLMJudge, instruction: str, after_text: str) -> Tuple[int, str]:
    """Metric 11: Instruction Success Rate. Binary 1/2."""
    prompt = f"""You are evaluating whether a generated Audio Description (AD) follows a specific instruction.

INSTRUCTION: {instruction}

GENERATED AD: {after_text}

Score:
  2 = The AD fully follows the instruction
  1 = The AD does NOT follow the instruction

Output format: Score: X  (one line, then brief reason)
"""
    return judge.score_binary(prompt)


def compute_ifr_judge(judge: LLMJudge, instruction: str, after_text: str) -> Tuple[float, str]:
    """Metric 12: Instruction Following Ratio. Decompose into sub-requirements."""
    prompt = f"""You are evaluating how well a generated Audio Description (AD) follows an instruction.
Break the instruction into individual requirements and check each one.

INSTRUCTION: {instruction}

GENERATED AD: {after_text}

List each requirement as [MET] or [NOT MET], then give a ratio as Score: X/Y (e.g., Score: 2/3).

Output format:
Requirement 1: [MET/NOT MET] (brief description)
Requirement 2: [MET/NOT MET] (brief description)
Score: X/Y
"""
    raw = judge.generate(prompt, max_new_tokens=256)
    # Parse X/Y
    ratio = 0.0
    m = re.search(r'score\s*[:=]?\s*(\d+)\s*/\s*(\d+)', raw, re.IGNORECASE)
    if m:
        num, den = int(m.group(1)), int(m.group(2))
        ratio = num / max(den, 1)
    return round(ratio, 4), raw


def compute_user_alignment(judge: LLMJudge, instruction: str, after_text: str) -> Tuple[int, str]:
    """Metric 13: User-Alignment Score (1-5)."""
    prompt = f"""You are evaluating how well a generated Audio Description (AD) aligns with the user's intent.

USER INSTRUCTION: {instruction}

GENERATED AD: {after_text}

Rate alignment 1-5:
  5 = Perfectly captures user's intent and instruction
  4 = Mostly aligned, minor gaps
  3 = Partially aligned
  2 = Poorly aligned
  1 = Completely ignores user intent

Output format: Score: X  (one line, then brief reason)
"""
    return judge.score_1_to_5(prompt)


def compute_preference_accuracy(judge: LLMJudge, instruction: str, before_text: str, after_text: str) -> Tuple[str, str]:
    """Metric 14: Preference Accuracy. Does after > before for the given instruction?"""
    prompt = f"""You are comparing two Audio Description (AD) versions for the same video segment.
The user provided an instruction. Decide which version better follows the instruction.

USER INSTRUCTION: {instruction}

AD VERSION A (before instruction): {before_text}
AD VERSION B (after instruction): {after_text}

Which version better follows the instruction?
  1 = Version A is better
  2 = Version B is better (expected if instruction was applied)
  3 = About the same

Output format: Choice: X  (one line, then brief reason)
"""
    raw = judge.generate(prompt, max_new_tokens=128)
    choice = "tie"
    m = re.search(r'choice\s*[:=]?\s*(\d)', raw, re.IGNORECASE)
    if m:
        val = int(m.group(1))
        if val == 2:
            choice = "after"
        elif val == 1:
            choice = "before"
        else:
            choice = "tie"
    return choice, raw


def compute_mse(llm_score: float, embedding_score: float) -> float:
    """Metric 15: MSE between LLM-judged alignment and embedding-based alignment."""
    # Normalize LLM score from 1-5 to 0-1
    llm_norm = (llm_score - 1) / 4.0
    return (llm_norm - embedding_score) ** 2


def compute_ndcg(scores: List[float], k: Optional[int] = None) -> float:
    """Metric 16: nDCG — normalized Discounted Cumulative Gain.

    Uses the scores as relevance grades. Measures how well the natural order
    (by insertion time) ranks the segments by their instruction-alignment scores.
    """
    if not scores:
        return 0.0
    if k is None:
        k = len(scores)

    # DCG of the actual order
    dcg = sum(score / math.log2(i + 2) for i, score in enumerate(scores[:k]))

    # Ideal DCG (sorted by score descending)
    ideal = sorted(scores, reverse=True)[:k]
    idcg = sum(score / math.log2(i + 2) for i, score in enumerate(ideal))

    return dcg / idcg if idcg > 0 else 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# Full Evaluation Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def run_full_evaluation(
    experiment_json: Path,
    output_csv: Path,
    llm_model_path: Optional[str] = None,
    gpu: int = 0,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    skip_llm: bool = False,
) -> Dict[str, Any]:
    """
    Run all 16 metrics on an experiment JSON and output CSV.

    Args:
        experiment_json: Path to experiment result JSON.
        output_csv: Path to output CSV file.
        llm_model_path: Path to Qwen model (None = skip LLM metrics).
        gpu: GPU id.
        embedding_model: Sentence-transformers model name.
        skip_llm: If True, skip all LLM-judge metrics.

    Returns:
        Dict of aggregate metric scores.
    """
    from interactive_experiment import load_experiment_result

    # ── Load data ─────────────────────────────────────────────────────────
    result = load_experiment_result(experiment_json)
    records = result.insertion_records
    print(f"[eval] Loaded {len(records)} insertion records from {experiment_json}")

    if not records:
        print("[eval] No records to evaluate.")
        return {}

    # ── Load models ───────────────────────────────────────────────────────
    embedder = EmbeddingModel(model_name=embedding_model, device="cuda" if not skip_llm else "cpu")

    judge: Optional[LLMJudge] = None
    if not skip_llm and llm_model_path:
        judge = LLMJudge(model_path=llm_model_path, gpu=gpu)

    # ── Compute metrics per record ────────────────────────────────────────
    rows: List[Dict[str, Any]] = []
    after_texts = [r.text_after for r in records]

    for i, rec in enumerate(tqdm(records, desc="Evaluating")):
        row: Dict[str, Any] = {
            "insertion_id": rec.insertion_id,
            "movie_title": rec.movie_title,
            "segment_idx": rec.segment_idx,
            "timestamp_sec": rec.insert_timestamp_sec,
            "category_id": rec.category_id,
            "category_name": rec.category_name,
            "instruction": rec.instruction_text,
            "instruction_lang": rec.instruction_language,
            "active_instr_count": rec.active_instructions_count,
            "text_before": rec.text_before,
            "text_after": rec.text_after,
            "ref_ad": rec.ref_ad,
        }

        # ── A. Embedding metrics ──────────────────────────────────────────
        ica = compute_ica(embedder, rec.instruction_text, rec.text_after)
        sss = compute_sss(embedder, rec.text_before, rec.text_after)
        idd = compute_idd(ica, sss)
        sec = compute_sec(embedder, rec.text_before, rec.text_after)
        se = compute_se(embedder, rec.instruction_text, rec.text_after)

        row["ICA"] = round(ica, 4)
        row["SSS"] = round(sss, 4)
        row["IDD"] = round(idd, 4)
        row["SEC"] = round(sec, 4)
        row["SE"] = round(se, 4)

        # ── B. Timing metrics ─────────────────────────────────────────────
        row["Latency_Before_sec"] = rec.latency_before_sec
        row["Latency_After_sec"] = rec.latency_after_sec
        row["TTFF_Before_sec"] = rec.ttff_before_sec
        row["TTFF_After_sec"] = rec.ttff_after_sec

        # ── C. LLM-judge metrics ─────────────────────────────────────────
        if judge is not None:
            # ISR (binary)
            isr_score, isr_rationale = compute_isr_judge(judge, rec.instruction_text, rec.text_after)
            row["ISR"] = isr_score  # 1=fail, 2=pass
            row["ISR_rationale"] = isr_rationale[:200]

            # IFR (decomposed ratio)
            ifr_score, ifr_rationale = compute_ifr_judge(judge, rec.instruction_text, rec.text_after)
            row["IFR"] = ifr_score
            row["IFR_rationale"] = ifr_rationale[:200]

            # User-Alignment (1-5)
            ua_score, ua_rationale = compute_user_alignment(judge, rec.instruction_text, rec.text_after)
            row["User_Alignment"] = ua_score
            row["User_Alignment_rationale"] = ua_rationale[:200]

            # Preference Accuracy
            pref_choice, pref_rationale = compute_preference_accuracy(
                judge, rec.instruction_text, rec.text_before, rec.text_after
            )
            row["Preference_Accuracy"] = pref_choice  # "after", "before", "tie"
            row["Preference_Accuracy_rationale"] = pref_rationale[:200]

            # MSE (LLM score vs embedding alignment)
            row["MSE"] = round(compute_mse(float(ua_score), ica), 4)
        else:
            for col in ["ISR", "ISR_rationale", "IFR", "IFR_rationale",
                        "User_Alignment", "User_Alignment_rationale",
                        "Preference_Accuracy", "Preference_Accuracy_rationale", "MSE"]:
                row[col] = ""

        rows.append(row)

    # ── Transition Smoothness (computed over full sequence) ───────────────
    ts_score = compute_transition_smoothness(embedder, after_texts)
    for row in rows:
        row["Transition_Smoothness"] = round(ts_score, 4)

    # ── nDCG (computed over full sequence using IDD as relevance) ─────────
    idd_scores = [r["IDD"] for r in rows]
    ndcg_score = compute_ndcg(idd_scores)
    for row in rows:
        row["nDCG"] = round(ndcg_score, 4)

    # ── Write CSV ─────────────────────────────────────────────────────────
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    # Define column order
    columns = [
        "insertion_id", "movie_title", "segment_idx", "timestamp_sec",
        "category_id", "category_name", "instruction", "instruction_lang",
        "active_instr_count",
        # Embedding metrics
        "ICA", "SSS", "IDD", "SEC", "SE",
        # Timing
        "Latency_Before_sec", "Latency_After_sec",
        "TTFF_Before_sec", "TTFF_After_sec",
        # Transition
        "Transition_Smoothness",
        # LLM metrics
        "ISR", "IFR", "User_Alignment", "Preference_Accuracy",
        "MSE", "nDCG",
        # Rationales (optional)
        "ISR_rationale", "IFR_rationale",
        "User_Alignment_rationale", "Preference_Accuracy_rationale",
        # Texts
        "text_before", "text_after", "ref_ad",
    ]

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[eval] CSV saved to {output_csv}")

    # ── Aggregate summary ─────────────────────────────────────────────────
    summary: Dict[str, Any] = {
        "num_records": len(rows),
        "movie": result.movie_title,
    }
    for col in ["ICA", "SSS", "IDD", "SEC", "SE",
                "Latency_Before_sec", "Latency_After_sec",
                "TTFF_Before_sec", "TTFF_After_sec",
                "Transition_Smoothness", "nDCG"]:
        vals = [r[col] for r in rows if isinstance(r.get(col), (int, float))]
        if vals:
            summary[f"{col}_mean"] = round(float(np.mean(vals)), 4)
            summary[f"{col}_std"] = round(float(np.std(vals)), 4)

    # LLM metrics summary
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

    print("\n" + "=" * 60)
    print("AGGREGATE METRICS SUMMARY")
    print("=" * 60)
    for k, v in summary.items():
        print(f"  {k}: {v}")

    return summary


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Evaluate interactive AD experiment results")
    parser.add_argument("--experiment-json", required=True, help="Path to experiment result JSON")
    parser.add_argument("--output-csv", required=True, help="Path to output CSV")
    parser.add_argument("--llm-model", default=None, help="Path to Qwen LLM model (None = skip LLM metrics)")
    parser.add_argument("--gpu", type=int, default=0, help="GPU id")
    parser.add_argument("--embedding-model", default="/mnt/temp_disk/ghl/models/paraphrase-multilingual-MiniLM-L12-v2",
                        help="Sentence-transformers model name")
    parser.add_argument("--skip-llm", action="store_true", help="Skip all LLM-judge metrics")
    args = parser.parse_args()

    run_full_evaluation(
        experiment_json=Path(args.experiment_json),
        output_csv=Path(args.output_csv),
        llm_model_path=args.llm_model,
        gpu=args.gpu,
        embedding_model=args.embedding_model,
        skip_llm=args.skip_llm,
    )


if __name__ == "__main__":
    main()
