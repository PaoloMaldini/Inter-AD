#!/usr/bin/env python3
"""
run_eval.py — Evaluate interactive AD experiment results.

Computes 16 metrics on experiment insertion records and outputs a CSV.

Metrics:
  A. Computational (embedding-based):
    1. ICA   — Instruction-Content Alignment
    2. SSS   — Semantic Shift Score
    3. IDD   — Instruction-Driven Divergence
    4. SEC   — Side-Effect Control
    5. SE    — Semantic Exploration
    6. Transition Smoothness

  B. Timing (from experiment records):
    7. Latency Before/After (sec)
    8. TTFF Before/After (sec)

  C. LLM-Judge (Qwen-based):
    9. ISR    — Instruction Success Rate
   10. IFR    — Instruction Following Ratio
   11. User-Alignment Score
   12. Preference Accuracy
   13. MSE    — LLM vs embedding divergence
   14. nDCG   — Ranking quality

Usage:
    # Evaluate single experiment:
    conda activate videollava
    python streamingAD/run_eval.py \
        --experiment-json experiment_results/Shawshank_experiment.json \
        --output-csv eval_results/shawshank_metrics.csv \
        --llm-model /mnt/disk5new/gcc/models/Qwen2.5-7B-Instruct \
        --gpu 1

    # Evaluate multiple experiments:
    python streamingAD/run_eval.py \
        --experiment-json experiment_results/*.json \
        --output-dir eval_results/ \
        --llm-model /mnt/disk5new/gcc/models/Qwen2.5-7B-Instruct \
        --gpu 1

    # Skip LLM metrics (faster, only embedding metrics):
    python streamingAD/run_eval.py \
        --experiment-json experiment_results/Shawshank_experiment.json \
        --output-csv eval_results/shawshank_metrics.csv \
        --skip-llm
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import re
import sys
import time
from datetime import datetime
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

from interactive_experiment import load_experiment_result, InsertionRecord, MovieExperimentResult


# ═══════════════════════════════════════════════════════════════════════════════
# Embedding Model
# ═══════════════════════════════════════════════════════════════════════════════

class EmbeddingModel:
    """Sentence embedding model for computing text similarities.

    Falls back to a local TF-IDF embedding if `sentence-transformers` is
    unavailable or incompatible with the current `huggingface_hub` version.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = "cuda"):
        self._backend = "tfidf"
        self._tfidf = None
        self._vocab: Optional[Any] = None
        try:
            import torch
            from sentence_transformers import SentenceTransformer
            if device.startswith("cuda") and not torch.cuda.is_available():
                device = "cpu"
            print(f"[embedding] Loading {model_name} on {device} ...")
            self.model = SentenceTransformer(model_name, device=device)
            self._backend = "st"
            print("[embedding] Model loaded.")
        except Exception as e:
            print(f"[embedding] SentenceTransformer unavailable ({type(e).__name__}: {e}).")
            print("[embedding] Falling back to local TF-IDF embedding.")
            self._init_tfidf()

    def _init_tfidf(self) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        self._tfidf = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            sublinear_tf=True,
            dtype=np.float32,
        )

    def _ensure_tfidf_fit(self, texts: List[str]) -> None:
        if self._vocab is not None:
            return
        if not texts:
            return
        try:
            self._tfidf.fit(texts)
            self._vocab = self._tfidf.vocabulary_
        except ValueError:
            # Fallback to character n-grams for very short / stopword-only texts
            from sklearn.feature_extraction.text import TfidfVectorizer
            self._tfidf = TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                sublinear_tf=True,
                dtype=np.float32,
            )
            self._tfidf.fit(texts)
            self._vocab = self._tfidf.vocabulary_

    def encode(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        if self._backend == "st":
            return self.model.encode(texts, batch_size=batch_size, show_progress_bar=False,
                                     normalize_embeddings=True).astype(np.float32)
        self._ensure_tfidf_fit(texts)
        vecs = self._tfidf.transform(texts).toarray().astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        return vecs / norms

    def cosine_sim(self, text_a: str, text_b: str) -> float:
        emb = self.encode([text_a, text_b])
        return float(np.dot(emb[0], emb[1]))


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
        for pattern in [r'\bscore\s*[:=]?\s*(\d)', r'\b(\d)\b']:
            m = re.search(pattern, raw, re.IGNORECASE)
            if m:
                val = int(m.group(1))
                if val in (1, 2):
                    return val, raw
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
    return max(0.0, sim)


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
    """Metric 9: Instruction Success Rate. Binary 1/2."""
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
    """Metric 10: Instruction Following Ratio. Decompose into sub-requirements."""
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
    ratio = 0.0
    m = re.search(r'score\s*[:=]?\s*(\d+)\s*/\s*(\d+)', raw, re.IGNORECASE)
    if m:
        num, den = int(m.group(1)), int(m.group(2))
        ratio = num / max(den, 1)
    return round(ratio, 4), raw


def compute_user_alignment(judge: LLMJudge, instruction: str, after_text: str) -> Tuple[int, str]:
    """Metric 11: User-Alignment Score (1-5)."""
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
    """Metric 12: Preference Accuracy. Does after > before for the given instruction?"""
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
    """Metric 13: MSE between LLM-judged alignment and embedding-based alignment."""
    llm_norm = (llm_score - 1) / 4.0
    return (llm_norm - embedding_score) ** 2


def compute_ndcg(scores: List[float], k: Optional[int] = None) -> float:
    """Metric 14: nDCG — normalized Discounted Cumulative Gain."""
    if not scores:
        return 0.0
    if k is None:
        k = len(scores)
    dcg = sum(score / math.log2(i + 2) for i, score in enumerate(scores[:k]))
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
    Run all metrics on an experiment JSON and output CSV.

    Returns:
        Dict of aggregate metric scores.
    """
    # ── Load data ─────────────────────────────────────────────────────────
    result = load_experiment_result(experiment_json)
    records = result.insertion_records
    print(f"[eval] Loaded {len(records)} insertion records from {experiment_json}")

    if not records:
        print("[eval] No records to evaluate.")
        return {}

    # ── Load models ───────────────────────────────────────────────────────
    embedder = EmbeddingModel(model_name=embedding_model, device="cuda")

    judge: Optional[LLMJudge] = None
    if not skip_llm and llm_model_path:
        judge = LLMJudge(model_path=llm_model_path, gpu=gpu)

    # ── Compute metrics per record ────────────────────────────────────────
    rows: List[Dict[str, Any]] = []
    after_texts = [r.text_after for r in records]

    for i, rec in enumerate(tqdm(records, desc="Evaluating")):
        # Build structured active instructions string for CSV
        active_instr_str = " | ".join(
            f"[{a.get('category', '')}] {a.get('template', '')}"
            for a in getattr(rec, 'all_active_instructions', [])
        ) if hasattr(rec, 'all_active_instructions') and rec.all_active_instructions else ""

        row: Dict[str, Any] = {
            "insertion_id": rec.insertion_id,
            "movie_title": rec.movie_title,
            "segment_idx": rec.segment_idx,
            "timestamp_sec": rec.insert_timestamp_sec,
            "category_id": rec.category_id,
            "category_name": rec.category_name,
            "instruction_new": rec.instruction_text,                 # the NEW instruction
            "instruction_before": getattr(rec, 'instruction_before', ''),  # full instruction for text_before
            "instruction_after": getattr(rec, 'instruction_after', ''),    # full instruction for text_after
            "all_active_instructions": active_instr_str,             # structured list
            "instruction_lang": rec.instruction_language,
            "active_instr_count": rec.active_instructions_count,
            "text_before": rec.text_before,
            "text_after": rec.text_after,
            "ref_ad": rec.ref_ad,
        }

        # Use the NEW instruction for alignment metrics (measures response to the new change)
        # Use instruction_after for LLM judge (evaluates the full prompt context)
        new_instr = rec.instruction_text
        full_instr = getattr(rec, 'instruction_after', rec.instruction_text)

        # ── A. Embedding metrics ──────────────────────────────────────────
        ica = compute_ica(embedder, new_instr, rec.text_after)
        sss = compute_sss(embedder, rec.text_before, rec.text_after)
        idd = compute_idd(ica, sss)
        sec = compute_sec(embedder, rec.text_before, rec.text_after)
        se = compute_se(embedder, new_instr, rec.text_after)

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
        # Use the NEW instruction for judging whether the system responded to it
        if judge is not None:
            isr_score, isr_rationale = compute_isr_judge(judge, new_instr, rec.text_after)
            row["ISR"] = isr_score
            row["ISR_rationale"] = isr_rationale[:200]

            ifr_score, ifr_rationale = compute_ifr_judge(judge, new_instr, rec.text_after)
            row["IFR"] = ifr_score
            row["IFR_rationale"] = ifr_rationale[:200]

            ua_score, ua_rationale = compute_user_alignment(judge, new_instr, rec.text_after)
            row["User_Alignment"] = ua_score
            row["User_Alignment_rationale"] = ua_rationale[:200]

            pref_choice, pref_rationale = compute_preference_accuracy(
                judge, new_instr, rec.text_before, rec.text_after
            )
            row["Preference_Accuracy"] = pref_choice
            row["Preference_Accuracy_rationale"] = pref_rationale[:200]

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
        "ISR_rationale", "IFR_rationale",
        "User_Alignment_rationale", "Preference_Accuracy_rationale",
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
    parser = argparse.ArgumentParser(
        description="Evaluate interactive AD experiment results (16 metrics → CSV)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single experiment with LLM judge:
  python streamingAD/run_eval.py \\
      --experiment-json experiment_results/Shawshank_experiment.json \\
      --output-csv eval_results/shawshank_metrics.csv \\
      --llm-model /mnt/disk5new/gcc/models/Qwen2.5-7B-Instruct \\
      --gpu 1

  # Multiple experiments:
  python streamingAD/run_eval.py \\
      --experiment-json experiment_results/*.json \\
      --output-dir eval_results/ \\
      --llm-model /mnt/disk5new/gcc/models/Qwen2.5-7B-Instruct \\
      --gpu 1

  # Skip LLM metrics (faster):
  python streamingAD/run_eval.py \\
      --experiment-json experiment_results/Shawshank_experiment.json \\
      --output-csv eval_results/shawshank_metrics.csv \\
      --skip-llm
        """
    )

    parser.add_argument("--experiment-json", nargs="+", required=True,
                        help="Path(s) to experiment result JSON file(s)")
    parser.add_argument("--output-csv", type=str, default=None,
                        help="Output CSV path (for single experiment)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (for multiple experiments)")
    parser.add_argument("--llm-model", type=str, default=None,
                        help="Path to Qwen LLM model (None = skip LLM metrics)")
    parser.add_argument("--gpu", type=int, default=0, help="GPU id")
    parser.add_argument("--embedding-model",
                        default="/mnt/temp_disk/ghl/models/paraphrase-multilingual-MiniLM-L12-v2",
                        help="Sentence-transformers model name or local path")
    parser.add_argument("--skip-llm", action="store_true", help="Skip all LLM-judge metrics")

    args = parser.parse_args()

    # ── Validate args ─────────────────────────────────────────────────────
    if len(args.experiment_json) > 1 and not args.output_dir:
        parser.error("--output-dir required when evaluating multiple experiments")

    if len(args.experiment_json) == 1 and not args.output_csv and not args.output_dir:
        # Default: derive CSV name from JSON name
        json_path = Path(args.experiment_json[0])
        args.output_csv = str(json_path.parent / f"{json_path.stem}_metrics.csv")

    # ── Run evaluations ───────────────────────────────────────────────────
    all_summaries: List[Dict[str, Any]] = []

    for json_path in args.experiment_json:
        json_path = Path(json_path)
        if not json_path.exists():
            print(f"[warn] File not found: {json_path}, skipping")
            continue

        # Determine output CSV path
        if args.output_dir:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_csv = output_dir / f"{json_path.stem}_metrics.csv"
        else:
            output_csv = Path(args.output_csv)

        print(f"\n{'=' * 60}")
        print(f"Evaluating: {json_path.name}")
        print(f"Output CSV: {output_csv}")
        print(f"{'=' * 60}")

        summary = run_full_evaluation(
            experiment_json=json_path,
            output_csv=output_csv,
            llm_model_path=args.llm_model,
            gpu=args.gpu,
            embedding_model=args.embedding_model,
            skip_llm=args.skip_llm,
        )
        summary["experiment_json"] = str(json_path)
        summary["output_csv"] = str(output_csv)
        all_summaries.append(summary)

    # ── Print combined summary ────────────────────────────────────────────
    if len(all_summaries) > 1:
        print(f"\n{'#' * 60}")
        print("COMBINED SUMMARY")
        print(f"{'#' * 60}")
        for s in all_summaries:
            print(f"\n  Movie: {s.get('movie', 'N/A')}")
            print(f"  Records: {s.get('num_records', 0)}")
            for key in ["ICA_mean", "SSS_mean", "IDD_mean", "SEC_mean", "SE_mean",
                         "ISR_pass_rate", "IFR_mean", "User_Alignment_mean",
                         "Preference_After_pct", "MSE_mean", "nDCG_mean"]:
                if key in s:
                    print(f"  {key}: {s[key]}")

    print(f"\nDone! Evaluated {len(all_summaries)} experiment(s).")


if __name__ == "__main__":
    main()
