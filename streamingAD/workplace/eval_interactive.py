#!/usr/bin/env python3
"""
eval_interactive.py — Final aggregate evaluator for interactive AD experiments.

This workplace version is the authoritative evaluator for interactive AD runs.
It reads `*_ad_output.json` files, aggregates all AD texts before/after each
instruction insertion, and outputs a single per-movie row with:

1. Embedding metrics
2. LLM-judge metrics
3. Baseline-vs-instructed comparison metrics
4. Timing / latency metrics

It does not depend on Video-LLaMA runtime code.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
STREAMING_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = STREAMING_ROOT.parent

os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".hf_cache"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(PROJECT_ROOT / ".hf_cache" / "hub"))
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(PROJECT_ROOT / ".hf_cache" / "sentence_transformers"))

if str(STREAMING_ROOT) not in sys.path:
    sys.path.insert(0, str(STREAMING_ROOT))

DEFAULT_EMBEDDING_MODEL = "/mnt/temp_disk/ghl/models/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_LLM_MODEL = "/mnt/disk5new/gcc/models/Qwen2.5-7B-Instruct"

METRIC_METADATA: Dict[str, Dict[str, str]] = {
    "State_Update_Coverage": {
        "group": "State / Control",
        "direction": "↑",
        "meaning": "Fraction of requested control slots reflected in the committed state",
    },
    "Conflict_Override_Accuracy": {
        "group": "State / Control",
        "direction": "↑",
        "meaning": "Fraction of conflicting active slots replaced by the new value",
    },
    "NonConflict_Retention": {
        "group": "State / Control",
        "direction": "↑",
        "meaning": "Fraction of unrelated active slots preserved after an update",
    },
    "Stale_Control_Rate": {
        "group": "State / Control",
        "direction": "↓",
        "meaning": "Fraction of superseded controls still present after an update",
    },
    "Window_Segments": {
        "group": "Evaluation Window",
        "direction": "—",
        "meaning": "Number of eligible AD segments evaluated before the next insertion",
    },
    "Immediate_Shift": {
        "group": "Paired Intervention",
        "direction": "↑",
        "meaning": "Semantic change in the matched segment before versus after the intervention",
    },
    "Immediate_Word_Delta": {
        "group": "Paired Intervention",
        "direction": "—",
        "meaning": "After-minus-before word count in the matched segment",
    },
    "Budget_Compliance_Rate": {
        "group": "Temporal Fit",
        "direction": "↑",
        "meaning": "Fraction of generated ADs within their recorded word budget",
    },
    "Mean_Word_Overflow": {
        "group": "Temporal Fit",
        "direction": "↓",
        "meaning": "Mean number of words above the recorded budget",
    },
    "Immediate_ISR": {
        "group": "LLM Judge",
        "direction": "↑",
        "meaning": "Whether the first eligible post-intervention AD follows the request",
    },
    "Immediate_IFR": {
        "group": "LLM Judge",
        "direction": "↑",
        "meaning": "Requirement-level compliance of the first eligible post-intervention AD",
    },
    "SSS": {
        "group": "Embedding / Shift",
        "direction": "↑",
        "meaning": "Semantic Shift Score: before vs after changed more",
    },
    "SSS_vs_baseline": {
        "group": "Baseline Comparison",
        "direction": "↑",
        "meaning": "Shift from baseline after instruction",
    },
    "ICA": {
        "group": "Embedding / Alignment",
        "direction": "↑",
        "meaning": "Instruction-Content Alignment: after text matches instruction better",
    },
    "IDD": {
        "group": "Embedding / Alignment",
        "direction": "↑",
        "meaning": "Instruction-Driven Divergence: combines alignment and shift",
    },
    "SEC": {
        "group": "Embedding / Control",
        "direction": "↑",
        "meaning": "Side-Effect Control: preserve unrelated content better",
    },
    "SE": {
        "group": "Embedding / Exploration",
        "direction": "↑",
        "meaning": "Semantic Exploration: output moves away from raw instruction wording",
    },
    "Style_Consistency": {
        "group": "Embedding / Style",
        "direction": "↑",
        "meaning": "After-region ADs stay stylistically consistent",
    },
    "Transition_Smoothness": {
        "group": "Embedding / Style",
        "direction": "↑",
        "meaning": "Neighboring ADs transition more smoothly",
    },
    "Persistence_Early": {
        "group": "Embedding / Persistence",
        "direction": "↑",
        "meaning": "Instruction effect appears in early after-region segments",
    },
    "Persistence_Late": {
        "group": "Embedding / Persistence",
        "direction": "↑",
        "meaning": "Instruction effect persists into later after-region segments",
    },
    "Before_Shift": {
        "group": "Causality Check",
        "direction": "↓",
        "meaning": "Change before insertion should stay low",
    },
    "ISR": {
        "group": "LLM Judge",
        "direction": "↑",
        "meaning": "Instruction Success Rate: judge says instruction was followed",
    },
    "IFR": {
        "group": "LLM Judge",
        "direction": "↑",
        "meaning": "Instruction Following Ratio: fraction of sub-requirements satisfied",
    },
    "User_Alignment": {
        "group": "LLM Judge",
        "direction": "↑",
        "meaning": "Judge rating of alignment to user intent",
    },
    "Preference_Accuracy": {
        "group": "LLM Judge",
        "direction": "↑",
        "meaning": "Judge prefers after-version over before-version",
    },
    "MSE": {
        "group": "LLM vs Embedding",
        "direction": "↓",
        "meaning": "Disagreement between LLM alignment and embedding alignment",
    },
    "nDCG": {
        "group": "Ranking",
        "direction": "↑",
        "meaning": "Ranking quality of insertion effects across events",
    },
    "Cumulative_Latency_sec": {
        "group": "Timing",
        "direction": "↓",
        "meaning": "Total generation latency over the whole movie",
    },
    "Avg_Inference_Time_sec": {
        "group": "Timing",
        "direction": "↓",
        "meaning": "Average inference time per AD segment",
    },
    "Median_Inference_Time_sec": {
        "group": "Timing",
        "direction": "↓",
        "meaning": "Median inference time per AD segment",
    },
    "P95_Inference_Time_sec": {
        "group": "Timing",
        "direction": "↓",
        "meaning": "95th percentile inference time",
    },
    "P99_Inference_Time_sec": {
        "group": "Timing",
        "direction": "↓",
        "meaning": "99th percentile inference time",
    },
    "Avg_RTF": {
        "group": "Timing",
        "direction": "↓",
        "meaning": "Average real-time factor: inference time / gap duration",
    },
    "Global_RTF": {
        "group": "Timing",
        "direction": "↓",
        "meaning": "Total generation time divided by total evaluated movie-gap duration",
    },
    "Median_RTF": {
        "group": "Timing",
        "direction": "↓",
        "meaning": "Median real-time factor",
    },
    "Avg_TPOT_sec": {
        "group": "Timing",
        "direction": "↓",
        "meaning": "Average time per output token/word proxy",
    },
    "Throughput_ADs_per_sec": {
        "group": "Timing",
        "direction": "↑",
        "meaning": "Generation throughput in ADs per second",
    },
    "Avg_TTTR_sec": {
        "group": "Timing",
        "direction": "↓",
        "meaning": "Average time-to-text-response after user insertion",
    },
    "Max_TTTR_sec": {
        "group": "Timing",
        "direction": "↓",
        "meaning": "Worst-case time-to-text-response",
    },
    "Min_TTTR_sec": {
        "group": "Timing",
        "direction": "↓",
        "meaning": "Best-case time-to-text-response",
    },
    "Avg_Inference_Before_sec": {
        "group": "Timing",
        "direction": "↓",
        "meaning": "Average inference time before any insertion",
    },
    "Avg_Inference_After_sec": {
        "group": "Timing",
        "direction": "↓",
        "meaning": "Average inference time after insertions begin",
    },
}


class EmbeddingModel:
    """Sentence embedding model with local TF-IDF fallback."""

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL, device: str = "cuda"):
        self._backend = "tfidf"
        self._tfidf = None
        self._vocab: Optional[Any] = None
        try:
            import torch
            from sentence_transformers import SentenceTransformer

            if device.startswith("cuda") and not torch.cuda.is_available():
                device = "cpu"
            print(f"[embedding] Loading {model_name} on {device} ...", flush=True)
            self.model = SentenceTransformer(model_name, device=device)
            self._backend = "st"
            print("[embedding] Model loaded.", flush=True)
        except Exception as e:
            print(f"[embedding] SentenceTransformer unavailable ({type(e).__name__}: {e}).", flush=True)
            print("[embedding] Falling back to local TF-IDF embedding.", flush=True)
            self._init_tfidf()

    def _init_tfidf(self) -> None:
        # A fitted TF-IDF vocabulary from the first pair of texts makes all
        # later out-of-vocabulary movie descriptions look like zero vectors.
        # Hashing avoids cross-segment vocabulary leakage and needs no fitting.
        from sklearn.feature_extraction.text import HashingVectorizer

        self._tfidf = HashingVectorizer(
            n_features=2 ** 16,
            analyzer="word",
            ngram_range=(1, 2),
            alternate_sign=False,
            norm=None,
            dtype=np.float32,
        )

    def _ensure_tfidf_fit(self, texts: List[str]) -> None:
        if self._tfidf.__class__.__name__ == "HashingVectorizer":
            return
        if self._vocab is not None or not texts:
            return
        try:
            self._tfidf.fit(texts)
            self._vocab = self._tfidf.vocabulary_
        except ValueError:
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
            return self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,
            ).astype(np.float32)
        self._ensure_tfidf_fit(texts)
        vecs = self._tfidf.transform(texts).toarray().astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        return vecs / norms

    def cosine_sim(self, text_a: str, text_b: str) -> float:
        emb = self.encode([text_a, text_b])
        return float(np.dot(emb[0], emb[1]))


class LLMJudge:
    """Local Qwen judge for instruction-following metrics."""

    def __init__(self, model_path: str, gpu: int = 0):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if torch.cuda.is_available():
            self.device = f"cuda:{gpu}"
            torch_dtype = torch.bfloat16
        else:
            self.device = "cpu"
            torch_dtype = torch.float32

        print(f"[llm_judge] Loading {model_path} on {self.device} ...", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        ).to(self.device).eval()
        print("[llm_judge] Model loaded.", flush=True)

    def generate(self, prompt: str, max_new_tokens: int = 256, temperature: float = 0.0) -> str:
        import torch

        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature > 0 else 1.0,
                do_sample=(temperature > 0),
            )
        return self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        ).strip()

    def score_1_to_5(self, prompt: str) -> Tuple[int, str]:
        raw = self.generate(prompt, max_new_tokens=128)
        score = 0
        patterns = [
            r"\bscore\s*[:=]?\s*(\d)",
            r"\brating\s*[:=]?\s*(\d)",
            r"\b(\d)\s*/\s*5",
            r"\b(\d)\s*out\s*of\s*5",
            r"[\[:]\s*(\d)\s*[\]]",
        ]
        for pattern in patterns:
            m = re.search(pattern, raw, re.IGNORECASE)
            if m:
                score = int(m.group(1))
                break
        return min(max(score, 0), 5), raw

    def score_binary_passfail(self, prompt: str) -> Tuple[int, str]:
        raw = self.generate(prompt, max_new_tokens=128)
        patterns = [r"\bscore\s*[:=]?\s*(\d)", r"\bresult\s*[:=]?\s*(\d)"]
        for pattern in patterns:
            m = re.search(pattern, raw, re.IGNORECASE)
            if m:
                val = int(m.group(1))
                if val in (1, 2):
                    return val, raw
        # Do not infer a score from arbitrary words in the rationale. An
        # unparseable judge response is recorded as invalid by the caller.
        return 0, raw


def load_ad_output(json_path: Path) -> Dict[str, Any]:
    with json_path.open(encoding="utf-8") as f:
        return json.load(f)


def find_instructed_files(instructed_dir: Path) -> List[Path]:
    return sorted(instructed_dir.glob("*_ad_output.json"))


def find_matching_baseline(baseline_dir: Path, movie_title: str) -> Optional[Path]:
    key = movie_title.lower().strip()
    for f in sorted(baseline_dir.glob("*_ad_output.json")):
        data = load_ad_output(f)
        title = data.get("movie", data.get("movie_title", f.stem.replace("_ad_output", "")))
        if title.lower().strip() == key:
            return f
    return None


def _aggregate_texts(entries: List[Dict[str, Any]], start_idx: int, end_idx: int) -> List[str]:
    texts: List[str] = []
    for e in entries[start_idx:end_idx]:
        t = e.get("ad_text", "").strip()
        if t and not t.startswith("[ERROR"):
            texts.append(t)
    return texts


def _safe_concat(texts: List[str]) -> str:
    return " ".join(t.strip() for t in texts if t and t.strip()).strip()


CONTROL_SLOTS = (
    "language",
    "verbosity",
    "style",
    "focus_primary",
    "focus_character",
    "focus_action",
    "focus_emotion",
    "focus_environment",
    "focus_interaction",
    "format",
    "constraints",
    "others",
)

CATEGORY_SLOT_MAP = {
    "lang_switch": "language",
    "detail_detailed": "verbosity",
    "detail_concise": "verbosity",
    "style": "style",
    "character": "focus_character",
    "event_action": "focus_action",
    "event_romance": "focus_interaction",
    "event_drama": "focus_emotion",
    "event_environment": "focus_environment",
}


def _event_slots(event: Dict[str, Any]) -> List[str]:
    """Recover expected control slots from serialized request components."""
    slots: List[str] = []
    components = event.get("components") or []
    if isinstance(components, list):
        for component in components:
            if not isinstance(component, dict):
                continue
            category = str(component.get("category_id", ""))
            slot = CATEGORY_SLOT_MAP.get(category)
            if slot and slot not in slots:
                slots.append(slot)
    if not slots:
        category = str(event.get("category_id", event.get("category", "")))
        slot = CATEGORY_SLOT_MAP.get(category)
        if slot:
            slots.append(slot)
    return slots or ["others"]


def _state_values(state: Any) -> Dict[str, str]:
    if not isinstance(state, dict):
        return {}
    values: Dict[str, str] = {}
    for slot in CONTROL_SLOTS:
        value = state.get(slot, "")
        if isinstance(value, list):
            value = " | ".join(str(item).strip() for item in value if str(item).strip())
        value = str(value or "").strip()
        if value:
            values[slot] = value
    return values


def _previous_state(events: List[Dict[str, Any]], event_index: int) -> Dict[str, str]:
    if event_index <= 0:
        return {}
    return _state_values(events[event_index - 1].get("prompt_state_after", {}))


def _state_transition_metrics(
    before_state: Dict[str, str],
    after_state: Dict[str, str],
    expected_slots: List[str],
) -> Dict[str, float]:
    expected = list(dict.fromkeys(expected_slots)) or ["others"]
    updated = sum(1 for slot in expected if before_state.get(slot, "") != after_state.get(slot, ""))
    conflicting = [slot for slot in expected if before_state.get(slot, "") and before_state.get(slot, "") != after_state.get(slot, "")]
    overridden = sum(1 for slot in conflicting if after_state.get(slot, "") and after_state.get(slot, "") != before_state.get(slot, ""))
    unrelated = [slot for slot in before_state if slot not in expected]
    retained = sum(1 for slot in unrelated if after_state.get(slot, "") == before_state.get(slot, ""))
    stale = sum(1 for slot in conflicting if after_state.get(slot, "") == before_state.get(slot, ""))
    return {
        "State_Update_Coverage": updated / max(len(expected), 1),
        "Conflict_Override_Accuracy": overridden / max(len(conflicting), 1) if conflicting else 1.0,
        "NonConflict_Retention": retained / max(len(unrelated), 1) if unrelated else 1.0,
        "Stale_Control_Rate": stale / max(len(conflicting), 1) if conflicting else 0.0,
    }


def _word_count(text: str) -> int:
    text = str(text or "").strip()
    if not text:
        return 0
    # English and whitespace-delimited languages use word boundaries. For CJK,
    # fall back to characters because whitespace is not a reliable word unit.
    if re.search(r"[\u3400-\u9fff]", text) and not re.search(r"\b[a-zA-Z]{2,}\b", text):
        return len(re.findall(r"[\u3400-\u9fff]", text))
    return len(re.findall(r"\b\w+\b", text))


def compute_budget_metrics(entries: List[Dict[str, Any]]) -> Dict[str, float]:
    valid = 0
    compliant = 0
    overflow: List[int] = []
    for entry in entries:
        text = str(entry.get("ad_text", "") or "").strip()
        budget = entry.get("max_words")
        if not text or text.startswith("[ERROR") or not isinstance(budget, (int, float)) or budget <= 0:
            continue
        valid += 1
        excess = max(0, _word_count(text) - int(budget))
        overflow.append(excess)
        if excess == 0:
            compliant += 1
    return {
        "Budget_Compliance_Rate": compliant / max(valid, 1),
        "Mean_Word_Overflow": float(np.mean(overflow)) if overflow else 0.0,
        "num_budget_checked": float(valid),
    }


def _event_windows(entries: List[Dict[str, Any]], insertion_events: List[Dict[str, Any]]) -> List[Tuple[int, int]]:
    """Return [start, end) windows bounded by the next insertion."""
    ordered = sorted(enumerate(insertion_events), key=lambda item: int(item[1].get("gap_idx", 0)))
    windows: Dict[int, Tuple[int, int]] = {}
    for pos, (original_idx, event) in enumerate(ordered):
        start = max(0, int(event.get("gap_idx", 0)))
        next_start = len(entries)
        if pos + 1 < len(ordered):
            next_start = max(start + 1, int(ordered[pos + 1][1].get("gap_idx", len(entries))))
        windows[original_idx] = (start, min(next_start, len(entries)))
    return [windows.get(i, (0, 0)) for i in range(len(insertion_events))]


def _pairwise_style_consistency(embedder: EmbeddingModel, texts: List[str]) -> float:
    if len(texts) < 2:
        return 1.0
    embs = embedder.encode(texts)
    sims: List[float] = []
    for i in range(len(embs)):
        for j in range(i + 1, len(embs)):
            sims.append(float(np.dot(embs[i], embs[j])))
    return float(np.mean(sims)) if sims else 1.0


def compute_ica(embedder: EmbeddingModel, instruction: str, after_text: str) -> float:
    if not instruction.strip() or not after_text.strip():
        return 0.0
    return max(0.0, embedder.cosine_sim(instruction, after_text))


def compute_sss(embedder: EmbeddingModel, before_text: str, after_text: str) -> float:
    if not before_text.strip() or not after_text.strip():
        return 0.0
    return 1.0 - embedder.cosine_sim(before_text, after_text)


def compute_idd(ica: float, sss: float, alpha: float = 0.5, beta: float = 0.5) -> float:
    return alpha * ica + beta * sss


def compute_sec(embedder: EmbeddingModel, before_text: str, after_text: str) -> float:
    if not before_text.strip() or not after_text.strip():
        return 1.0
    return max(0.0, embedder.cosine_sim(before_text, after_text))


def compute_se(embedder: EmbeddingModel, instruction: str, after_text: str) -> float:
    if not instruction.strip() or not after_text.strip():
        return 0.0
    return 1.0 - embedder.cosine_sim(instruction, after_text)


def compute_transition_smoothness(embedder: EmbeddingModel, texts: List[str]) -> float:
    if len(texts) < 2:
        return 1.0
    sims: List[float] = []
    for i in range(len(texts) - 1):
        if texts[i].strip() and texts[i + 1].strip():
            sims.append(embedder.cosine_sim(texts[i], texts[i + 1]))
    return float(np.mean(sims)) if sims else 1.0


def compute_ndcg(scores: List[float], k: Optional[int] = None) -> float:
    if not scores:
        return 0.0
    if k is None:
        k = len(scores)
    dcg = sum(score / math.log2(i + 2) for i, score in enumerate(scores[:k]))
    ideal = sorted(scores, reverse=True)[:k]
    idcg = sum(score / math.log2(i + 2) for i, score in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def compute_isr_judge(judge: LLMJudge, instruction: str, after_text: str) -> Tuple[Optional[float], str]:
    prompt = f"""You are evaluating whether a generated Audio Description (AD) follows a specific instruction.

INSTRUCTION: {instruction}

GENERATED AD: {after_text}

Score:
  2 = The AD fully or clearly follows the instruction
  1 = The AD does not follow the instruction

Output format: Score: X
Then give one short reason.
"""
    raw_score, raw = judge.score_binary_passfail(prompt)
    if raw_score == 0:
        return None, raw
    return (1.0 if raw_score >= 2 else 0.0), raw


def compute_ifr_judge(judge: LLMJudge, instruction: str, after_text: str) -> Tuple[Optional[float], str]:
    prompt = f"""You are evaluating how well a generated Audio Description (AD) follows a user instruction.
Break the instruction into separate requirements and check each one.

INSTRUCTION: {instruction}

GENERATED AD: {after_text}

Output format:
Requirement 1: [MET/NOT MET] brief text
Requirement 2: [MET/NOT MET] brief text
Score: X/Y
"""
    raw = judge.generate(prompt, max_new_tokens=256)
    ratio: Optional[float] = None
    m = re.search(r"score\s*[:=]?\s*(\d+)\s*/\s*(\d+)", raw, re.IGNORECASE)
    if m:
        num, den = int(m.group(1)), int(m.group(2))
        ratio = num / max(den, 1)
    return (round(ratio, 4) if ratio is not None else None), raw


def compute_user_alignment(judge: LLMJudge, instruction: str, after_text: str) -> Tuple[int, str]:
    prompt = f"""You are evaluating how well a generated Audio Description (AD) aligns with the user's intent.

USER INSTRUCTION: {instruction}

GENERATED AD: {after_text}

Rate alignment from 1 to 5:
  5 = Perfectly aligned
  4 = Mostly aligned
  3 = Partially aligned
  2 = Weakly aligned
  1 = Not aligned

Output format: Score: X
Then give one short reason.
"""
    return judge.score_1_to_5(prompt)


def compute_preference_accuracy(
    judge: LLMJudge,
    instruction: str,
    before_text: str,
    after_text: str,
) -> Tuple[str, str]:
    # Deterministically randomize A/B position per example to avoid a fixed
    # position bias in the judge while keeping runs reproducible.
    swap = sum(ord(ch) for ch in (instruction + before_text + after_text)) % 2 == 1
    version_a = after_text if swap else before_text
    version_b = before_text if swap else after_text
    prompt = f"""You are comparing two Audio Description (AD) versions for the same movie context.
Decide which version better follows the user instruction.

USER INSTRUCTION: {instruction}

AD VERSION A: {version_a}
AD VERSION B: {version_b}

Choose:
  1 = Version A is better
  2 = Version B is better
  3 = About the same

Output format: Choice: X
Then give one short reason.
"""
    raw = judge.generate(prompt, max_new_tokens=128)
    choice = "invalid"
    m = re.search(r"choice\s*[:=]?\s*(\d)", raw, re.IGNORECASE)
    if m:
        val = int(m.group(1))
        if val == 2:
            choice = "before" if swap else "after"
        elif val == 1:
            choice = "after" if swap else "before"
    return choice, raw


def compute_mse(llm_score: float, embedding_score: float) -> float:
    llm_norm = (llm_score - 1.0) / 4.0
    return (llm_norm - embedding_score) ** 2


def _preference_to_score(choice: str) -> Optional[float]:
    if choice == "after":
        return 1.0
    if choice == "tie":
        return 0.5
    if choice == "invalid":
        return None
    return 0.0


def _mean_std(rows: List[Dict[str, Any]], key: str) -> Tuple[Optional[float], Optional[float]]:
    vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
    if not vals:
        return None, None
    return round(float(np.mean(vals)), 4), round(float(np.std(vals)), 4)


def _round_optional(value: Any, digits: int = 4) -> Optional[float]:
    return round(float(value), digits) if isinstance(value, (int, float)) else None


def compute_timing_metrics(
    entries: List[Dict[str, Any]],
    insertion_events: List[Dict[str, Any]],
) -> Dict[str, Any]:
    if not entries:
        return {}

    all_inf_times: List[float] = []
    all_rtf: List[float] = []
    all_tpot: List[float] = []
    total_gap_duration_sec = 0.0

    for e in entries:
        inf_time = e.get("inference_time_sec", 0.0)
        gap_dur = e.get("gap_duration_sec", 0.0)
        ad_text = e.get("ad_text", "").strip()
        if inf_time <= 0 or ad_text.startswith("[ERROR"):
            continue

        all_inf_times.append(inf_time)
        if gap_dur > 0:
            all_rtf.append(inf_time / gap_dur)
            total_gap_duration_sec += gap_dur

        word_count = len(ad_text.split())
        if word_count > 0:
            all_tpot.append(inf_time / word_count)

    timing: Dict[str, Any] = {
        "num_valid_segments": len(all_inf_times),
        "Cumulative_Latency_sec": round(sum(all_inf_times), 3),
        "Avg_Inference_Time_sec": round(float(np.mean(all_inf_times)), 3) if all_inf_times else 0.0,
        "Std_Inference_Time_sec": round(float(np.std(all_inf_times)), 3) if all_inf_times else 0.0,
        "Median_Inference_Time_sec": round(float(np.median(all_inf_times)), 3) if all_inf_times else 0.0,
        "P95_Inference_Time_sec": round(float(np.percentile(all_inf_times, 95)), 3) if all_inf_times else 0.0,
        "P99_Inference_Time_sec": round(float(np.percentile(all_inf_times, 99)), 3) if all_inf_times else 0.0,
        "Avg_RTF": round(float(np.mean(all_rtf)), 4) if all_rtf else 0.0,
        "Global_RTF": round(float(sum(all_inf_times) / total_gap_duration_sec), 4) if total_gap_duration_sec > 0 else 0.0,
        "Median_RTF": round(float(np.median(all_rtf)), 4) if all_rtf else 0.0,
        "Avg_TPOT_sec": round(float(np.mean(all_tpot)), 4) if all_tpot else 0.0,
        "Throughput_ADs_per_sec": round(len(all_inf_times) / sum(all_inf_times), 4) if sum(all_inf_times) > 0 else 0.0,
    }
    timing.update(compute_budget_metrics(entries))

    # These fields are optional so old experiment JSON remains compatible. The
    # end-to-end runner can provide them for true playback-level evaluation.
    deadline_values = [
        e.get("deadline_met")
        for e in entries
        if isinstance(e.get("deadline_met"), bool)
    ]
    if deadline_values:
        timing["Deadline_Satisfaction_Rate"] = float(np.mean(deadline_values))
    speech_overruns = [
        float(e.get("speech_overrun_sec", 0.0))
        for e in entries
        if isinstance(e.get("speech_overrun_sec"), (int, float))
    ]
    if speech_overruns:
        timing["Mean_Speech_Overrun_sec"] = float(np.mean(speech_overruns))
        timing["Speech_Overrun_Rate"] = float(np.mean([value > 0 for value in speech_overruns]))

    tttr_list: List[float] = []
    for event in insertion_events:
        gap_idx = event.get("gap_idx", 0)
        for e in entries[gap_idx:]:
            inf_time = e.get("inference_time_sec", 0.0)
            ad_text = e.get("ad_text", "").strip()
            if inf_time > 0 and not ad_text.startswith("[ERROR"):
                tttr_list.append(inf_time)
                break

    timing["TTTR_per_insertion_sec"] = [round(t, 3) for t in tttr_list]
    timing["Avg_TTTR_sec"] = round(float(np.mean(tttr_list)), 3) if tttr_list else 0.0
    timing["Max_TTTR_sec"] = round(float(np.max(tttr_list)), 3) if tttr_list else 0.0
    timing["Min_TTTR_sec"] = round(float(np.min(tttr_list)), 3) if tttr_list else 0.0

    if insertion_events:
        first_insertion_idx = min(e.get("gap_idx", 0) for e in insertion_events)
        before_times = [
            e.get("inference_time_sec", 0.0)
            for e in entries[:first_insertion_idx]
            if e.get("inference_time_sec", 0.0) > 0
        ]
        after_times = [
            e.get("inference_time_sec", 0.0)
            for e in entries[first_insertion_idx:]
            if e.get("inference_time_sec", 0.0) > 0
        ]
        timing["Avg_Inference_Before_sec"] = round(float(np.mean(before_times)), 3) if before_times else 0.0
        timing["Avg_Inference_After_sec"] = round(float(np.mean(after_times)), 3) if after_times else 0.0

    return timing


def compute_self_metrics(
    embedder: EmbeddingModel,
    entries: List[Dict[str, Any]],
    insertion_events: List[Dict[str, Any]],
    insertion_records: Optional[List[Dict[str, Any]]] = None,
    judge: Optional[LLMJudge] = None,
    segment_judge: bool = False,
) -> List[Dict[str, Any]]:
    if not insertion_events or not entries:
        return []

    n = len(entries)
    windows = _event_windows(entries, insertion_events)
    paired = {
        int(record.get("insertion_id")): record
        for record in (insertion_records or [])
        if isinstance(record, dict) and record.get("insertion_id") is not None
    }
    rows: List[Dict[str, Any]] = []

    for idx, event in enumerate(insertion_events, start=1):
        event_index = idx - 1
        gap_idx, window_end = windows[event_index]
        instruction = event.get("instruction_text", "").strip()
        category = event.get("category", "")
        insertion_id = event.get("insertion_id", idx)

        before_texts = _aggregate_texts(entries, max(0, gap_idx - 1), gap_idx)
        after_texts = _aggregate_texts(entries, gap_idx, window_end)
        if not after_texts:
            continue

        before_concat = _safe_concat(before_texts)
        after_concat = _safe_concat(after_texts)
        half = len(after_texts) // 2
        early_after = _safe_concat(after_texts[:half]) if half > 0 else ""
        late_after = _safe_concat(after_texts[half:]) if half > 0 else ""
        record = paired.get(int(insertion_id), {})
        paired_before = str(record.get("text_before", "") or "").strip()
        paired_after = str(record.get("text_after", "") or "").strip()
        state_before = _previous_state(insertion_events, event_index)
        state_after = _state_values(event.get("prompt_state_after", {}))
        state_metrics = _state_transition_metrics(state_before, state_after, _event_slots(event))
        immediate_shift = compute_sss(embedder, paired_before, paired_after) if paired_before and paired_after else compute_sss(embedder, before_texts[-1] if before_texts else "", after_texts[0])

        row: Dict[str, Any] = {
            "insertion_id": insertion_id,
            "gap_idx": gap_idx,
            "timestamp_sec": event.get("timestamp_sec", 0.0),
            "category": category,
            "instruction": instruction,
            "num_before_segments": len(before_texts),
            "num_after_segments": len(after_texts),
            "Window_Segments": len(after_texts),
            "SSS": round(immediate_shift, 4) if before_concat else 0.0,
            "ICA": round(compute_ica(embedder, instruction, after_concat), 4),
            "Style_Consistency": round(_pairwise_style_consistency(embedder, after_texts), 4),
            "Transition_Smoothness": round(compute_transition_smoothness(embedder, after_texts), 4),
            # Populated with local-window compliance below when segment_judge is
            # enabled. Semantic shift is not persistence and is not used here.
            "Persistence_Early": None,
            "Persistence_Late": None,
            "Before_Shift": 0.0,
            "Immediate_Shift": round(immediate_shift, 4),
            "Immediate_Word_Delta": _word_count(paired_after) - _word_count(paired_before) if paired_before and paired_after else 0,
            **{key: round(value, 4) for key, value in state_metrics.items()},
        }
        row["IDD"] = round(compute_idd(row["ICA"], row["SSS"]), 4)
        row["SEC"] = round(compute_sec(embedder, before_concat, after_concat), 4) if before_concat else 1.0
        row["SE"] = round(compute_se(embedder, instruction, after_concat), 4)

        if judge is not None:
            print(f"    [llm self {idx}/{len(insertion_events)}] insertion_id={insertion_id}", flush=True)
            isr, isr_raw = compute_isr_judge(judge, instruction, after_concat)
            ifr, ifr_raw = compute_ifr_judge(judge, instruction, after_concat)
            ua, ua_raw = compute_user_alignment(judge, instruction, after_concat)
            pref_choice, pref_raw = compute_preference_accuracy(judge, instruction, before_concat or after_concat, after_concat)
            row["ISR"] = _round_optional(isr)
            row["IFR"] = _round_optional(ifr)
            row["User_Alignment"] = ua if ua > 0 else None
            row["Preference_Accuracy"] = _round_optional(_preference_to_score(pref_choice))
            row["Preference_Choice"] = pref_choice
            row["MSE"] = round(compute_mse(float(ua), row["ICA"]), 4) if ua > 0 else None
            if segment_judge and after_texts:
                immediate_isr, _ = compute_isr_judge(judge, instruction, after_texts[0])
                immediate_ifr, _ = compute_ifr_judge(judge, instruction, after_texts[0])
                row["Immediate_ISR"] = _round_optional(immediate_isr)
                row["Immediate_IFR"] = _round_optional(immediate_ifr)
                early_isr, _ = compute_isr_judge(judge, instruction, early_after or after_texts[0])
                late_isr, _ = compute_isr_judge(judge, instruction, late_after or after_texts[-1])
                row["Persistence_Early"] = _round_optional(early_isr)
                row["Persistence_Late"] = _round_optional(late_isr)
            else:
                row["Immediate_ISR"] = None
                row["Immediate_IFR"] = None
            row["ISR_rationale"] = isr_raw[:300]
            row["IFR_rationale"] = ifr_raw[:300]
            row["User_Alignment_rationale"] = ua_raw[:300]
            row["Preference_Accuracy_rationale"] = pref_raw[:300]
        else:
            row["ISR"] = None
            row["IFR"] = None
            row["User_Alignment"] = None
            row["Preference_Accuracy"] = None
            row["Preference_Choice"] = ""
            row["MSE"] = None
            row["Immediate_ISR"] = None
            row["Immediate_IFR"] = None

        rows.append(row)

    return rows


def compute_baseline_metrics(
    embedder: EmbeddingModel,
    baseline_entries: List[Dict[str, Any]],
    instructed_entries: List[Dict[str, Any]],
    insertion_events: List[Dict[str, Any]],
    judge: Optional[LLMJudge] = None,
    segment_judge: bool = False,
) -> List[Dict[str, Any]]:
    if not insertion_events or not baseline_entries or not instructed_entries:
        return []

    n = min(len(baseline_entries), len(instructed_entries))
    windows = _event_windows(instructed_entries, insertion_events)
    rows: List[Dict[str, Any]] = []

    for idx, event in enumerate(insertion_events, start=1):
        gap_idx, window_end = windows[idx - 1]
        window_end = min(window_end, n)
        instruction = event.get("instruction_text", "").strip()
        category = event.get("category", "")
        insertion_id = event.get("insertion_id", idx)

        base_before_texts = _aggregate_texts(baseline_entries, max(0, gap_idx - 1), gap_idx)
        inst_before_texts = _aggregate_texts(instructed_entries, max(0, gap_idx - 1), gap_idx)
        base_after_texts = _aggregate_texts(baseline_entries, gap_idx, window_end)
        inst_after_texts = _aggregate_texts(instructed_entries, gap_idx, window_end)
        if not inst_after_texts:
            continue

        base_before_concat = _safe_concat(base_before_texts)
        inst_before_concat = _safe_concat(inst_before_texts)
        base_after_concat = _safe_concat(base_after_texts)
        inst_after_concat = _safe_concat(inst_after_texts)

        half = len(inst_after_texts) // 2
        inst_early = _safe_concat(inst_after_texts[:half]) if half > 0 else ""
        inst_late = _safe_concat(inst_after_texts[half:]) if half > 0 else ""
        base_early = _safe_concat(base_after_texts[:half]) if half > 0 else ""
        base_late = _safe_concat(base_after_texts[half:]) if half > 0 else ""

        sss_vs_baseline = compute_sss(embedder, base_after_concat, inst_after_concat) if base_after_concat else 0.0
        ica = compute_ica(embedder, instruction, inst_after_concat)

        row: Dict[str, Any] = {
            "insertion_id": insertion_id,
            "gap_idx": gap_idx,
            "timestamp_sec": event.get("timestamp_sec", 0.0),
            "category": category,
            "instruction": instruction,
            "SSS_vs_baseline": round(sss_vs_baseline, 4),
            "Window_Segments": len(inst_after_texts),
            "ICA": round(ica, 4),
            "IDD": round(compute_idd(ica, sss_vs_baseline), 4),
            "SEC": round(compute_sec(embedder, base_after_concat, inst_after_concat), 4) if base_after_concat else 1.0,
            "SE": round(compute_se(embedder, instruction, inst_after_concat), 4),
            "Style_Consistency": round(_pairwise_style_consistency(embedder, inst_after_texts), 4),
            "Transition_Smoothness": round(compute_transition_smoothness(embedder, inst_after_texts), 4),
            "Persistence_Early": None,
            "Persistence_Late": None,
            "Before_Shift": round(compute_sss(embedder, base_before_concat, inst_before_concat), 4) if base_before_concat and inst_before_concat else 0.0,
        }

        if judge is not None:
            print(f"    [llm baseline {idx}/{len(insertion_events)}] insertion_id={insertion_id}", flush=True)
            isr, isr_raw = compute_isr_judge(judge, instruction, inst_after_concat)
            ifr, ifr_raw = compute_ifr_judge(judge, instruction, inst_after_concat)
            ua, ua_raw = compute_user_alignment(judge, instruction, inst_after_concat)
            pref_choice, pref_raw = compute_preference_accuracy(
                judge,
                instruction,
                base_after_concat or inst_after_concat,
                inst_after_concat,
            )
            row["ISR"] = _round_optional(isr)
            row["IFR"] = _round_optional(ifr)
            row["User_Alignment"] = ua if ua > 0 else None
            row["Preference_Accuracy"] = _round_optional(_preference_to_score(pref_choice))
            row["Preference_Choice"] = pref_choice
            row["MSE"] = round(compute_mse(float(ua), row["ICA"]), 4) if ua > 0 else None
            if segment_judge and inst_after_texts:
                immediate_isr, _ = compute_isr_judge(judge, instruction, inst_after_texts[0])
                immediate_ifr, _ = compute_ifr_judge(judge, instruction, inst_after_texts[0])
                row["Immediate_ISR"] = _round_optional(immediate_isr)
                row["Immediate_IFR"] = _round_optional(immediate_ifr)
                early_isr, _ = compute_isr_judge(judge, instruction, inst_early or inst_after_texts[0])
                late_isr, _ = compute_isr_judge(judge, instruction, inst_late or inst_after_texts[-1])
                row["Persistence_Early"] = _round_optional(early_isr)
                row["Persistence_Late"] = _round_optional(late_isr)
            else:
                row["Immediate_ISR"] = None
                row["Immediate_IFR"] = None
            row["ISR_rationale"] = isr_raw[:300]
            row["IFR_rationale"] = ifr_raw[:300]
            row["User_Alignment_rationale"] = ua_raw[:300]
            row["Preference_Accuracy_rationale"] = pref_raw[:300]
        else:
            row["ISR"] = None
            row["IFR"] = None
            row["User_Alignment"] = None
            row["Preference_Accuracy"] = None
            row["Preference_Choice"] = ""
            row["MSE"] = None
            row["Immediate_ISR"] = None
            row["Immediate_IFR"] = None

        rows.append(row)

    return rows


def eval_one_movie(
    embedder: EmbeddingModel,
    movie_title: str,
    instructed_path: Path,
    output_dir: Path,
    baseline_path: Optional[Path] = None,
    judge: Optional[LLMJudge] = None,
    segment_judge: bool = False,
) -> Optional[Dict[str, Any]]:
    instructed_data = load_ad_output(instructed_path)
    experiment_path = instructed_path.with_name(
        instructed_path.name.replace("_ad_output.json", "_experiment.json")
    )
    experiment_data: Dict[str, Any] = {}
    if experiment_path.exists():
        try:
            experiment_data = load_ad_output(experiment_path)
        except Exception as exc:
            print(f"  [warn] Could not load companion experiment JSON: {exc}", flush=True)
    instructed_entries = instructed_data.get("ad_entries", [])
    insertion_events = instructed_data.get("insertion_events", [])
    insertion_records = experiment_data.get("insertion_records", [])

    if not instructed_entries:
        print("  [skip] No instructed AD entries", flush=True)
        return None
    if not insertion_events:
        print("  [skip] No insertion events", flush=True)
        return None

    self_metrics = compute_self_metrics(
        embedder,
        instructed_entries,
        insertion_events,
        insertion_records=insertion_records,
        judge=judge,
        segment_judge=segment_judge,
    )

    baseline_metrics: List[Dict[str, Any]] = []
    baseline_entries: List[Dict[str, Any]] = []
    if baseline_path and baseline_path.exists():
        baseline_data = load_ad_output(baseline_path)
        baseline_entries = baseline_data.get("ad_entries", [])
        if baseline_entries:
            baseline_metrics = compute_baseline_metrics(
                embedder,
                baseline_entries,
                instructed_entries,
                insertion_events,
                judge=judge,
                segment_judge=segment_judge,
            )

    timing_metrics = compute_timing_metrics(instructed_entries, insertion_events)

    agg: Dict[str, Any] = {
        "movie": movie_title,
        "method": instructed_data.get("method", "unknown"),
        "num_segments_instructed": len(instructed_entries),
        "num_segments_baseline": len(baseline_entries),
        "num_insertions": len(insertion_events),
        "used_companion_experiment_json": bool(insertion_records),
    }

    self_keys = [
        "SSS",
        "ICA",
        "IDD",
        "SEC",
        "SE",
        "Style_Consistency",
        "Transition_Smoothness",
        "Persistence_Early",
        "Persistence_Late",
        "Before_Shift",
        "ISR",
        "IFR",
        "User_Alignment",
        "Preference_Accuracy",
        "MSE",
        "Window_Segments",
        "Immediate_Shift",
        "Immediate_Word_Delta",
        "State_Update_Coverage",
        "Conflict_Override_Accuracy",
        "NonConflict_Retention",
        "Stale_Control_Rate",
        "Immediate_ISR",
        "Immediate_IFR",
    ]
    for key in self_keys:
        mean_val, std_val = _mean_std(self_metrics, key)
        agg[f"self_mean_{key}"] = mean_val
        agg[f"self_std_{key}"] = std_val

    # nDCG requires an external relevance ordering. Chronological insertion
    # order alone is not a relevance target, so do not report a pseudo-nDCG.
    agg["self_nDCG"] = None

    if baseline_metrics:
        bl_keys = [
            "SSS_vs_baseline",
            "ICA",
            "IDD",
            "SEC",
            "SE",
            "Style_Consistency",
            "Transition_Smoothness",
            "Persistence_Early",
            "Persistence_Late",
            "Before_Shift",
            "ISR",
            "IFR",
            "User_Alignment",
            "Preference_Accuracy",
            "MSE",
            "Window_Segments",
            "Immediate_Shift",
            "Immediate_Word_Delta",
            "State_Update_Coverage",
            "Conflict_Override_Accuracy",
            "NonConflict_Retention",
            "Stale_Control_Rate",
            "Immediate_ISR",
            "Immediate_IFR",
        ]
        for key in bl_keys:
            mean_val, std_val = _mean_std(baseline_metrics, key)
            agg[f"bl_mean_{key}"] = mean_val
            agg[f"bl_std_{key}"] = std_val

        agg["bl_nDCG"] = None

    agg.update(timing_metrics)

    movie_output = output_dir / f"{movie_title.replace(' ', '_')}_eval.json"
    with movie_output.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "movie": movie_title,
                "method": instructed_data.get("method", "unknown"),
                "aggregate": agg,
                "self_comparison": self_metrics,
                "baseline_comparison": baseline_metrics,
                "timing_metrics": timing_metrics,
                "insertion_events": insertion_events,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    return agg


def write_summary_csv(all_results: List[Dict[str, Any]], output_dir: Path, has_baseline: bool) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = output_dir / f"eval_interactive_{ts}.summary.csv"

    self_keys = [
        "SSS",
        "ICA",
        "IDD",
        "SEC",
        "SE",
        "Style_Consistency",
        "Transition_Smoothness",
        "Persistence_Early",
        "Persistence_Late",
        "Before_Shift",
        "ISR",
        "IFR",
        "User_Alignment",
        "Preference_Accuracy",
        "MSE",
        "Window_Segments",
        "Immediate_Shift",
        "Immediate_Word_Delta",
        "State_Update_Coverage",
        "Conflict_Override_Accuracy",
        "NonConflict_Retention",
        "Stale_Control_Rate",
        "Immediate_ISR",
        "Immediate_IFR",
    ]
    bl_keys = [
        "SSS_vs_baseline",
        "ICA",
        "IDD",
        "SEC",
        "SE",
        "Style_Consistency",
        "Transition_Smoothness",
        "Persistence_Early",
        "Persistence_Late",
        "Before_Shift",
        "ISR",
        "IFR",
        "User_Alignment",
        "Preference_Accuracy",
        "MSE",
        "Window_Segments",
        "Immediate_ISR",
        "Immediate_IFR",
    ]
    timing_keys = [
        "Cumulative_Latency_sec",
        "Avg_Inference_Time_sec",
        "Median_Inference_Time_sec",
        "P95_Inference_Time_sec",
        "P99_Inference_Time_sec",
        "Avg_RTF",
        "Global_RTF",
        "Median_RTF",
        "Avg_TPOT_sec",
        "Throughput_ADs_per_sec",
        "Avg_TTTR_sec",
        "Max_TTTR_sec",
        "Min_TTTR_sec",
        "Avg_Inference_Before_sec",
        "Avg_Inference_After_sec",
        "Budget_Compliance_Rate",
        "Mean_Word_Overflow",
        "Deadline_Satisfaction_Rate",
        "Mean_Speech_Overrun_sec",
        "Speech_Overrun_Rate",
    ]

    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "mean", "std", "num_movies"])

        writer.writerow(["--- Self Comparison (before vs after within instructed) ---", "", "", ""])
        for key in self_keys:
            vals = [r[f"self_mean_{key}"] for r in all_results if r.get(f"self_mean_{key}") is not None]
            if vals:
                writer.writerow([f"self_{key}", round(float(np.mean(vals)), 4), round(float(np.std(vals)), 4), len(vals)])

        if has_baseline:
            writer.writerow(["--- Baseline Comparison (instructed vs no-instruction) ---", "", "", ""])
            for key in bl_keys:
                vals = [r[f"bl_mean_{key}"] for r in all_results if r.get(f"bl_mean_{key}") is not None]
                if vals:
                    writer.writerow([f"bl_{key}", round(float(np.mean(vals)), 4), round(float(np.std(vals)), 4), len(vals)])

        writer.writerow(["--- Timing / Latency Metrics ---", "", "", ""])
        for key in timing_keys:
            vals = [r[key] for r in all_results if r.get(key) not in (None, "", [], 0.0)]
            if vals and all(isinstance(v, (int, float)) for v in vals):
                writer.writerow([key, round(float(np.mean(vals)), 4), round(float(np.std(vals)), 4), len(vals)])

    return summary_path


def write_readable_csv(all_results: List[Dict[str, Any]], output_dir: Path, has_baseline: bool) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"eval_interactive_{ts}.readable.csv"

    def _fmt_scalar(key: Optional[str], decimals: int = 4) -> str:
        if not key:
            return "N/A"
        vals = [r[key] for r in all_results if r.get(key) is not None]
        if not vals:
            return "N/A"
        return f"{round(float(np.mean(vals)), decimals)}"

    def _fmt_percent(key: Optional[str], decimals: int = 1) -> str:
        if not key:
            return "N/A"
        vals = [r[key] for r in all_results if r.get(key) is not None]
        if not vals:
            return "N/A"
        return f"{round(float(np.mean(vals)) * 100.0, decimals)}%"

    def _fmt_mean_std(mean_key: Optional[str], std_key: Optional[str]) -> str:
        if not mean_key:
            return "N/A"
        mean_vals = [r[mean_key] for r in all_results if r.get(mean_key) is not None]
        if not mean_vals:
            return "N/A"
        mean_val = round(float(np.mean(mean_vals)), 4)
        if std_key:
            std_vals = [r[std_key] for r in all_results if r.get(std_key) is not None]
            std_val = round(float(np.mean(std_vals)), 4) if std_vals else 0.0
        else:
            std_val = round(float(np.std(mean_vals)), 4)
        return f"{mean_val} ± {std_val}"

    def _fmt_agg_scalar(mean_key: Optional[str], decimals: int = 4) -> str:
        if not mean_key:
            return "N/A"
        mean_vals = [r[mean_key] for r in all_results if r.get(mean_key) is not None]
        if not mean_vals:
            return "N/A"
        return f"{round(float(np.mean(mean_vals)), decimals)}"

    def _fmt_agg_percent(mean_key: Optional[str], decimals: int = 1) -> str:
        if not mean_key:
            return "N/A"
        mean_vals = [r[mean_key] for r in all_results if r.get(mean_key) is not None]
        if not mean_vals:
            return "N/A"
        return f"{round(float(np.mean(mean_vals)) * 100.0, decimals)}%"

    category_specs = [
        (
            "State and Selective Control",
            "Whether the requested state transition is applied locally without stale or collateral controls",
            [
                {"metric": "State_Update_Coverage", "after": ("self_mean_State_Update_Coverage", "self_std_State_Update_Coverage")},
                {"metric": "Conflict_Override_Accuracy", "after": ("self_mean_Conflict_Override_Accuracy", "self_std_Conflict_Override_Accuracy")},
                {"metric": "NonConflict_Retention", "after": ("self_mean_NonConflict_Retention", "self_std_NonConflict_Retention")},
                {"metric": "Stale_Control_Rate", "after": ("self_mean_Stale_Control_Rate", "self_std_Stale_Control_Rate")},
            ],
        ),
        (
            "Instruction Following",
            "Whether the outputs actually follow user requests",
            [
                {"metric": "ICA", "after": ("self_mean_ICA", "self_std_ICA")},
                {"metric": "ISR", "after_scalar_pct": "self_mean_ISR"},
                {"metric": "IFR", "after_scalar": "self_mean_IFR"},
                {"metric": "User_Alignment", "after": ("self_mean_User_Alignment", "self_std_User_Alignment")},
                {
                    "metric": "Preference_Accuracy",
                    "before_after_scalar": "self_mean_Preference_Accuracy",
                    "vs_baseline_scalar": "bl_mean_Preference_Accuracy",
                },
                {"metric": "Immediate_ISR", "after_scalar_pct": "self_mean_Immediate_ISR"},
                {"metric": "Immediate_IFR", "after_scalar": "self_mean_Immediate_IFR"},
            ],
        ),
        (
            "Semantic Shift",
            "How strongly the instruction changes the content in the desired direction",
            [
                {"metric": "SSS", "before_after": ("self_mean_SSS", "self_std_SSS")},
                {"metric": "SSS_vs_baseline", "vs_baseline": ("bl_mean_SSS_vs_baseline", "bl_std_SSS_vs_baseline")},
                {
                    "metric": "IDD",
                    "before_after": ("self_mean_IDD", "self_std_IDD"),
                    "vs_baseline": ("bl_mean_IDD", "bl_std_IDD"),
                },
                {
                    "metric": "Persistence_Early",
                    "before_after": ("self_mean_Persistence_Early", "self_std_Persistence_Early"),
                    "vs_baseline": ("bl_mean_Persistence_Early", "bl_std_Persistence_Early"),
                },
                {
                    "metric": "Persistence_Late",
                    "before_after": ("self_mean_Persistence_Late", "self_std_Persistence_Late"),
                    "vs_baseline": ("bl_mean_Persistence_Late", "bl_std_Persistence_Late"),
                },
                {"metric": "Immediate_Shift", "before_after": ("self_mean_Immediate_Shift", "self_std_Immediate_Shift")},
                {"metric": "Window_Segments", "after_scalar": "self_mean_Window_Segments"},
            ],
        ),
        (
            "Style Stability",
            "Whether the AD remains coherent and stylistically stable after insertion",
            [
                {
                    "metric": "SEC",
                    "before_after": ("self_mean_SEC", "self_std_SEC"),
                    "vs_baseline": ("bl_mean_SEC", "bl_std_SEC"),
                },
                {"metric": "Style_Consistency", "after": ("self_mean_Style_Consistency", "self_std_Style_Consistency")},
                {"metric": "Transition_Smoothness", "after": ("self_mean_Transition_Smoothness", "self_std_Transition_Smoothness")},
                {"metric": "Before_Shift", "vs_baseline": ("bl_mean_Before_Shift", "bl_std_Before_Shift")},
                {"metric": "MSE", "after": ("self_mean_MSE", "self_std_MSE")},
            ],
        ),
        (
            "Runtime",
            "Practical deployment cost and responsiveness",
            [
                {"metric": "Avg_Inference_Time_sec", "after_scalar": "Avg_Inference_Time_sec"},
                {"metric": "P95_Inference_Time_sec", "after_scalar": "P95_Inference_Time_sec"},
                {"metric": "Avg_RTF", "after_scalar": "Avg_RTF"},
                {"metric": "Global_RTF", "after_scalar": "Global_RTF"},
                {"metric": "Throughput_ADs_per_sec", "after_scalar": "Throughput_ADs_per_sec"},
                {"metric": "Avg_TTTR_sec", "after_scalar": "Avg_TTTR_sec"},
                {"metric": "Cumulative_Latency_sec", "after_scalar": "Cumulative_Latency_sec"},
                {"metric": "Budget_Compliance_Rate", "after_scalar_pct": "Budget_Compliance_Rate"},
                {"metric": "Mean_Word_Overflow", "after_scalar": "Mean_Word_Overflow"},
                {"metric": "Deadline_Satisfaction_Rate", "after_scalar_pct": "Deadline_Satisfaction_Rate"},
            ],
        ),
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "category_meaning",
            "metric",
            "direction",
            "metric_meaning",
            "插入后AD自身评分",
            "同片前后变化评分",
            "相对无指令基线评分",
        ])

        for category, category_meaning, metrics in category_specs:
            writer.writerow([category, "", "", "", "", "", ""])
            first_metric = True
            for spec in metrics:
                metric = spec["metric"]
                meta = METRIC_METADATA.get(metric, {})
                writer.writerow([
                    category_meaning if first_metric else "",
                    metric,
                    meta.get("direction", ""),
                    meta.get("meaning", ""),
                    (
                        _fmt_mean_std(*spec["after"]) if "after" in spec else
                        _fmt_agg_percent(spec.get("after_scalar_pct")) if "after_scalar_pct" in spec else
                        _fmt_agg_scalar(spec.get("after_scalar")) if "after_scalar" in spec else
                        _fmt_scalar(spec.get("after_direct")) if "after_direct" in spec else
                        "N/A"
                    ),
                    (
                        _fmt_mean_std(*spec["before_after"]) if "before_after" in spec else
                        _fmt_agg_percent(spec.get("before_after_scalar_pct")) if "before_after_scalar_pct" in spec else
                        _fmt_agg_scalar(spec.get("before_after_scalar")) if "before_after_scalar" in spec else
                        _fmt_scalar(spec.get("before_after_direct")) if "before_after_direct" in spec else
                        "N/A"
                    ),
                    (
                        _fmt_mean_std(*spec["vs_baseline"]) if "vs_baseline" in spec else
                        _fmt_agg_percent(spec.get("vs_baseline_scalar_pct")) if "vs_baseline_scalar_pct" in spec else
                        _fmt_agg_scalar(spec.get("vs_baseline_scalar")) if "vs_baseline_scalar" in spec else
                        _fmt_scalar(spec.get("vs_baseline_direct")) if "vs_baseline_direct" in spec else
                        "N/A"
                    ) if has_baseline else "N/A",
                ])
                first_metric = False
            writer.writerow(["", "", "", "", "", "", ""])

    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Final aggregate evaluator for interactive AD experiments.")
    parser.add_argument("--instructed-dir", required=True, help="Directory containing instructed *_ad_output.json files")
    parser.add_argument("--baseline-dir", default=None, help="Optional baseline *_ad_output.json directory")
    parser.add_argument("--output-dir", default=None, help="Output directory; default is <instructed-dir>/eval_results/")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL, help="Sentence embedding model path/name")
    parser.add_argument("--device", default="cuda", help="Embedding device")
    parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL, help="Qwen judge model path")
    parser.add_argument("--judge-gpu", type=int, default=0, help="GPU index for the LLM judge")
    parser.add_argument("--skip-llm", action="store_true", help="Skip all LLM judge metrics")
    parser.add_argument(
        "--segment-judge",
        action="store_true",
        help="Also judge the first eligible post-instruction AD for immediate compliance (more LLM calls)",
    )
    args = parser.parse_args()

    instructed_dir = Path(args.instructed_dir)
    baseline_dir = Path(args.baseline_dir) if args.baseline_dir else None
    output_dir = Path(args.output_dir) if args.output_dir else instructed_dir / "eval_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60, flush=True)
    print("INTERACTIVE AD EVALUATION (Final Workplace Version)", flush=True)
    print("=" * 60, flush=True)
    print(f"  Instructed: {instructed_dir}", flush=True)
    print(f"  Baseline:   {baseline_dir or 'N/A'}", flush=True)
    print(f"  Output:     {output_dir}", flush=True)
    print(f"  Embedding:  {args.embedding_model}", flush=True)
    print(f"  LLM Judge:  {'SKIPPED' if args.skip_llm else args.llm_model}", flush=True)
    print(f"  Segment Judge: {'ON' if args.segment_judge else 'OFF'}", flush=True)
    print()

    inst_files = find_instructed_files(instructed_dir)
    print(f"Found {len(inst_files)} instructed files", flush=True)
    if not inst_files:
        print("No files found. Check --instructed-dir.", flush=True)
        return

    embedder = EmbeddingModel(model_name=args.embedding_model, device=args.device)

    judge: Optional[LLMJudge] = None
    if not args.skip_llm:
        if not args.llm_model or not Path(args.llm_model).exists():
            raise FileNotFoundError(f"LLM judge model not found: {args.llm_model}")
        judge = LLMJudge(model_path=args.llm_model, gpu=args.judge_gpu)

    all_results: List[Dict[str, Any]] = []

    for inst_file in inst_files:
        data = load_ad_output(inst_file)
        title = data.get("movie", data.get("movie_title", inst_file.stem.replace("_ad_output", "")))
        method = data.get("method", "unknown")
        print(
            f"\n[movie] [{method}] {title} — {len(data.get('ad_entries', []))} ADs, {len(data.get('insertion_events', []))} insertions",
            flush=True,
        )

        baseline_path: Optional[Path] = None
        if baseline_dir:
            baseline_path = find_matching_baseline(baseline_dir, title)
            if baseline_path:
                print(f"  baseline={baseline_path.name}", flush=True)
            else:
                print("  baseline=NOT FOUND", flush=True)

        result = eval_one_movie(
            embedder=embedder,
            movie_title=title,
            instructed_path=inst_file,
            output_dir=output_dir,
            baseline_path=baseline_path,
            judge=judge,
            segment_judge=args.segment_judge,
        )
        if result:
            all_results.append(result)
            print(
                f"  done self_ICA={result.get('self_mean_ICA')} self_ISR={result.get('self_mean_ISR')} bl_SSS={result.get('bl_mean_SSS_vs_baseline')}",
                flush=True,
            )

    if not all_results:
        print("No valid results were produced.", flush=True)
        return

    has_baseline = any(r.get("bl_mean_SSS_vs_baseline") is not None for r in all_results)
    summary_path = write_summary_csv(all_results, output_dir, has_baseline)
    readable_csv_path = write_readable_csv(all_results, output_dir, has_baseline)

    print(f"\n{'=' * 60}", flush=True)
    print("RESULTS:", flush=True)
    print(f"  Per-movie: {output_dir}/", flush=True)
    print(f"  Readable:  {readable_csv_path}", flush=True)
    print(f"  Summary:   {summary_path}", flush=True)
    print(f"{'=' * 60}", flush=True)


if __name__ == "__main__":
    main()
