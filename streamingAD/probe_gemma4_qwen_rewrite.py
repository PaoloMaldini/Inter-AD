#!/usr/bin/env python3
"""
probe_gemma4_qwen_rewrite.py

Isolated probe:
1. pick a few low-ICA / low-ISR interactive AD cases
2. rewrite user instruction suffix with Qwen
3. generate AD with Gemma4
4. compare raw suffix vs rewritten suffix, with/without sanitized context

This does not modify the main pipeline.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
WORKPLACE_DIR = SCRIPT_DIR / "workplace"

os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".hf_cache"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(PROJECT_ROOT / ".hf_cache" / "hub"))
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(PROJECT_ROOT / ".hf_cache" / "sentence_transformers"))
os.environ.setdefault("PIP_CACHE_DIR", str(PROJECT_ROOT / ".pip_cache"))
os.environ.setdefault("TORCH_HOME", str(PROJECT_ROOT / ".torch_cache"))

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(WORKPLACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKPLACE_DIR))

from context_builder import (
    AVAILABLE_MODULES,
    AVAILABLE_PLOT_SUBMODULES,
    TASK_PROMPT_TEMPLATE,
    build_prompt_context,
    build_task_prompt,
)
from segment_db import extract_face_data, load_segment_db
from workplace.pipeline_enhanced_gemma4 import (
    Gemma4ADEngine,
    build_gemma4_engine,
    postprocess_ad,
)

FINAL_BY_MOVIE_DIR = Path("/mnt/disk1new/ylz/newAD/Step04_RunTest/step04_final_by_movie_new")
AD_CLIPS_DIR = Path("/mnt/disk1new/ylz/newAD/Step04_RunTest/ad_clips_final")
FACE_JSON_ROOT = Path("/mnt/disk1new/ylz/newAD/Step04_RunTest/step04_03_face_align/json")

DEFAULT_CASES_CSV = PROJECT_ROOT / "eval_results" / "all_movies_unified.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "compare" / "gemma4_qwen_rewrite_probe"
DEFAULT_GEMMA4_MODEL = "/mnt/disk1new/ylz/newAD/models/gemma-4-26b-a4b-it"
DEFAULT_LLM_MODEL = "/mnt/disk5new/gcc/models/Qwen2.5-7B-Instruct"

PROMPT_LEAK_PATTERNS = (
    "Plot Database",
    "character mapping",
    "Target AD Clip",
    "[CMDQA AD text]",
    "[Nearby screenplay descriptions]",
    "[Scene indices]",
    "[Record types]",
    "[Related shot structured info]",
)


class EmbeddingModel:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = "cuda"):
        self._backend = "bow"
        self._tfidf = None
        self._vocab: Optional[Any] = None
        try:
            from sentence_transformers import SentenceTransformer
            if device.startswith("cuda") and not torch.cuda.is_available():
                device = "cpu"
            print(f"[embedding] Loading {model_name} on {device} ...")
            self.model = SentenceTransformer(model_name, device=device)
            self._backend = "st"
            print("[embedding] Model loaded.")
        except Exception as e:
            print(f"[embedding] SentenceTransformer unavailable ({type(e).__name__}: {e}).")
            try:
                print("[embedding] Falling back to local TF-IDF embedding.")
                self._init_tfidf()
                self._backend = "tfidf"
            except Exception as e2:
                print(f"[embedding] TF-IDF unavailable ({type(e2).__name__}: {e2}).")
                print("[embedding] Falling back to simple bag-of-words cosine.")

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
        if self._backend == "tfidf":
            self._ensure_tfidf_fit(texts)
            vecs = self._tfidf.transform(texts).toarray().astype(np.float32)
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-8)
            return vecs / norms
        return self._encode_bow(texts)

    def _encode_bow(self, texts: List[str]) -> np.ndarray:
        tokenized = []
        vocab: Dict[str, int] = {}
        for text in texts:
            tokens = re.findall(r"[a-zA-Z0-9\u4e00-\u9fff]+", (text or "").lower())
            tokenized.append(tokens)
            for tok in tokens:
                if tok not in vocab:
                    vocab[tok] = len(vocab)
        if not vocab:
            return np.zeros((len(texts), 1), dtype=np.float32)
        vecs = np.zeros((len(texts), len(vocab)), dtype=np.float32)
        for i, tokens in enumerate(tokenized):
            for tok in tokens:
                vecs[i, vocab[tok]] += 1.0
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        return vecs / norms

    def cosine_sim(self, text_a: str, text_b: str) -> float:
        emb = self.encode([text_a, text_b])
        return float(np.dot(emb[0], emb[1]))


class LLMJudge:
    def __init__(self, model_path: str, gpu: int = 0):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.device = f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"
        print(f"[llm_judge] Loading {model_path} on {self.device} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
        ).to(self.device).eval()
        print("[llm_judge] Model loaded.")

    def generate(self, prompt: str, max_new_tokens: int = 256, temperature: float = 0.0) -> str:
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
        return self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

    def score_1_to_5(self, prompt: str) -> Tuple[int, str]:
        raw = self.generate(prompt, max_new_tokens=128)
        score = 3
        for pattern in [
            r"\bscore\s*[:=]?\s*(\d)",
            r"\brating\s*[:=]?\s*(\d)",
            r"\b(\d)\s*/\s*5",
            r"\b(\d)\s*out\s*of\s*5",
            r"[\[:]\s*(\d)\s*[\]]",
        ]:
            m = re.search(pattern, raw, re.IGNORECASE)
            if m:
                score = int(m.group(1))
                break
        return min(max(score, 1), 5), raw

    def score_binary(self, prompt: str) -> Tuple[int, str]:
        raw = self.generate(prompt, max_new_tokens=128)
        for pattern in [r"\bscore\s*[:=]?\s*(\d)", r"\b(\d)\b"]:
            m = re.search(pattern, raw, re.IGNORECASE)
            if m:
                val = int(m.group(1))
                if val in (1, 2):
                    return val, raw
        if any(w in raw.lower() for w in ["pass", "follows", "correct", "yes"]):
            return 2, raw
        return 1, raw


def compute_ica(embedder: EmbeddingModel, instruction: str, after_text: str) -> float:
    if not instruction.strip() or not after_text.strip():
        return 0.0
    return embedder.cosine_sim(instruction, after_text)


def compute_isr_judge(judge: LLMJudge, instruction: str, after_text: str) -> Tuple[int, str]:
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
    prompt = f"""You are evaluating how well a generated Audio Description (AD) follows an instruction.
Break the instruction into individual requirements and check each one.

INSTRUCTION: {instruction}

GENERATED AD: {after_text}

List each requirement as [MET] or [NOT MET], then give a ratio as Score: X/Y.

Output format:
Requirement 1: [MET/NOT MET] (brief description)
Requirement 2: [MET/NOT MET] (brief description)
Score: X/Y
"""
    raw = judge.generate(prompt, max_new_tokens=256)
    ratio = 0.0
    m = re.search(r"score\s*[:=]?\s*(\d+)\s*/\s*(\d+)", raw, re.IGNORECASE)
    if m:
        num, den = int(m.group(1)), int(m.group(2))
        ratio = num / max(den, 1)
    return round(ratio, 4), raw


def compute_user_alignment(judge: LLMJudge, instruction: str, after_text: str) -> Tuple[int, str]:
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


@dataclass
class ProbeCase:
    movie_title: str
    segment_idx: int
    category_name: str
    instruction_new: str
    instruction_before: str
    instruction_after: str
    original_text_after: str
    original_ica: Optional[float]
    original_isr: Optional[float]
    original_ifr: Optional[float]
    original_user_alignment: Optional[float]


@dataclass
class VariantResult:
    name: str
    context_mode: str
    task_prompt: str
    generated_text: str
    latency_sec: float
    prompt_leak: bool
    ica_new_instruction: float
    isr_new_instruction: Optional[float]
    ifr_new_instruction: Optional[float]
    user_alignment_new_instruction: Optional[float]


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value in ("", None):
            return None
        return float(value)
    except Exception:
        return None


def load_probe_cases(csv_path: Path, max_cases: int) -> List[ProbeCase]:
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    rows.sort(key=lambda r: (_safe_float(r.get("ICA")) if _safe_float(r.get("ICA")) is not None else 999.0))
    cases: List[ProbeCase] = []
    seen: set[Tuple[str, int, str]] = set()
    for row in rows:
        key = (row["movie_title"], int(row["segment_idx"]), row["instruction_new"])
        if key in seen:
            continue
        seen.add(key)
        cases.append(
            ProbeCase(
                movie_title=row["movie_title"],
                segment_idx=int(row["segment_idx"]),
                category_name=row["category_name"],
                instruction_new=row["instruction_new"],
                instruction_before=row.get("instruction_before", ""),
                instruction_after=row.get("instruction_after", ""),
                original_text_after=row.get("text_after", ""),
                original_ica=_safe_float(row.get("ICA")),
                original_isr=_safe_float(row.get("ISR")),
                original_ifr=_safe_float(row.get("IFR")),
                original_user_alignment=_safe_float(row.get("User_Alignment")),
            )
        )
        if len(cases) >= max_cases:
            break
    return cases


def sanitize_context_text(context_text: str) -> str:
    cleaned: List[str] = []
    for raw in context_text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line in PROMPT_LEAK_PATTERNS:
            continue
        if line.startswith("Plot Database (The Subtext):"):
            continue
        if line.startswith("character mapping"):
            continue
        if line.startswith("Target AD Clip:"):
            continue
        if line.startswith("[") and line.endswith("]"):
            continue
        line = re.sub(r"^-+\s*", "", line).strip()
        if not line or line == "None":
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


def detect_prompt_leak(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    return any(pattern in stripped for pattern in PROMPT_LEAK_PATTERNS)


def normalize_suffix(text: str) -> str:
    text = " ".join(str(text).strip().split())
    text = re.sub(r'^[\'"]|[\'"]$', "", text)
    text = re.sub(r"^(Also:\s*)", "", text, flags=re.IGNORECASE)
    text = text.strip()
    text = text.rstrip(". ")
    return text


def build_full_prompt_from_suffix(suffix: str) -> str:
    suffix = normalize_suffix(suffix)
    if not suffix:
        return build_task_prompt("")
    return TASK_PROMPT_TEMPLATE.format(f"Also: {suffix}. ")


def rewrite_instruction_suffix(
    judge: LLMJudge,
    instruction_before: str,
    instruction_new: str,
    keep_words_under: int = 22,
) -> str:
    prompt = f"""You rewrite user control requests into a short Audio Description task suffix.

The base task already enforces:
- describe what is happening in the clip concisely
- focus on visible actions, movements, and expressions
- use character names if available
- do not quote dialogue

Persistent active preferences from earlier interactions:
{instruction_before or "None"}

New user request to prioritize now:
{instruction_new}

Write ONE short English suffix that can be appended after "Also: ".
Requirements:
- Keep it suitable for a single-sentence AD task.
- Prioritize the NEW request over older ones.
- Resolve conflicts by preserving concise visible-only AD behavior.
- Convert vague style requests into AD-suitable wording.
- Do not mention prompts, databases, subtitles, sections, or meta instructions.
- Keep it under {keep_words_under} words if possible.

Output only the suffix, one line, no bullets, no explanation.
"""
    raw = judge.generate(prompt, max_new_tokens=96, temperature=0.0)
    line = raw.splitlines()[0].strip() if raw.strip() else ""
    return normalize_suffix(line)


def _parse_clip_stem(ad_id: str, clip_index: Any, ad_order: int) -> str:
    ad_id = str(ad_id or "").strip()
    if ad_id:
        stem = Path(ad_id).stem
        if stem:
            m = re.search(r"(clip\d{4})_+(ad\d{4})", stem)
            if m:
                return f"{m.group(1)}_{m.group(2)}"
            return stem
    if clip_index is not None:
        try:
            idx = int(float(clip_index))
            return f"clip{idx:04d}_ad{ad_order:04d}"
        except Exception:
            pass
    return f"clip{ad_order:04d}_ad{ad_order:04d}"


def _resolve_dir(base_dir: Path, movie_title: str) -> Optional[Path]:
    if base_dir.is_dir():
        if (base_dir / movie_title).is_dir():
            return base_dir / movie_title
        for p in base_dir.iterdir():
            if p.is_dir() and p.name.replace("_", " ").lower() == movie_title.lower():
                return p
    return None


def load_case_assets(case: ProbeCase) -> Tuple[Path, str, Optional[List[Path]], Optional[List[str]]]:
    seg_db = load_segment_db(case.movie_title, final_by_movie_dir=FINAL_BY_MOVIE_DIR)
    seg = seg_db.segments[case.segment_idx]
    clip_stem = _parse_clip_stem(
        ad_id=str(seg.get("ad_id", "")).strip(),
        clip_index=seg.get("clip_index"),
        ad_order=case.segment_idx + 1,
    )

    clips_dir = _resolve_dir(AD_CLIPS_DIR, case.movie_title)
    face_dir = _resolve_dir(FACE_JSON_ROOT, case.movie_title)
    if clips_dir is None:
        raise FileNotFoundError(f"Clips directory not found for {case.movie_title}")

    clip_path = clips_dir / f"{clip_stem}.mp4"
    if not clip_path.is_file():
        raise FileNotFoundError(f"Clip not found: {clip_path}")

    face_avatars: Optional[List[Path]] = None
    character_names: Optional[List[str]] = None
    face_matches: List[Dict[str, Any]] = []
    if face_dir:
        face_json = face_dir / f"{clip_stem}.json"
        face_matches, avatars = extract_face_data(face_json, max_face_records=4)
        if avatars:
            face_avatars = avatars
        if face_matches:
            character_names = [m.get("role_name", "") for m in face_matches if m.get("role_name")]

    context_text, _ = build_prompt_context(
        segment=seg,
        modules=AVAILABLE_MODULES,
        plot_submodules=AVAILABLE_PLOT_SUBMODULES,
        face_matches=face_matches,
        max_description_lines=5,
        max_dialog_lines=8,
    )
    return clip_path, context_text, face_avatars, character_names


def run_variant(
    engine: Gemma4ADEngine,
    embedder: EmbeddingModel,
    *,
    variant_name: str,
    context_mode: str,
    context_text: str,
    task_prompt: str,
    clip_path: Path,
    face_avatars: Optional[List[Path]],
    character_names: Optional[List[str]],
    instruction_new: str,
) -> VariantResult:
    generated_text, latency_sec, _ = engine.infer_one_segment(
        clip_path=clip_path,
        context_text=context_text,
        task_prompt=task_prompt,
        temperature=0.2,
        max_new_tokens=96,
        face_avatars=face_avatars,
        character_names=character_names,
        num_beams=3,
    )
    generated_text = postprocess_ad(generated_text, max_words=22) or generated_text.strip()
    ica = compute_ica(embedder, instruction_new, generated_text)
    return VariantResult(
        name=variant_name,
        context_mode=context_mode,
        task_prompt=task_prompt,
        generated_text=generated_text,
        latency_sec=round(latency_sec, 4),
        prompt_leak=detect_prompt_leak(generated_text),
        ica_new_instruction=round(ica, 4),
        isr_new_instruction=None,
        ifr_new_instruction=None,
        user_alignment_new_instruction=None,
    )


def release_model(obj: Any) -> None:
    if obj is None:
        return
    for attr in ("model", "tokenizer", "processor"):
        if hasattr(obj, attr):
            try:
                delattr(obj, attr)
            except Exception:
                pass
    del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description="Gemma4 + Qwen prompt rewrite probe.")
    parser.add_argument("--cases-csv", type=str, default=str(DEFAULT_CASES_CSV))
    parser.add_argument("--max-cases", type=int, default=2)
    parser.add_argument("--gemma-gpu", type=str, default="0")
    parser.add_argument("--llm-gpu", type=int, default=1)
    parser.add_argument("--gemma-model", type=str, default=DEFAULT_GEMMA4_MODEL)
    parser.add_argument("--llm-model", type=str, default=DEFAULT_LLM_MODEL)
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--skip-judge", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"[probe] Gemma CUDA_VISIBLE_DEVICES={args.gemma_gpu}")
    print(f"[probe] Qwen GPU={args.llm_gpu}")

    cases = load_probe_cases(Path(args.cases_csv), max_cases=args.max_cases)
    print(f"[probe] Loaded {len(cases)} cases from {args.cases_csv}")

    print("[probe] Phase 1/4: rewriting prompt suffixes with Qwen")
    rewrite_judge = LLMJudge(model_path=args.llm_model, gpu=args.llm_gpu)
    rewritten_suffixes: List[str] = []
    for i, case in enumerate(cases, start=1):
        suffix = rewrite_instruction_suffix(
            rewrite_judge,
            instruction_before=case.instruction_before,
            instruction_new=case.instruction_new,
        )
        rewritten_suffixes.append(suffix)
        print(f"  [{i}/{len(cases)}] {case.movie_title} seg={case.segment_idx} -> {suffix}")
    release_model(rewrite_judge)

    print("[probe] Phase 2/4: loading embedder")
    embedder = EmbeddingModel(
        model_name=str(PROJECT_ROOT / ".hf_cache" / "sentence_transformers" / "all-MiniLM-L6-v2"),
        device=f"cuda:{args.llm_gpu}" if torch.cuda.is_available() else "cpu",
    )

    print("[probe] Phase 3/4: loading Gemma4 and generating variants")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gemma_gpu
    engine = build_gemma4_engine(model_path=args.gemma_model)

    all_results: List[Dict[str, Any]] = []
    for case, rewritten_suffix in zip(cases, rewritten_suffixes):
        print(f"\n[case] {case.movie_title} seg={case.segment_idx} cat={case.category_name}")
        clip_path, full_context, face_avatars, character_names = load_case_assets(case)
        sanitized_context = sanitize_context_text(full_context)

        variants = [
            ("baseline_current", "full", full_context, build_task_prompt(case.instruction_after)),
            ("baseline_sanitized_context", "sanitized", sanitized_context, build_task_prompt(case.instruction_after)),
            ("rewrite_current_context", "full", full_context, build_full_prompt_from_suffix(rewritten_suffix)),
            ("rewrite_sanitized_context", "sanitized", sanitized_context, build_full_prompt_from_suffix(rewritten_suffix)),
        ]

        case_result: Dict[str, Any] = {
            "movie_title": case.movie_title,
            "segment_idx": case.segment_idx,
            "category_name": case.category_name,
            "instruction_new": case.instruction_new,
            "instruction_before": case.instruction_before,
            "instruction_after": case.instruction_after,
            "rewritten_suffix": rewritten_suffix,
            "original_metrics": {
                "ICA": case.original_ica,
                "ISR": case.original_isr,
                "IFR": case.original_ifr,
                "User_Alignment": case.original_user_alignment,
            },
            "original_text_after": case.original_text_after,
            "variants": [],
        }

        for name, context_mode, context_text, task_prompt in variants:
            result = run_variant(
                engine=engine,
                embedder=embedder,
                variant_name=name,
                context_mode=context_mode,
                context_text=context_text,
                task_prompt=task_prompt,
                clip_path=clip_path,
                face_avatars=face_avatars,
                character_names=character_names,
                instruction_new=case.instruction_new,
            )
            case_result["variants"].append(asdict(result))
            print(
                f"  - {name}: ICA={result.ica_new_instruction:.4f} "
                f"leak={result.prompt_leak} text={result.generated_text[:100]}"
            )

        all_results.append(case_result)

    release_model(engine)

    if not args.skip_judge:
        print("\n[probe] Phase 4/4: loading Qwen judge and scoring variants")
        judge = LLMJudge(model_path=args.llm_model, gpu=args.llm_gpu)
        for case in all_results:
            print(f"  [judge] {case['movie_title']} seg={case['segment_idx']}")
            for variant in case["variants"]:
                isr, _ = compute_isr_judge(judge, case["instruction_new"], variant["generated_text"])
                ifr, _ = compute_ifr_judge(judge, case["instruction_new"], variant["generated_text"])
                ua, _ = compute_user_alignment(judge, case["instruction_new"], variant["generated_text"])
                variant["isr_new_instruction"] = _safe_float(isr)
                variant["ifr_new_instruction"] = _safe_float(ifr)
                variant["user_alignment_new_instruction"] = _safe_float(ua)
                print(
                    f"    - {variant['name']}: ISR={variant['isr_new_instruction']} "
                    f"IFR={variant['ifr_new_instruction']} UA={variant['user_alignment_new_instruction']}"
                )
        release_model(judge)

    json_path = output_dir / f"probe_gemma4_qwen_rewrite_{timestamp}.json"
    csv_path = output_dir / f"probe_gemma4_qwen_rewrite_{timestamp}.csv"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump({"created_at": timestamp, "cases": all_results}, f, ensure_ascii=False, indent=2)

    csv_rows: List[Dict[str, Any]] = []
    for case in all_results:
        for variant in case["variants"]:
            csv_rows.append({
                "movie_title": case["movie_title"],
                "segment_idx": case["segment_idx"],
                "category_name": case["category_name"],
                "instruction_new": case["instruction_new"],
                "variant": variant["name"],
                "context_mode": variant["context_mode"],
                "original_ICA": case["original_metrics"]["ICA"],
                "original_ISR": case["original_metrics"]["ISR"],
                "original_IFR": case["original_metrics"]["IFR"],
                "ICA_new_instruction": variant["ica_new_instruction"],
                "ISR_new_instruction": variant["isr_new_instruction"],
                "IFR_new_instruction": variant["ifr_new_instruction"],
                "User_Alignment_new_instruction": variant["user_alignment_new_instruction"],
                "prompt_leak": variant["prompt_leak"],
                "latency_sec": variant["latency_sec"],
                "task_prompt": variant["task_prompt"],
                "generated_text": variant["generated_text"],
                "rewritten_suffix": case["rewritten_suffix"],
                "original_text_after": case["original_text_after"],
            })

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()) if csv_rows else [])
        if csv_rows:
            writer.writeheader()
            writer.writerows(csv_rows)

    print(f"\n[probe] JSON: {json_path}")
    print(f"[probe] CSV:  {csv_path}")


if __name__ == "__main__":
    main()
