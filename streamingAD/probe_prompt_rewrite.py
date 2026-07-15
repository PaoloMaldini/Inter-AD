#!/usr/bin/env python3
"""
probe_prompt_rewrite.py

Small, isolated A/B probe for interactive AD prompt design.

Goal:
1. Re-run a few short clips from existing low-ICA / low-ISR cases.
2. Compare current "Also: ..." injection against a Qwen-rewritten prompt.
3. Test whether sanitizing structured context reduces prompt leakage.

This script DOES NOT modify the main experiment pipeline.
It only reads existing eval rows / movie assets and writes comparison results.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".hf_cache"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(PROJECT_ROOT / ".hf_cache" / "hub"))
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(PROJECT_ROOT / ".hf_cache" / "sentence_transformers"))
os.environ.setdefault("PIP_CACHE_DIR", str(PROJECT_ROOT / ".pip_cache"))
os.environ.setdefault("TORCH_HOME", str(PROJECT_ROOT / ".torch_cache"))

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from ad_engine import ADEngine, build_ad_engine
from context_builder import (
    AVAILABLE_MODULES,
    AVAILABLE_PLOT_SUBMODULES,
    TASK_PROMPT_TEMPLATE,
    build_prompt_context,
    build_task_prompt,
)
from interactive_experiment import AD_CLIPS_DIR, FACE_JSON_ROOT, FINAL_BY_MOVIE_DIR, _parse_clip_stem, _resolve_dir
from run_eval import (
    EmbeddingModel,
    LLMJudge,
    compute_ica,
    compute_ifr_judge,
    compute_isr_judge,
    compute_user_alignment,
)
from segment_db import extract_face_data, load_segment_db, to_float

DEFAULT_LLM_MODEL = "/mnt/disk5new/gcc/models/Qwen2.5-7B-Instruct"
DEFAULT_CASES_CSV = PROJECT_ROOT / "eval_results" / "all_movies_unified.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "compare" / "prompt_rewrite_probe"

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


@dataclass
class ProbeCase:
    movie_title: str
    segment_idx: int
    category_name: str
    instruction_new: str
    instruction_before: str
    instruction_after: str
    original_text_after: str
    original_text_before: str
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
    if not rows:
        raise ValueError(f"No rows found in {csv_path}")

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
                original_text_before=row.get("text_before", ""),
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
    """Remove structured section headers that are likely to leak into output."""
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


def load_case_assets(case: ProbeCase) -> Tuple[Path, str, Optional[List[Path]], Optional[List[str]], str]:
    seg_db = load_segment_db(case.movie_title, final_by_movie_dir=FINAL_BY_MOVIE_DIR)
    seg = seg_db.segments[case.segment_idx]
    ad_id = str(seg.get("ad_id", "")).strip()
    clip_stem = _parse_clip_stem(ad_id=ad_id, clip_index=seg.get("clip_index"), ad_order=case.segment_idx + 1)

    clips_dir = _resolve_dir(AD_CLIPS_DIR / case.movie_title, case.movie_title)
    face_dir = _resolve_dir(FACE_JSON_ROOT / case.movie_title, case.movie_title)
    if clips_dir is None:
        raise FileNotFoundError(f"Clips directory not found for {case.movie_title}")

    clip_path = clips_dir / f"{clip_stem}.mp4"
    if not clip_path.is_file():
        raise FileNotFoundError(f"Clip not found: {clip_path}")

    face_matches: List[Dict[str, Any]] = []
    face_avatars: Optional[List[Path]] = None
    character_names: Optional[List[str]] = None
    if face_dir:
        face_json = face_dir / f"{clip_stem}.json"
        face_matches, _ = extract_face_data(face_json, max_face_records=4)
        if face_matches:
            character_names = [m.get("role_name", "") for m in face_matches if m.get("role_name")]
            face_avatars = [Path(m.get("avatar_path", "")) for m in face_matches if m.get("avatar_path")]

    context_text, _ = build_prompt_context(
        segment=seg,
        modules=AVAILABLE_MODULES,
        plot_submodules=AVAILABLE_PLOT_SUBMODULES,
        face_matches=face_matches,
        max_description_lines=5,
        max_dialog_lines=8,
    )
    ref_ad = str(seg.get("cmdqa", {}).get("text", "")).strip()
    return clip_path, context_text, face_avatars, character_names, ref_ad


def run_variant(
    engine: ADEngine,
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
        face_avatars=face_avatars,
        character_names=character_names,
    )

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
    for attr in ("model", "tokenizer", "chat"):
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
    parser = argparse.ArgumentParser(description="Probe prompt rewrite strategies on a few short AD clips.")
    parser.add_argument("--cases-csv", type=str, default=str(DEFAULT_CASES_CSV),
                        help="Existing unified eval CSV used to choose cases")
    parser.add_argument("--max-cases", type=int, default=3,
                        help="How many low-ICA cases to re-run")
    parser.add_argument("--ad-gpu", type=int, default=0,
                        help="GPU id for the AD generator model")
    parser.add_argument("--llm-gpu", type=int, default=1,
                        help="GPU id for the Qwen rewrite/judge model")
    parser.add_argument("--llm-model", type=str, default=DEFAULT_LLM_MODEL,
                        help="Qwen model used for rewrite and judging")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR),
                        help="Directory to write probe outputs")
    parser.add_argument("--skip-judge", action="store_true",
                        help="Skip Qwen judging; only compute ICA and leak flags")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"[probe] AD model GPU: {args.ad_gpu}")
    print(f"[probe] Qwen model GPU: {args.llm_gpu}")

    cases = load_probe_cases(Path(args.cases_csv), max_cases=args.max_cases)
    print(f"[probe] Loaded {len(cases)} cases from {args.cases_csv}")

    # Phase 1: rewrite prompts with Qwen
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

    # Phase 2: load embedder
    print("[probe] Phase 2/4: loading embedder")
    embedder = EmbeddingModel(
        model_name=str(PROJECT_ROOT / ".hf_cache" / "sentence_transformers" / "all-MiniLM-L6-v2"),
        device=f"cuda:{args.llm_gpu}" if torch.cuda.is_available() else "cpu",
    )

    # Phase 3: generate AD variants
    print("[probe] Phase 3/4: loading AD engine and generating variants")
    engine = build_ad_engine(gpu_id=args.ad_gpu)

    all_results: List[Dict[str, Any]] = []

    for case, rewritten_suffix in zip(cases, rewritten_suffixes):
        print(f"\n[case] {case.movie_title} seg={case.segment_idx} cat={case.category_name}")
        clip_path, full_context, face_avatars, character_names, ref_ad = load_case_assets(case)
        sanitized_context = sanitize_context_text(full_context)

        variants = [
            (
                "baseline_current",
                "full",
                full_context,
                build_task_prompt(case.instruction_after),
            ),
            (
                "baseline_sanitized_context",
                "sanitized",
                sanitized_context,
                build_task_prompt(case.instruction_after),
            ),
            (
                "rewrite_current_context",
                "full",
                full_context,
                build_full_prompt_from_suffix(rewritten_suffix),
            ),
            (
                "rewrite_sanitized_context",
                "sanitized",
                sanitized_context,
                build_full_prompt_from_suffix(rewritten_suffix),
            ),
        ]

        case_result: Dict[str, Any] = {
            "movie_title": case.movie_title,
            "segment_idx": case.segment_idx,
            "category_name": case.category_name,
            "instruction_new": case.instruction_new,
            "instruction_before": case.instruction_before,
            "instruction_after": case.instruction_after,
            "rewritten_suffix": rewritten_suffix,
            "ref_ad": ref_ad,
            "original_metrics": {
                "ICA": case.original_ica,
                "ISR": case.original_isr,
                "IFR": case.original_ifr,
                "User_Alignment": case.original_user_alignment,
            },
            "original_text_before": case.original_text_before,
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
                f"leak={result.prompt_leak} "
                f"text={result.generated_text[:80]}"
            )

        all_results.append(case_result)

    release_model(engine)

    if not args.skip_judge:
        print("\n[probe] Phase 4/4: loading Qwen judge and scoring generated variants")
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

    # Write outputs
    json_path = output_dir / f"probe_prompt_rewrite_{timestamp}.json"
    csv_path = output_dir / f"probe_prompt_rewrite_{timestamp}.csv"

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
