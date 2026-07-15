#!/usr/bin/env python3
"""
eval_api.py — LLM/VLM-as-Judge metrics using closed-source API (OpenAI-compatible).

Metrics:
  4. SegEval             — Paragraph-level coherence & narrative flow
  5. Streaming Coherence — Logic hallucination detection across long sequences
  6. CSR (Caption Success Rate) — Instruction + vision-grounded accuracy

Supports:
  - OpenAI API (GPT-4o, GPT-4o-mini)
  - Any OpenAI-compatible endpoint (vLLM, LiteLLM proxy, Azure, etc.)

Usage:
    # Set API key and run
    export OPENAI_API_KEY="sk-..."
    python streamingAD/eval_api.py \
        --run-dir batch_ad_output/focusedad_run_20260531_021518 \
        --model gpt-4o \
        --metrics all

    # Custom endpoint (e.g., Azure, local proxy)
    python streamingAD/eval_api.py \
        --run-dir batch_ad_output/focusedad_run_20260531_021518 \
        --model gpt-4o \
        --base-url https://your-proxy.com/v1 \
        --api-key sk-xxx

    # CSR with vision (requires frames extracted alongside AD)
    python streamingAD/eval_api.py --run-dir ... --metrics csr --frames-dir /path/to/frames
"""

from __future__ import annotations

import argparse
import base64
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


# ── API Client ────────────────────────────────────────────────

class APIJudge:
    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_retries: int = 3,
    ):
        try:
            from openai import OpenAI
        except ImportError:
            print("Please install openai: pip install openai")
            sys.exit(1)

        self.model = model
        self.client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY", "sk-placeholder"),
            base_url=base_url or os.environ.get("OPENAI_BASE_URL"),
        )
        self.max_retries = max_retries
        print(f"[API Judge] Model: {model}, base_url: {base_url or 'default'}")

    def chat(self, messages: List[Dict[str, Any]], max_tokens: int = 512, temperature: float = 0.0) -> str:
        for attempt in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                return resp.choices[0].message.content.strip()
            except Exception as e:
                print(f"  API attempt {attempt + 1}/{self.max_retries} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    return f"[API ERROR] {e}"
        return "[API ERROR]"

    def score_1_to_5(self, messages: List[Dict[str, Any]]) -> Tuple[int, str]:
        raw = self.chat(messages, max_tokens=256, temperature=0.0)
        score = 3
        rationale = raw
        import re
        for pattern in [r'\bscore\s*[:=]?\s*(\d)', r'\brating\s*[:=]?\s*(\d)', r'\b(\d)\s*/\s*5']:
            m = re.search(pattern, raw, re.IGNORECASE)
            if m:
                score = int(m.group(1))
                break
        return min(max(score, 1), 5), rationale


# ── Utility: build paragraph from sorted ADs ──────────────────

def _build_paragraph(entries: List[Dict[str, Any]]) -> str:
    sorted_entries = sorted(entries, key=lambda e: e.get("gap_start_sec", 0))
    lines = []
    for i, e in enumerate(sorted_entries):
        ts = e.get("gap_start_sec", 0)
        m, s = divmod(int(ts), 60)
        h, m = divmod(m, 60)
        stamp = f"[{h:02d}:{m:02d}:{s:02d}]"
        lines.append(f"{stamp} {e.get('ad_text', '')}")
    return "\n".join(lines)


def _encode_image(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ── Metric 4: SegEval ────────────────────────────────────────

def eval_segeval(
    entries: List[Dict[str, Any]],
    judge: APIJudge,
) -> Dict[str, Any]:
    """
    Evaluate paragraph-level coherence: take all ADs from one movie as a continuous
    narrative and score its flow, consistency, and AD quality as a whole.
    """
    paragraph = _build_paragraph(entries)
    if len(paragraph) < 50:
        return {"metric": "SegEval", "score": 0, "num_ads": len(entries), "error": "Too short"}

    system_prompt = """You are an expert evaluator of Audio Description (AD) for films.
AD is a narration track that describes visual elements for blind audiences.
It should be: concise, present-tense, visually grounded, non-redundant, and smoothly flowing."""

    user_prompt = f"""Evaluate the following continuous Audio Description script (multiple AD sentences 
stitched together as a paragraph). Rate its overall quality on a 1-5 scale:

  5 = Excellent: Smooth flow, varied descriptions, no redundancy, vivid visual language
  4 = Good: Minor issues in one area (flow, redundancy, or description quality)
  3 = Acceptable: Some repetition or awkward transitions, but serviceable
  2 = Poor: Noticeable redundancy, awkward phrasing, or factual contradictions
  1 = Very poor: Incoherent, highly redundant, or clearly hallucinated

AD PARAGRAPH:
{paragraph[:6000]}

Output format:
Score: X
Brief: (2-3 sentences explaining the score)
"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    score, rationale = judge.score_1_to_5(messages)
    return {
        "metric": "SegEval",
        "score": score,
        "num_ads": len(entries),
        "paragraph_length_chars": len(paragraph),
        "rationale": rationale,
    }


# ── Metric 5: Streaming Coherence ─────────────────────────────

def eval_streaming_coherence(
    entries: List[Dict[str, Any]],
    judge: APIJudge,
) -> Dict[str, Any]:
    """
    Detect action hallucinations and logical contradictions across sequential ADs.
    E.g., "He sits down." followed later by "He sits down again." without standing up.
    """
    sorted_entries = sorted(entries, key=lambda e: e.get("gap_start_sec", 0))
    if len(sorted_entries) < 3:
        return {"metric": "Streaming Coherence", "score": 5, "num_ads": len(sorted_entries), "error": "Too few"}

    # Build labeled sequence
    lines = []
    for i, e in enumerate(sorted_entries):
        ts = e.get("gap_start_sec", 0)
        m, s = divmod(int(ts), 60)
        h, m = divmod(m, 60)
        lines.append(f"AD#{i+1} [{h:02d}:{m:02d}:{s:02d}]: {e.get('ad_text', '')}")
    sequence = "\n".join(lines)

    system_prompt = """You are an expert at detecting logical contradictions and action hallucinations 
in Audio Description sequences. AD is spoken narration for blind audiences describing 
visual actions in films. Each AD sentence describes what happens during a short gap 
between dialogues."""

    user_prompt = f"""Review the following sequence of Audio Description (AD) sentences. 
These describe different gaps through a film, in chronological order.

Identify any LOGICAL CONTRADICTIONS or ACTION HALLUCINATIONS:
  - Contradiction: AD#N says X happened, but AD#M implies X didn't happen or conflicts
    (e.g., "He enters the room" then later "He enters the room" with no exit in between)
  - Hallucination: AD describes something impossible or clearly made-up
    (e.g., "The car flies" in a non-fantasy film, or describing objects not present)
  - Redundant re-description: Same action described identically multiple times

Rate overall coherence 1-5:
  5 = Perfect: No contradictions, no hallucinations, no redundancy
  4 = Minor: One small issue (e.g., slight redundancy)
  3 = Moderate: 2-3 minor issues or 1 contradiction
  2 = Significant: Multiple contradictions or 1 clear hallucination
  1 = Severe: Major contradictions throughout, clearly unreliable

AD SEQUENCE:
{sequence[:8000]}

Output format:
Score: X
Issues found: (list each issue with AD# references, or "None")
"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    score, rationale = judge.score_1_to_5(messages)
    return {
        "metric": "Streaming Coherence",
        "score": score,
        "num_ads": len(sorted_entries),
        "rationale": rationale,
    }


# ── Metric 6: CSR (Caption Success Rate) ─────────────────────

def eval_csr(
    entries: List[Dict[str, Any]],
    judge: APIJudge,
    frames_dir: Optional[Path] = None,
    instruction: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Caption Success Rate: evaluate whether generated AD both follows instructions
    AND accurately describes what's in the video.

    Without frames: text-only evaluation (instruction following + plausibility).
    With frames: VLM evaluation (vision-grounded accuracy + instruction following).
    """
    DEFAULT_INSTRUCTION = (
        "Describe what is happening in this clip concisely. "
        "Focus on visible actions, movements, and expressions."
    )
    instr = (instruction or DEFAULT_INSTRUCTION).strip()

    scores = []
    per_sample = []

    use_vision = frames_dir is not None and frames_dir.is_dir()

    for entry in entries:
        ad_text = entry.get("ad_text", "")
        if not ad_text:
            continue
        gap_id = entry.get("gap_id", "")

        if use_vision:
            frame_path = frames_dir / f"gap_{gap_id:04d}.png"
            if not frame_path.exists():
                frame_path = frames_dir / f"gap_{gap_id}.png"

            if frame_path.exists():
                b64 = _encode_image(frame_path)
                messages = [
                    {"role": "system", "content": (
                        "You are a VLM judge evaluating Audio Description quality. "
                        "Check if the AD text accurately describes what is visible in the frame "
                        "AND follows the given instruction."
                    )},
                    {"role": "user", "content": [
                        {"type": "text", "text": (
                            f"INSTRUCTION: {instr}\n\n"
                            f"GENERATED AD: {ad_text}\n\n"
                            f"Rate 1-5 whether the AD both follows the instruction AND "
                            f"accurately describes what you see in the image.\n"
                            f"5=Perfect, 1=Completely wrong. Output: Score: X"
                        )},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    ]},
                ]
            else:
                use_vision = False

        if not use_vision:
            cb = str(entry.get("context_before", ""))[:300]
            ca = str(entry.get("context_after", ""))[:300]
            messages = [
                {"role": "system", "content": (
                    "You evaluate Audio Description quality. Check instruction compliance "
                    "and whether the description is plausible given dialogue context."
                )},
                {"role": "user", "content": (
                    f"INSTRUCTION: {instr}\n\n"
                    f"DIALOGUE BEFORE: {cb}\n"
                    f"GENERATED AD: {ad_text}\n"
                    f"DIALOGUE AFTER: {ca}\n\n"
                    f"Rate 1-5: does the AD follow the instruction AND seem plausible?\n"
                    f"5=Perfect, 1=Complete failure. Output: Score: X"
                )},
            ]

        score, rationale = judge.score_1_to_5(messages)
        scores.append(float(score))
        per_sample.append({
            "gap_id": gap_id,
            "ad_text": ad_text,
            "score": score,
            "rationale": rationale,
            "vision_used": use_vision,
        })

    avg = round(sum(scores) / len(scores), 3) if scores else 0
    success_rate = round(sum(1 for s in scores if s >= 3) / len(scores), 3) if scores else 0
    return {
        "metric": "CSR",
        "score": avg,
        "success_rate": success_rate,
        "num_samples": len(scores),
        "vision_used": use_vision,
        "per_sample": per_sample,
    }


# ── Batch runner ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="API-based LLM/VLM evaluation for AD")
    parser.add_argument("--run-dir", type=str, default="",
                        help="Path to run directory (default: auto-detect latest)")
    parser.add_argument("--model", type=str, default="gpt-4o",
                        help="API model name (gpt-4o, gpt-4o-mini, claude-3-5-sonnet, etc.)")
    parser.add_argument("--base-url", type=str, default="",
                        help="Custom API base URL (OpenAI-compatible)")
    parser.add_argument("--api-key", type=str, default="",
                        help="API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--metrics", type=str, default="all",
                        help="Comma-separated: segeval,coherence,csr,all")
    parser.add_argument("--instruction", type=str, default="",
                        help="Custom instruction for CSR metric")
    parser.add_argument("--frames-dir", type=str, default="",
                        help="Directory with extracted frames for VLM-based CSR")
    parser.add_argument("--limit", type=int, default=0,
                        help="Max movies to evaluate (0 = all)")
    args = parser.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir.strip() else _find_latest_run_dir()
    if not run_dir or not run_dir.exists():
        print(f"Run dir not found: {run_dir}")
        sys.exit(1)

    metric_list = ["segeval", "coherence", "csr"] if args.metrics == "all" else args.metrics.split(",")
    frames_dir = Path(args.frames_dir) if args.frames_dir.strip() else None

    print(f"Run dir:    {run_dir}")
    print(f"Model:      {args.model}")
    print(f"Base URL:   {args.base_url or 'default'}")
    print(f"Metrics:    {metric_list}")
    print(f"Frames dir: {frames_dir or 'N/A (text-only CSR)'}\n")

    judge = APIJudge(
        model=args.model,
        api_key=args.api_key or None,
        base_url=args.base_url or None,
    )

    ad_files = sorted(run_dir.glob("*_ad_output.json"))
    if args.limit > 0:
        ad_files = ad_files[:args.limit]
    if not ad_files:
        print("No _ad_output.json files found.")
        sys.exit(1)
    print(f"Found {len(ad_files)} movie(s)\n")

    all_results: Dict[str, List[Dict[str, Any]]] = {m: [] for m in metric_list}

    for idx, ad_file in enumerate(ad_files):
        movie = ad_file.stem.replace("_ad_output", "")
        with open(ad_file) as f:
            data = json.load(f)
        entries = data.get("ad_entries", [])

        print(f"[{idx + 1}/{len(ad_files)}] {movie} ({len(entries)} ADs)")

        if "segeval" in metric_list:
            r = eval_segeval(entries, judge)
            all_results["segeval"].append({**r, "movie": movie})
            print(f"  SegEval: {r['score']}/5")

        if "coherence" in metric_list:
            r = eval_streaming_coherence(entries, judge)
            all_results["coherence"].append({**r, "movie": movie})
            print(f"  Streaming Coherence: {r['score']}/5")

        if "csr" in metric_list:
            r = eval_csr(entries, judge, frames_dir=frames_dir, instruction=args.instruction or None)
            all_results["csr"].append({**r, "movie": movie})
            print(f"  CSR: avg={r['score']}, success_rate={r['success_rate']}")

    # ── Aggregate & save ──
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    for metric_name, movie_results in all_results.items():
        if not movie_results:
            continue

        json_path = run_dir / f"eval_api_{metric_name}_{ts}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(movie_results, f, ensure_ascii=False, indent=2)

        csv_path = run_dir / f"eval_api_{metric_name}_{ts}.csv"
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            import csv
            writer = csv.writer(f)
            if metric_name in ("segeval", "coherence"):
                writer.writerow(["movie", "score", "num_ads"])
                for mr in movie_results:
                    writer.writerow([mr["movie"], mr["score"], mr.get("num_ads", 0)])
            elif metric_name == "csr":
                writer.writerow(["movie", "avg_score", "success_rate", "num_samples", "vision_used"])
                for mr in movie_results:
                    writer.writerow([mr["movie"], mr["score"], mr["success_rate"], mr["num_samples"], mr.get("vision_used", False)])

        print(f"\nSaved: {json_path}")
        print(f"Saved: {csv_path}")

    print(f"\nDone. Output: {run_dir}")


if __name__ == "__main__":
    main()
