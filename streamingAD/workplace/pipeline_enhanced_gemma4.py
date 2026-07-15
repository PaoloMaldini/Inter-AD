#!/usr/bin/env python3
"""
Enhanced Pipeline for CMD-AD benchmark -- Gemma4 edition.

Same as pipeline_cmdad_enhanced.py but replaces the Video-LLaMA generation
engine with Gemma4 (gemma-4-26b-a4b-it).

Improvements carried over:
  1. Improved prompt with word limit + Few-shot examples
  2. Multi-candidate generation (multiple temperatures)
  3. Heuristic-based candidate selection (no external API)
  4. Output post-processing (remove artifacts, enforce constraints)
  5. Streaming curation (deduplicate with recent AD history)

Usage:
    python streamingAD/workplace/pipeline_enhanced_gemma4.py
    python streamingAD/workplace/pipeline_enhanced_gemma4.py --num-candidates 5
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
import argparse
from collections import defaultdict
from difflib import SequenceMatcher
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from PIL import Image as PILImage

# -- Default Config --

DEFAULT_GPU            = "0,2,3"  # GPU 2,3 for Gemma4; GPU 0 for Qwen selector
DEFAULT_SPLIT          = "eval"
DEFAULT_MAX_CLIPS      = 0
DEFAULT_NUM_CANDIDATES = 3
DEFAULT_MAX_TOKENS     = 128
DEFAULT_BASE_TEMP      = 0.1
DEFAULT_TEMP_STEP      = 0.2
DEFAULT_SELECTOR_MODEL = "/mnt/disk1new/cxx/models/Qwen2.5-VL-7B-Instruct"
GEMMA4_MODEL_PATH      = "/mnt/disk1new/ylz/newAD/models/gemma-4-26b-a4b-it"

# -- Paths --

STREAMING_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT   = STREAMING_ROOT.parent.parent  # workplace -> streamingAD -> newAD

CMDAD_DIR      = PROJECT_ROOT / "datasets" / "cmdad"
ANNO_CSV       = CMDAD_DIR / "cmd_ad_anno_v1.csv"
CLIPS_DIR      = CMDAD_DIR / "clips"
CHARBANK_JSON  = CMDAD_DIR / "AutoAD-Zero" / "resources" / "charbanks" / "cmdad_charbank.json"
CLIPS_MANIFEST = CMDAD_DIR / "clip_manifest.json"

OUTPUT_DIR     = PROJECT_ROOT / "compare" / "cmdad_enhanced_gemma4"

# -- Same prompt as original pipeline (well-tuned for this task) --
TASK_PROMPT = (
    "Describe what is happening in this clip concisely. "
    "Focus on visible actions, movements, and expressions. "
    "If character names are mentioned in the context, use them "
    "(e.g. 'Don Vito Corleone walks...' not 'A man walks...'). "
    "Do not quote dialogue."
)


# ====================================================================
# Frame extraction
# ====================================================================

def extract_frames(clip_path: Path, num_frames: int = 8) -> List[PILImage.Image]:
    """Extract evenly-spaced frames from a video clip using ffmpeg."""
    frames = []
    try:
        r = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
             '-of', 'default=noprint_wrappers=1:nokey=1', str(clip_path)],
            capture_output=True, text=True)
        duration = float(r.stdout.strip())
        if duration <= 0:
            return []
    except (ValueError, subprocess.SubprocessError):
        return []

    for i in range(num_frames):
        t = duration * (i + 0.5) / num_frames
        r = subprocess.run(
            ['ffmpeg', '-ss', str(t), '-i', str(clip_path),
             '-frames:v', '1', '-q:v', '2', '-f', 'image2', '-'],
            capture_output=True)
        if r.returncode == 0 and len(r.stdout) > 100:
            try:
                img = PILImage.open(BytesIO(r.stdout)).convert('RGB')
                frames.append(img)
            except Exception:
                pass
    return frames


# ====================================================================
# Gemma4 AD Engine
# ====================================================================

class Gemma4ADEngine:
    """AD generation engine using Gemma4 multimodal model."""

    def __init__(self, model, processor, device: str = "cuda"):
        self.model = model
        self.processor = processor
        self.device = device

    def infer_one_segment(
        self,
        clip_path: Path,
        context_text: str,
        task_prompt: str,
        temperature: float = 0.2,
        max_new_tokens: int = 128,
        face_avatars: Optional[List[Path]] = None,
        character_names: Optional[List[str]] = None,
        num_beams: int = 1,
    ) -> Tuple[str, float, str]:
        """
        Generate AD for a single clip using Gemma4.
        Returns (ad_text, inference_time_sec, full_prompt).
        """
        torch.cuda.empty_cache()

        # -- Extract video frames --
        frames = extract_frames(clip_path, num_frames=8)
        if not frames:
            return "", 0.0, ""

        # -- Load face avatar images --
        avatar_images: List[PILImage.Image] = []
        if face_avatars:
            for avatar_path in face_avatars:
                try:
                    img = PILImage.open(str(avatar_path)).convert('RGB')
                    avatar_images.append(img)
                except Exception:
                    pass

        # -- Build multimodal content --
        content = []

        # Context text
        if context_text:
            content.append({"type": "text", "text": context_text + "\n"})

        # Face avatars with character name annotation
        if avatar_images and character_names:
            for img in avatar_images:
                content.append({"type": "image", "image": img})
            content.append({
                "type": "text",
                "text": f"[These faces belong to: {', '.join(character_names)}]\n"
            })

        # Video frames
        for img in frames:
            content.append({"type": "image", "image": img})

        # Task prompt
        content.append({"type": "text", "text": task_prompt})

        messages = [{"role": "user", "content": content}]
        full_prompt = context_text + "\n" + task_prompt if context_text else task_prompt

        # -- Prepare inputs --
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)

        all_images = avatar_images + frames
        inputs = self.processor(
            text=[text], images=all_images if all_images else None,
            return_tensors='pt', padding=True,
        ).to(self.model.device)

        # -- Generate --
        gen_kwargs = dict(max_new_tokens=max_new_tokens)
        if num_beams > 1:
            gen_kwargs.update(num_beams=num_beams, do_sample=False)
        else:
            if temperature > 0.01:
                gen_kwargs.update(do_sample=True, temperature=temperature)
            else:
                gen_kwargs.update(do_sample=False)

        start_time = time.monotonic()
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
        elapsed = time.monotonic() - start_time

        generated = outputs[0][inputs['input_ids'].shape[-1]:]
        answer = self.processor.decode(generated, skip_special_tokens=True).strip()

        return answer, elapsed, full_prompt

    def select_best_candidate(
        self,
        candidates: List[str],
        selector=None,
        clip_path: Path = None,
    ) -> str:
        """
        Select the best AD candidate.
        If selector (Qwen2.5-VL) is provided, use it for multimodal selection.
        Otherwise, use Gemma4 itself for text-only selection.
        """
        if not candidates:
            return ""
        if len(candidates) == 1:
            return candidates[0]

        import random
        shuffled = list(enumerate(candidates))
        random.shuffle(shuffled)

        # Build selection prompt
        selection_prompt = (
            "You are an expert at evaluating Audio Descriptions (AD) for movies.\n"
            "A good AD:\n"
            "- Describes specific, visible actions and movements\n"
            "- Uses character names when available (not 'a man', 'the woman')\n"
            "- Is one sentence, under 20 words\n"
            "- Captures the most important action in the scene\n\n"
            "A bad AD:\n"
            "- Is too vague ('someone is doing something')\n"
            "- Only describes static poses without actions\n"
            "- Misses the main action\n"
            "- Contains dialogue or internal thoughts\n\n"
            "Examples:\n\n"
            "Candidates:\n"
            "1. A person walks across the room.\n"
            "2. Don Corleone stands up from his desk and walks to the window.\n"
            "3. There is a man in a suit.\n"
            "Best: 2\n\n"
            "Candidates:\n"
            "1. Two people are talking.\n"
            "2. Michael kisses his father's hand.\n"
            "3. Someone looks at something.\n"
            "Best: 2\n\n"
            "Candidates:\n"
            "1. A woman sits down slowly.\n"
            "2. Something happens in a room.\n"
            "3. Kay opens the door and walks into the office.\n"
            "Best: 3\n\n"
            "Candidates:\n"
            "1. The camera pans across the city.\n"
            "2. People are walking around.\n"
            "3. Vito gestures to his men to leave the room.\n"
            "Best: 3\n\n"
            "Now evaluate:\nCandidates:\n"
        )

        for new_idx, (orig_idx, text) in enumerate(shuffled):
            selection_prompt += f"{new_idx + 1}. {text}\n"
        selection_prompt += "Best:"

        try:
            if selector is not None:
                # Use Qwen2.5-VL multimodal selector
                answer = selector.generate(
                    selection_prompt, clip_path=clip_path, max_new_tokens=5)
            else:
                # Use Gemma4 for text-only selection
                messages = [{"role": "user", "content": selection_prompt}]
                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)
                inputs = self.processor(
                    text=[text], images=None,
                    return_tensors='pt', padding=True,
                ).to(self.model.device)

                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs, max_new_tokens=5, do_sample=False)
                generated = outputs[0][inputs['input_ids'].shape[-1]:]
                answer = self.processor.decode(
                    generated, skip_special_tokens=True).strip()

            # Parse answer to get index
            nums = re.findall(r'\d+', answer)
            if nums:
                chosen = int(nums[0]) - 1
                if 0 <= chosen < len(shuffled):
                    return shuffled[chosen][1]
        except Exception:
            pass

        return candidates[0]


def build_gemma4_engine(model_path: str = GEMMA4_MODEL_PATH) -> Gemma4ADEngine:
    """Load Gemma4 model and return a Gemma4ADEngine instance."""
    from transformers import AutoProcessor, Gemma4ForConditionalGeneration

    print(f"  Loading Gemma4 from {model_path}...")
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = Gemma4ForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
        max_memory={1: "48GiB", 2: "48GiB"},  # cuda:1,2 = physical GPU 2,3; leave cuda:0 for Qwen
    ).eval()

    device = str(model.device)
    print(f"  Gemma4 loaded, primary device: {device}")
    return Gemma4ADEngine(model, processor, device)


# ====================================================================
# Post-processing
# ====================================================================

def postprocess_ad(text: str, max_words: int = 20) -> str:
    """Clean up raw model output into a valid AD sentence."""
    if not text:
        return ""

    text = text.replace("</s>", "").strip()

    # Remove common reasoning / meta artifacts
    artifact_prefixes = [
        "The script identifies", "Based on the visual", "Based on the movie",
        "Here is the audio", "The audio description", "Here is a description",
        "The description is:", "Audio description:", "AD:", "Caption:",
        "I will describe", "Let me describe", "The scene shows",
        "Looking at the video", "In this clip,", "In this scene,",
        "Based on the", "According to the",
        "Sure, here", "Okay, here", "Here is a sentence",
        "Here is the description", "Okay, I'm ready",
    ]
    for prefix in artifact_prefixes:
        if text.startswith(prefix):
            # Try to find the actual AD after a colon
            colon_pos = text.find(":")
            if colon_pos != -1 and colon_pos < len(text) - 5:
                after_colon = text[colon_pos + 1:].strip()
                if len(after_colon.split()) >= 3:
                    text = after_colon
                    continue
            # Try second sentence
            parts = text.split(".", 1)
            if len(parts) > 1 and len(parts[1].strip()) > 10:
                text = parts[1].strip()
            else:
                lines = [l.strip() for l in text.split("\n") if l.strip()]
                if lines:
                    text = lines[-1]

    # Remove markdown formatting
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'^\*\s*', '', text, flags=re.MULTILINE)

    # Remove quotes wrapping the whole sentence
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1]
    if text.startswith("'") and text.endswith("'"):
        text = text[1:-1]

    # Take only the first sentence
    sentences = re.split(r'[.!?]\s+', text)
    if sentences:
        text = sentences[0].strip()
        if not text.endswith(('.', '!', '?')):
            text += '.'

    # Enforce word limit
    words = text.split()
    if len(words) > max_words:
        text = " ".join(words[:max_words])
        if not text.endswith(('.', '!', '?')):
            text += '.'

    # Remove empty or too-short outputs
    if len(words) < 1:
        return ""

    # Reject clearly non-AD outputs (chatty/meta responses)
    text_lower_check = text.lower()
    reject_patterns = [
        "i'm ready", "i am ready", "generate the audio",
        "describe the video", "here's a sentence", "audio description for",
        "the video clip contains", "the video shows", "the clip shows",
        "this video shows", "this clip shows",
    ]
    for pat in reject_patterns:
        if pat in text_lower_check:
            if len(words) <= 10:
                return ""

    # Capitalize first letter
    if text:
        text = text[0].upper() + text[1:]

    return text.strip()


# ====================================================================
# Candidate Selection (heuristic, no API)
# ====================================================================

def score_candidate(text: str, char_names: List[str],
                    recent_ads: List[str]) -> float:
    """
    Heuristic score for an AD candidate.
    Higher is better. Favors:
      - Presence of character names
      - Conciseness (10-18 words ideal)
      - No repetition of recent ADs
      - Contains action verbs
    """
    if not text:
        return -100.0

    words = text.split()
    score = 0.0

    # 1. Length penalty -- prefer 8-18 words
    n = len(words)
    if 8 <= n <= 18:
        score += 2.0
    elif 5 <= n <= 25:
        score += 1.0
    elif n > 25:
        score -= 2.0
    elif n < 3:
        score -= 5.0

    # 2. Character name bonus
    text_lower = text.lower()
    for name in char_names:
        if name.lower() in text_lower:
            score += 1.5

    # 3. Action verb bonus
    action_verbs = [
        "walks", "runs", "sits", "stands", "picks", "drops", "opens",
        "closes", "looks", "turns", "reaches", "grabs", "holds", "pushes",
        "pulls", "throws", "catches", "kicks", "hits", "punches",
        "smiles", "frowns", "nods", "shakes", "waves", "points",
        "enters", "exits", "approaches", "moves", "leads", "follows",
    ]
    for v in action_verbs:
        if v in text_lower:
            score += 0.3
    score = min(score, 5.0)

    # 4. Repetition penalty -- check overlap with recent ADs
    for recent in recent_ads[-3:]:
        if not recent:
            continue
        overlap = SequenceMatcher(None, text_lower, recent.lower()).ratio()
        if overlap > 0.7:
            score -= 3.0
        elif overlap > 0.5:
            score -= 1.5

    # 5. Artifact penalty -- still contains reasoning
    bad_starters = [
        "the script", "based on", "here is", "the audio",
        "the description", "i will", "let me", "looking at",
    ]
    for bs in bad_starters:
        if text_lower.startswith(bs):
            score -= 10.0

    return score


def select_best(candidates: List[str], char_names: List[str],
                recent_ads: List[str]) -> str:
    """Pick the best candidate from post-processed list."""
    if not candidates:
        return ""
    if len(candidates) == 1:
        return candidates[0]

    scored = [
        (score_candidate(c, char_names, recent_ads), c)
        for c in candidates if c
    ]
    if not scored:
        return ""
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1]


# ====================================================================
# Streaming Curation
# ====================================================================

class StreamingCurator:
    """Maintains a sliding window of recent ADs per (movie, video) for deduplication."""

    def __init__(self, window: int = 5):
        self.window = window
        self.history: Dict[str, List[str]] = defaultdict(list)

    def _key(self, imdbid: str, videoid: str, clip_filename: str) -> str:
        """Key by individual clip — each clip is fully independent."""
        return f"{imdbid}::{videoid}::{clip_filename}"

    def get_recent(self, imdbid: str, videoid: str, clip_filename: str = "") -> List[str]:
        return self.history[self._key(imdbid, videoid, clip_filename)][-self.window:]

    def is_too_similar(self, ad_text: str, imdbid: str, videoid: str,
                       clip_filename: str = "", threshold: float = 0.95) -> bool:
        recent = self.get_recent(imdbid, videoid, clip_filename)
        for prev in recent:
            if SequenceMatcher(None, ad_text.lower(), prev.lower()).ratio() > threshold:
                return True
        return False

    def postprocess_with_dedup(self, ad_text: str, imdbid: str, videoid: str,
                               clip_filename: str = "") -> str:
        """Per-clip dedup: only compares against same clip's own candidates."""
        if not ad_text:
            return ""
        if self.is_too_similar(ad_text, imdbid, videoid, clip_filename, threshold=0.95):
            return ""
        return ad_text

    def add(self, imdbid: str, videoid: str, clip_filename: str, ad_text: str):
        self.history[self._key(imdbid, videoid, clip_filename)].append(ad_text)


# ====================================================================
# Helpers
# ====================================================================

def load_charbank() -> Dict[str, list]:
    if CHARBANK_JSON.exists():
        with open(CHARBANK_JSON, encoding="utf-8") as f:
            return json.load(f)
    return {}


def load_clip_manifest() -> set:
    if CLIPS_MANIFEST.exists():
        with open(CLIPS_MANIFEST) as f:
            manifest = json.load(f)
        return {Path(m["file"]).stem for m in manifest}
    return set()


def build_imdb_to_movie_map() -> Dict[str, str]:
    name2imdb_path = PROJECT_ROOT / "streamingAD" / "face_gallery_data" / "name2imdbid.json"
    if not name2imdb_path.exists():
        return {}
    with open(name2imdb_path, encoding="utf-8") as f:
        n2i = json.load(f)
    return {v: k for k, v in n2i.items()}


def find_clip_path(row: pd.Series, manifest_stems: set) -> Optional[Path]:
    split = row["split"]
    videoid = row["cmd_filename"].split("/")[-1]
    start = float(row["scaled_start"])
    end = float(row["scaled_end"])
    filename = f"{videoid}_{start:.1f}_{end:.1f}.mp4"
    path = CLIPS_DIR / split / filename
    if path.exists() and path.stat().st_size > 500:
        return path
    return None


def build_context(row: pd.Series, charbank: dict, nearby_ads: List[str],
                  detected_characters: Optional[List[str]] = None) -> str:
    """Build context text -- same as original pipeline."""
    parts = []

    imdbid = str(row.get("imdbid", "")).strip()
    movie_title = str(row.get("movie_title", "")).strip()

    if imdbid and imdbid in charbank:
        chars = charbank[imdbid]
        char_names = [c["role"] for c in chars[:10] if c.get("role")]
        if char_names:
            parts.append(f"[Movie: {movie_title}]")
            parts.append(f"[Characters in movie: {', '.join(char_names)}]")

    if detected_characters:
        parts.append(f"[Characters visible in this clip: {', '.join(detected_characters)}]")

    if nearby_ads:
        parts.append("[Previous ADs for context:]")
        for ad in nearby_ads[-3:]:
            parts.append(f"  {ad}")

    parts.append(f"\n[Task] {TASK_PROMPT}")
    return "\n".join(parts)


def get_nearby_ads(ad_df: pd.DataFrame, movie_ads: pd.DataFrame,
                   current_idx: int, window: int = 3) -> List[str]:
    movie_indices = movie_ads.index.tolist()
    try:
        pos = movie_indices.index(current_idx)
    except ValueError:
        return []
    start = max(0, pos - window)
    return [str(movie_ads.iloc[i]["text"]) for i in range(start, pos)
            if pd.notna(movie_ads.iloc[i]["text"])]


# ====================================================================
# Multi-candidate inference (streaming, per-segment)
# ====================================================================

def infer_multi_candidate(
    engine: Gemma4ADEngine,
    clip_path: Path,
    context_text: str,
    task_prompt: str,
    num_candidates: int,
    base_temp: float,
    temp_step: float,
    max_new_tokens: int,
    face_avatars: Optional[List[Path]] = None,
    character_names: Optional[List[str]] = None,
) -> Tuple[List[dict], float]:
    """
    Generate multiple AD candidates for a single clip.
    - Candidate 0: beam search (num_beams=3) for most accurate output
    - Candidate 1..N: sampling with increasing temperatures for diversity
    Returns (list_of_candidate_dicts, total_inference_time).
    Each dict has keys: 'text', 'is_beam'
    """
    candidates = []
    total_time = 0.0

    for c in range(num_candidates):
        try:
            if c == 0:
                # First candidate: beam search for accuracy
                ad_text, inf_time, _ = engine.infer_one_segment(
                    clip_path=clip_path,
                    context_text=context_text,
                    task_prompt=task_prompt,
                    temperature=base_temp,
                    max_new_tokens=max_new_tokens,
                    face_avatars=face_avatars,
                    character_names=character_names,
                    num_beams=3,
                )
                candidates.append({"text": ad_text, "is_beam": True})
            else:
                # Additional candidates: sampling with increasing temp for diversity
                temp = base_temp + c * temp_step
                ad_text, inf_time, _ = engine.infer_one_segment(
                    clip_path=clip_path,
                    context_text=context_text,
                    task_prompt=task_prompt,
                    temperature=temp,
                    max_new_tokens=max_new_tokens,
                    face_avatars=face_avatars,
                    character_names=character_names,
                    num_beams=1,
                )
                candidates.append({"text": ad_text, "is_beam": False})
            total_time += inf_time
        except Exception as e:
            print(f"      [WARN] candidate {c} failed: {e}")

    return candidates, total_time


# ====================================================================
# Main
# ====================================================================

def main():
    parser = argparse.ArgumentParser(description="Enhanced CMD-AD Pipeline (Gemma4)")
    parser.add_argument("--gpu", default=DEFAULT_GPU,
                        help="CUDA_VISIBLE_DEVICES value")
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--max-clips", type=int, default=DEFAULT_MAX_CLIPS)
    parser.add_argument("--num-candidates", type=int, default=DEFAULT_NUM_CANDIDATES)
    parser.add_argument("--base-temp", type=float, default=DEFAULT_BASE_TEMP)
    parser.add_argument("--temp-step", type=float, default=DEFAULT_TEMP_STEP)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--model-path", default=GEMMA4_MODEL_PATH,
                        help="Path to Gemma4 model")
    parser.add_argument("--selector-model", default=DEFAULT_SELECTOR_MODEL,
                        help="Path to Qwen selector model (empty string to disable)")
    parser.add_argument("--fast", action="store_true",
                        help="Fast mode: 1 candidate, no selector, beam search only")
    args = parser.parse_args()
    if args.selector_model == "":
        args.selector_model = None
    if args.fast:
        args.num_candidates = 1
        args.selector_model = None
        print("  [FAST MODE] 1 candidate, beam search, no selector")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    sys.path.insert(0, str(STREAMING_ROOT))
    sys.path.insert(0, str(PROJECT_ROOT / "streamingAD"))  # for face_gallery

    print(f"\n{'='*60}")
    print(f"[Enhanced CMD-AD Pipeline — Gemma4] split={args.split}")
    print(f"  Gemma4 model:     {args.model_path}")
    print(f"  Candidates per clip: {args.num_candidates}")
    print(f"  Temperatures: {args.base_temp} + {args.temp_step}*k")
    print(f"  Max tokens:   {args.max_tokens}")
    print(f"  Selector:     {args.selector_model or 'Gemma4 (self)'}")
    print(f"  Output:       {OUTPUT_DIR}")
    print(f"{'='*60}")

    # -- Load annotations --
    print("\n[Phase 1] Loading annotations...")
    ad_df = pd.read_csv(ANNO_CSV)
    total_all = len(ad_df)
    if args.split != "all":
        ad_df = ad_df[ad_df["split"] == args.split]
    print(f"  Total ADs: {total_all}, filtered ({args.split}): {len(ad_df)}")

    # -- Load charbank --
    print("[Phase 2] Loading charbank...")
    charbank = load_charbank()
    print(f"  {len(charbank)} movies in charbank")

    # -- Load clip manifest --
    print("[Phase 3] Loading clip manifest...")
    manifest_stems = load_clip_manifest()
    print(f"  {len(manifest_stems)} clips in manifest")

    # -- Find available clips --
    print("[Phase 4] Finding available clips...")
    entries = []
    for idx, row in ad_df.iterrows():
        clip_path = find_clip_path(row, manifest_stems)
        if clip_path is not None:
            entries.append((idx, row, clip_path))
    if args.max_clips > 0:
        entries = entries[:args.max_clips]
    print(f"  Available clips: {len(entries)}/{len(ad_df)}")
    if not entries:
        print("  No clips found. Check CLIPS_DIR.")
        return

    # -- Load face gallery --
    print("[Phase 5] Loading face gallery...")
    gallery_cache: Dict[str, Tuple] = {}
    imdb_to_movie = build_imdb_to_movie_map()
    print(f"  IMDB->movie mapping: {len(imdb_to_movie)} entries")
    try:
        from face_gallery import load_gallery
        print("  Face gallery module loaded.")
    except Exception:
        load_gallery = None
        print("  Face gallery not available, skipping.")

    # -- Build Gemma4 engine --
    print("\n[Phase 6] Loading Gemma4 model...")
    engine = build_gemma4_engine(model_path=args.model_path)

    # -- Load Qwen2.5-VL multimodal selector --
    selector = None
    if args.selector_model:
        print("[Phase 6b] Loading Qwen2.5-VL selector...")
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor as QwenProcessor

            sel_processor = QwenProcessor.from_pretrained(
                args.selector_model, trust_remote_code=True)
            sel_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                args.selector_model,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            ).to("cuda:0")  # Qwen on physical GPU 0; Gemma4 on GPU 2,3
            sel_model.eval()

            class QwenVLSelector:
                """Multimodal selector: sees video frames + candidate text."""
                def __init__(self, model, processor, device):
                    self.model = model
                    self.processor = processor
                    self.device = device

                def _extract_frames(self, clip_path: Path, num_frames: int = 8) -> list:
                    """Extract evenly-spaced frames from video clip."""
                    try:
                        r = subprocess.run(
                            ['ffprobe', '-v', 'error', '-show_entries',
                             'format=duration', '-of',
                             'default=noprint_wrappers=1:nokey=1',
                             str(clip_path)],
                            capture_output=True, text=True)
                        duration = float(r.stdout.strip())
                        if duration <= 0:
                            return []

                        frames = []
                        for i in range(num_frames):
                            t = duration * (i + 0.5) / num_frames
                            r = subprocess.run(
                                ['ffmpeg', '-ss', str(t), '-i', str(clip_path),
                                 '-frames:v', '1', '-f', 'rawvideo',
                                 '-pix_fmt', 'rgb24', '-'],
                                capture_output=True)
                            if r.returncode == 0 and len(r.stdout) == 480 * 480 * 3:
                                img = PILImage.frombytes('RGB', (480, 480), r.stdout)
                                frames.append(img)
                        return frames
                    except Exception:
                        return []

                def generate(self, prompt: str, clip_path: Path = None,
                             max_new_tokens: int = 5) -> str:
                    """Generate with optional video frames."""
                    images = []
                    if clip_path is not None:
                        images = self._extract_frames(clip_path, num_frames=8)

                    if images:
                        content = []
                        for img in images:
                            content.append({"type": "image", "image": img})
                        content.append({"type": "text", "text": prompt})
                        messages = [{"role": "user", "content": content}]
                    else:
                        messages = [{"role": "user", "content": prompt}]

                    text = self.processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True)
                    inputs = self.processor(
                        text=[text],
                        images=images if images else None,
                        return_tensors='pt',
                        padding=True,
                    ).to(self.device)

                    with torch.no_grad():
                        outputs = self.model.generate(
                            **inputs, max_new_tokens=max_new_tokens,
                            do_sample=False,
                            top_p=None, top_k=None, temperature=None)
                    generated = outputs[0][inputs["input_ids"].shape[-1]:]
                    return self.processor.decode(
                        generated, skip_special_tokens=True).strip()

            selector = QwenVLSelector(sel_model, sel_processor, "cuda:0")
            print("  Qwen2.5-VL selector loaded (multimodal).")
        except Exception as e:
            print(f"  [WARN] Failed to load Qwen selector: {e}")
            import traceback; traceback.print_exc()
            selector = None
    else:
        print("[Phase 6b] No selector model specified, using Gemma4 for selection.")

    # -- Init streaming curator --
    curator = StreamingCurator(window=5)

    # -- Inference (streaming, segment-by-segment) --
    print(f"[Phase 7] Running enhanced inference on {len(entries)} clips...")
    results = []
    total_time = 0.0

    for i, (idx, row, clip_path) in enumerate(entries):
        imdbid = str(row.get("imdbid", "")).strip()
        movie_title = str(row.get("movie_title", "")).strip()
        gt_text = str(row.get("text", "")).strip()
        split = str(row.get("split", "")).strip()
        videoid = row["cmd_filename"].split("/")[-1]

        # Nearby ADs for context — same video only (prevent cross-video bleeding)
        movie_ads = ad_df[(ad_df["imdbid"] == imdbid) & (ad_df["cmd_filename"].str.contains(videoid))]
        nearby = get_nearby_ads(ad_df, movie_ads, idx, window=3)

        # Per-clip isolation: no generated AD context injection
        recent_generated: List[str] = []
        combined_nearby = nearby

        # Face detection
        chars: List[str] = []
        face_avatars: List[Path] = []
        if load_gallery is not None and imdbid:
            if imdbid not in gallery_cache:
                try:
                    movie_name = imdb_to_movie.get(imdbid, imdbid)
                    embs, meta = load_gallery(movie_name)
                    gallery_cache[imdbid] = (embs, meta)
                except Exception:
                    gallery_cache[imdbid] = (None, None)
            embs, meta = gallery_cache[imdbid]
            if embs is not None and meta is not None:
                try:
                    from face_gallery import detect_faces_in_clip
                    chars, face_avatars = detect_faces_in_clip(
                        clip_path, embs, meta, threshold=0.35,
                    )
                except Exception:
                    pass

        # Build context
        context = build_context(row, charbank, combined_nearby,
                                detected_characters=chars)

        # -- Multi-candidate generation --
        raw_candidates, inf_time = infer_multi_candidate(
            engine=engine,
            clip_path=clip_path,
            context_text=context,
            task_prompt=TASK_PROMPT,
            num_candidates=args.num_candidates,
            base_temp=args.base_temp,
            temp_step=args.temp_step,
            max_new_tokens=args.max_tokens,
            face_avatars=face_avatars,
            character_names=chars,
        )
        total_time += inf_time

        # -- Post-process all candidates --
        processed = [postprocess_ad(cand["text"]) for cand in raw_candidates]
        processed = [c for c in processed if c]

        # -- Select best candidate via LLM --
        best = engine.select_best_candidate(processed, selector=selector, clip_path=clip_path)

        # -- Streaming curation dedup (per-clip isolation) --
        clip_stem = clip_path.stem
        best = curator.postprocess_with_dedup(best, imdbid, videoid, clip_stem)
        if best:
            curator.add(imdbid, videoid, clip_stem, best)

        results.append({
            "clip_idx": i,
            "imdbid": imdbid,
            "movie_title": movie_title,
            "videoid": videoid,
            "split": split,
            "cmd_filename": str(row.get("cmd_filename", "")),
            "scaled_start": float(row.get("scaled_start", 0)),
            "scaled_end": float(row.get("scaled_end", 0)),
            "duration": float(row.get("duration", 0)),
            "gt_text": gt_text,
            "ad_text": best,
            "all_candidates": processed,
            "characters": chars,
            "inference_time_sec": round(inf_time, 3),
        })

        if (i + 1) % 20 == 0 or (i + 1) == len(entries):
            n_valid = sum(1 for r in results if r["ad_text"])
            print(
                f"  [{i + 1}/{len(entries)}] "
                f"avg_time={total_time / (i + 1):.2f}s  "
                f"valid={n_valid}/{i + 1}  "
                f"movie={movie_title[:30]}"
            )

    # -- Save results --
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / "cmdad_output.json"

    n_valid = sum(1 for r in results if r["ad_text"])
    output = {
        "method": "pipeline_enhanced_gemma4",
        "model": args.model_path,
        "dataset": "CMD-AD",
        "split": args.split,
        "total_ad_entries": len(ad_df),
        "available_clips": len(entries),
        "generated_count": n_valid,
        "num_candidates": args.num_candidates,
        "base_temp": args.base_temp,
        "temp_step": args.temp_step,
        "max_tokens": args.max_tokens,
        "inference_total_time_sec": round(total_time, 1),
        "time_per_clip_sec": round(total_time / max(len(results), 1), 2),
        "task_prompt": TASK_PROMPT,
        "few_shot_examples": "N/A (using original prompt)",
        "ad_entries": results,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {output_file}")
    print(f"  Generated: {n_valid}/{len(entries)}")
    print(f"  Total time: {total_time:.0f}s")


if __name__ == "__main__":
    main()
