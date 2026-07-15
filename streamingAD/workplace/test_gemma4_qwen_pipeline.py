"""Gemma4 generates candidates → Qwen2.5-VL selects best (multimodal).
v2: Integrated face recognition via face_gallery for character identification."""
import os, sys, json, time, subprocess, random, re
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "0,2,3"

import torch
import numpy as np
from PIL import Image as PILImage
from io import BytesIO
from typing import Dict, List, Optional, Tuple
from transformers import (
    AutoProcessor, Qwen2_5_VLForConditionalGeneration,
    Gemma4ForConditionalGeneration,
)

GEMMA4_PATH = "/mnt/disk1new/ylz/newAD/models/gemma-4-26b-a4b-it"
QWENVL_PATH = "/mnt/disk1new/cxx/models/Qwen2.5-VL-7B-Instruct"
CLIPS_DIR  = Path("/mnt/disk1new/ylz/newAD/datasets/cmdad/clips/eval")
ANNO       = "/mnt/disk1new/ylz/newAD/datasets/cmdad/cmd_ad_anno_v1.csv"
CHARBANK   = "/mnt/disk1new/ylz/newAD/datasets/cmdad/AutoAD-Zero/resources/charbanks/cmdad_charbank.json"
MAX_CLIPS  = int(sys.argv[1]) if len(sys.argv) > 1 else 50

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # streamingAD dir


# ── Frame extraction ──────────────────────────────────────

def extract_frames(clip_path, num_frames=8):
    frames = []
    r = subprocess.run(
        ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
         '-of', 'default=noprint_wrappers=1:nokey=1', str(clip_path)],
        capture_output=True, text=True)
    try:
        duration = float(r.stdout.strip())
    except ValueError:
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


# ── Face gallery loading ──────────────────────────────────

def load_face_deps():
    """Try to load face_gallery modules. Returns (load_gallery_fn, detect_faces_fn) or (None, None)."""
    try:
        from face_gallery import load_gallery, detect_faces_in_clip
        return load_gallery, detect_faces_in_clip
    except Exception as e:
        print(f"  Face gallery unavailable: {e}")
        return None, None


def build_imdb_to_movie_map(charbank, df):
    """Build imdbid → movie_name mapping from annotation data."""
    mapping = {}
    for _, row in df.iterrows():
        imdbid = str(row.get('imdbid', '')).strip()
        title = str(row.get('movie_title', '')).strip()
        if imdbid and title:
            mapping[imdbid] = title
    return mapping


# ── Character detection per clip ──────────────────────────

def detect_characters_for_clip(
    clip_path: Path,
    imdbid: str,
    imdb_to_movie: dict,
    gallery_cache: dict,
    load_gallery_fn,
    detect_faces_fn,
) -> Tuple[List[str], List[Path]]:
    """Detect characters in a clip using face recognition.
    Returns (character_names, face_avatar_paths).
    Falls back to empty if no face data available."""
    chars: List[str] = []
    avatars: List[Path] = []

    if load_gallery_fn is None or detect_faces_fn is None:
        return chars, avatars

    if not imdbid:
        return chars, avatars

    # Cache gallery per movie
    if imdbid not in gallery_cache:
        try:
            movie_name = imdb_to_movie.get(imdbid, imdbid)
            embs, meta = load_gallery_fn(movie_name)
            gallery_cache[imdbid] = (embs, meta)
        except Exception:
            gallery_cache[imdbid] = (None, None)

    embs, meta = gallery_cache[imdbid]
    if embs is None or meta is None:
        return chars, avatars

    try:
        chars, avatars = detect_faces_fn(clip_path, embs, meta, threshold=0.35)
    except Exception:
        pass

    return chars, avatars


# ── Generation ────────────────────────────────────────────

def generate_candidates(model, processor, frames, prompt,
                        num_cand=3, face_avatars=None, character_names=None):
    """Generate multiple AD candidates with different decoding strategies.
    Optionally includes face avatar images for character identification."""
    candidates = []

    # Build multimodal message: [face images] + [video frames] + [prompt]
    msg_content = []

    # Add face reference images first (so model knows who's who)
    if face_avatars and character_names:
        for avatar_path in face_avatars[:4]:  # max 4 face references
            try:
                face_img = PILImage.open(avatar_path).convert('RGB')
                msg_content.append({"type": "image", "image": face_img})
            except Exception:
                pass
        msg_content.append({
            "type": "text",
            "text": f"[Character faces: {', '.join(character_names)}. "
                    "Match these faces to people in the video below.]"
        })

    # Add video frames
    for img in frames:
        msg_content.append({"type": "image", "image": img})

    # Add prompt
    msg_content.append({"type": "text", "text": prompt})

    messages = [{"role": "user", "content": msg_content}]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)

    for c in range(num_cand):
        if c == 0:
            do_sample = False
            temperature = None
            num_beams = 3
        else:
            do_sample = True
            temperature = 0.3 + c * 0.2
            num_beams = 1

        inputs = processor(
            text=[text], images=[img for item in msg_content if item['type'] == 'image' for img in [item['image']]],
            return_tensors='pt', padding=True).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=64,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                num_beams=num_beams,
            )
        generated = outputs[0][inputs['input_ids'].shape[-1]:]
        answer = processor.decode(generated, skip_special_tokens=True).strip()
        candidates.append(answer)

    return candidates


# ── Selection ─────────────────────────────────────────────

SELECTOR_SYSTEM = (
    "You will be provided with several audio descriptions of a video clip.\n"
    "Your task is to choose the one that is most helpful to the blind.\n"
    "A good AD: describes specific visible actions, uses character names, is under 20 words.\n"
    "A bad AD: vague, only describes static poses, misses main action.\n"
    "Reply with ONLY the number of the best description."
)

FEW_SHOT = [
    ("Dix Handley lies on a couch, his eyes closed.\nDix Handley lies on a couch.\n"
     "The woman groggily opens her eyes and sees him standing over her.", "3"),
    ("Dix Handley sits on a couch.\nThe blonde saunters over and leans in for a kiss.\n"
     "Doll Conovan sits on a couch, looking at Dix.", "2"),
    ("Dix Handley opens a safe deposit box.\nHe opens the safe door and pulls out a stack of cash.\n"
     "Dix Handley and Alonzo Emmerich are in a room.", "2"),
    ("Dix Handley looks at Alonzo D.\nDix takes a drag on his cigarette.\n"
     "Dix Handley and Alonzo Emmerich stand in a room.", "2"),
]


def select_best(selector_model, selector_processor, candidates, frames):
    """Qwen2.5-VL multimodal selection."""
    if len(candidates) <= 1:
        return candidates[0] if candidates else ""

    # Shuffle to avoid positional bias
    indexed = list(enumerate(candidates))
    random.shuffle(indexed)
    shuffled = [c for _, c in indexed]

    # Build few-shot examples
    few_shot_text = ""
    for ex_cands, ex_ans in FEW_SHOT:
        few_shot_text += f"\n{ex_cands}Answer: {ex_ans}\n"

    cand_text = "\n".join(f"{i+1}. {c}" for i, c in enumerate(shuffled))

    prompt = (
        f"{SELECTOR_SYSTEM}\n"
        f"\nExamples:\n{few_shot_text}"
        f"\nNow choose the best:\n{cand_text}\n"
        f"Answer:"
    )

    # Build multimodal input
    content = []
    for img in frames:
        content.append({"type": "image", "image": img})
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]

    text = selector_processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    inputs = selector_processor(
        text=[text], images=frames, return_tensors='pt',
        padding=True).to(selector_model.device)

    with torch.no_grad():
        outputs = selector_model.generate(
            **inputs, max_new_tokens=5, do_sample=False)
    generated = outputs[0][inputs['input_ids'].shape[-1]:]
    answer = selector_processor.decode(generated, skip_special_tokens=True).strip()

    # Parse answer
    nums = re.findall(r'\d+', answer)
    if nums:
        shuffled_idx = int(nums[0]) - 1
        if 0 <= shuffled_idx < len(shuffled):
            return shuffled[shuffled_idx]

    return candidates[0]  # fallback


# ── Main pipeline ─────────────────────────────────────────

def main():
    import pandas as pd
    df = pd.read_csv(ANNO)
    df = df[df['split'] == 'eval']

    with open(CHARBANK, encoding='utf-8') as f:
        charbank = json.load(f)

    # Find clips
    clips = []
    for _, row in df.iterrows():
        videoid = row['cmd_filename'].split('/')[-1]
        start = float(row['scaled_start'])
        end = float(row['scaled_end'])
        filename = f"{videoid}_{start:.1f}_{end:.1f}.mp4"
        path = CLIPS_DIR / filename
        if path.exists() and path.stat().st_size > 500:
            clips.append((path, row))
            if len(clips) >= MAX_CLIPS:
                break

    print(f"Found {len(clips)} clips")

    # Load face gallery module
    print("Loading face gallery...", flush=True)
    load_gallery_fn, detect_faces_fn = load_face_deps()
    imdb_to_movie = build_imdb_to_movie_map(charbank, df)
    gallery_cache: Dict[str, Tuple] = {}
    face_hits = 0

    # Load Gemma 4 → GPU 2,3
    print("Loading Gemma 4 (GPU 2,3)...", flush=True)
    gemma_proc = AutoProcessor.from_pretrained(GEMMA4_PATH, trust_remote_code=True)
    gemma_model = Gemma4ForConditionalGeneration.from_pretrained(
        GEMMA4_PATH, torch_dtype=torch.bfloat16,
        device_map='auto', trust_remote_code=True).eval()
    print(f"  Gemma 4 loaded", flush=True)

    # Load Qwen2.5-VL → GPU 0
    print("Loading Qwen2.5-VL (GPU 0)...", flush=True)
    qwen_proc = AutoProcessor.from_pretrained(QWENVL_PATH, trust_remote_code=True)
    qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        QWENVL_PATH, torch_dtype=torch.bfloat16,
        trust_remote_code=True, device_map=0).eval()
    print(f"  Qwen2.5-VL loaded on GPU 0", flush=True)

    results = []
    total_time = 0
    valid = 0

    for i, (clip_path, row) in enumerate(clips):
        gt = row['text']
        imdbid = str(row.get('imdbid', '')).strip()
        movie = str(row.get('movie_title', '')).strip()

        # ── Face detection for this clip ──
        detected_chars, face_avatars = detect_characters_for_clip(
            clip_path, imdbid, imdb_to_movie, gallery_cache,
            load_gallery_fn, detect_faces_fn,
        )
        if detected_chars:
            face_hits += 1

        # ── Build context ──
        # Strategy:
        # - If face detection found characters → only list those (reliable)
        # - Otherwise → don't list characters at all (avoid misleading the model)
        if detected_chars:
            # Face-detected characters are reliable
            context = (f"Movie: {movie}. "
                       f"Characters in this scene: {', '.join(detected_chars)}.\n")
        elif imdbid and imdbid in charbank:
            # Fallback: list movie characters but with a caveat
            # chars = [c['role'] for c in charbank[imdbid][:5] if c.get('role')]
            # context = f"Movie: {movie}. Characters: {', '.join(chars)}.\n"
            # OLD behavior (causes confusion) — now we skip characters entirely
            context = f"Movie: {movie}.\n"
        else:
            context = ""

        frames = extract_frames(clip_path)
        if len(frames) < 3:
            print(f"[{i:3d}] SKIP - insufficient frames", flush=True)
            continue

        prompt = (
            f"{context}"
            "Describe what is happening in this video in one short sentence. "
            "Focus on visible actions and movements. "
            "Use character names if known from the faces provided. "
            "Keep the description under 20 words."
        )

        # Generate candidates (with face avatars if available)
        try:
            cands = generate_candidates(
                gemma_model, gemma_proc, frames, prompt,
                num_cand=3,
                face_avatars=face_avatars if detected_chars else None,
                character_names=detected_chars if detected_chars else None,
            )
        except Exception as e:
            print(f"[{i:3d}] SKIP - generation error: {e}", flush=True)
            continue

        # Select best
        try:
            best = select_best(qwen_model, qwen_proc, cands, frames)
        except Exception as e:
            print(f"[{i:3d}] SKIP - selection error: {e}", flush=True)
            best = cands[0] if cands else ""

        face_tag = f"[FACE:{','.join(detected_chars)}]" if detected_chars else "[no-face]"
        print(f"[{i:3d}] {face_tag} GT  : {gt}", flush=True)
        print(f"      AD  : {best}", flush=True)
        for ci, c in enumerate(cands):
            sel = " >>>" if c == best else "    "
            print(f"     {sel} [{ci}]: {c}", flush=True)
        print(flush=True)

        results.append({
            "idx": i, "gt": gt, "ad": best, "candidates": cands,
            "detected_chars": detected_chars,
        })
        valid += 1

        if (i + 1) % 10 == 0:
            print(f"  --- progress: {i+1}/{len(clips)}, face hits: {face_hits} ---\n", flush=True)

    # Summary
    avg_time = total_time / valid if valid > 0 else 0
    print(f"\n{'='*60}")
    print(f"Gemma4 + Qwen2.5-VL (v2 with face recognition)")
    print(f"  Clips: {valid}/{len(clips)} valid, {face_hits} with face detection")
    print(f"  Avg time: {avg_time:.1f}s/clip")
    print(f"{'='*60}")

    # Save
    out_path = Path("/mnt/disk1new/ylz/newAD/compare/gemma4_qwen_v2/gemma4_qwen_v2_output.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
