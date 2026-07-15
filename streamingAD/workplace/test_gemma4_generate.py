"""Test Gemma 4 directly generating AD for CMD-AD clips."""
import os, sys, json, time, subprocess
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

import torch
from PIL import Image as PILImage
from transformers import AutoProcessor, Gemma4ForConditionalGeneration

MODEL_PATH = "/mnt/disk1new/ylz/newAD/models/gemma-4-26b-a4b-it"
CLIPS_DIR = Path("/mnt/disk1new/ylz/newAD/datasets/cmdad/clips/eval")
ANNO = "/mnt/disk1new/ylz/newAD/datasets/cmdad/cmd_ad_anno_v1.csv"
CHARBANK_PATH = "/mnt/disk1new/ylz/newAD/datasets/cmdad/AutoAD-Zero/resources/charbanks/cmdad_charbank.json"


def extract_frames(clip_path, num_frames=8):
    """Extract frames using ffmpeg piped to PIL via PNG (more reliable)."""
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
                from io import BytesIO
                img = PILImage.open(BytesIO(r.stdout)).convert('RGB')
                frames.append(img)
            except Exception:
                pass
    return frames


def main():
    import pandas as pd
    df = pd.read_csv(ANNO)
    df = df[df['split'] == 'eval']

    with open(CHARBANK_PATH, encoding='utf-8') as f:
        charbank = json.load(f)

    # Find clips
    clips = []
    for _, row in df.head(20).iterrows():
        videoid = row['cmd_filename'].split('/')[-1]
        start = float(row['scaled_start'])
        end = float(row['scaled_end'])
        filename = f"{videoid}_{start:.1f}_{end:.1f}.mp4"
        path = CLIPS_DIR / filename
        if path.exists() and path.stat().st_size > 500:
            clips.append((path, row))

    print(f"Found {len(clips)} clips")

    # Load Gemma 4
    print("Loading Gemma 4...", flush=True)
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = Gemma4ForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16,
        trust_remote_code=True, device_map='auto').eval()
    print(f"Model loaded on {model.device}", flush=True)

    total_time = 0
    for i, (clip_path, row) in enumerate(clips):
        gt = row['text']
        imdbid = str(row.get('imdbid', '')).strip()
        movie = str(row.get('movie_title', '')).strip()

        # Build context
        context = ""
        if imdbid and imdbid in charbank:
            chars = [c['role'] for c in charbank[imdbid][:5] if c.get('role')]
            if chars:
                context = f"Movie: {movie}. Characters: {', '.join(chars)}.\n"

        frames = extract_frames(clip_path)
        if not frames:
            print(f"[{i:2d}] SKIP - no frames", flush=True)
            continue

        prompt = (
            f"{context}"
            "Describe what is happening in this video in one short sentence. "
            "Focus on visible actions and movements. "
            "Use character names if possible. Keep the description under 20 words."
        )

        content = []
        for img in frames:
            content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(
            text=[text], images=frames, return_tensors='pt',
            padding=True).to(model.device)

        start_time = time.monotonic()
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=64, do_sample=False)
        elapsed = time.monotonic() - start_time
        total_time += elapsed

        generated = outputs[0][inputs['input_ids'].shape[-1]:]
        answer = processor.decode(generated, skip_special_tokens=True).strip()

        print(f"[{i:2d}] GT: {gt}", flush=True)
        print(f"     Gemma4: {answer}", flush=True)
        print(f"     time: {elapsed:.1f}s", flush=True)
        print(flush=True)

        if (i + 1) % 5 == 0:
            print(f"  --- progress: {i+1}/{len(clips)}, avg={total_time/(i+1):.1f}s ---\n", flush=True)

    print(f"\nDone. {len(clips)} clips, total {total_time:.0f}s, avg {total_time/len(clips):.1f}s")


if __name__ == "__main__":
    main()
