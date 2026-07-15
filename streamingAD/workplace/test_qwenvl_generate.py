"""Quick test: use Qwen2.5-VL to generate AD directly."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from PIL import Image as PILImage
import subprocess
import json
import time
from pathlib import Path

MODEL_PATH = "/mnt/disk1new/cxx/models/Qwen2.5-VL-7B-Instruct"
CLIPS_DIR = Path("/mnt/disk1new/ylz/newAD/datasets/cmdad/clips/eval")
ANNO = "/mnt/disk1new/ylz/newAD/datasets/cmdad/cmd_ad_anno_v1.csv"

TASK_PROMPT = (
    "Describe what is happening in this video in one short sentence. "
    "Use character names if known. Focus on actions. Max 15 words."
)

CHARBANK_PATH = "/mnt/disk1new/ylz/newAD/datasets/cmdad/AutoAD-Zero/resources/charbanks/cmdad_charbank.json"

def extract_frames(clip_path, num_frames=8):
    frames = []
    r = subprocess.run(
        ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
         '-of', 'default=noprint_wrappers=1:nokey=1', str(clip_path)],
        capture_output=True, text=True)
    duration = float(r.stdout.strip())
    for i in range(num_frames):
        t = duration * (i + 0.5) / num_frames
        r = subprocess.run(
            ['ffmpeg', '-ss', str(t), '-i', str(clip_path),
             '-frames:v', '1', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'],
            capture_output=True)
        if r.returncode == 0 and len(r.stdout) == 480 * 480 * 3:
            frames.append(PILImage.frombytes('RGB', (480, 480), r.stdout))
    return frames

def main():
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    import pandas as pd
    df = pd.read_csv(ANNO)
    df = df[df['split'] == 'eval']

    # Load charbank
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

    # Load Qwen2.5-VL
    print("Loading Qwen2.5-VL...")
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16,
        trust_remote_code=True).cuda().eval()
    print("Model loaded.\n")

    # Generate AD for each clip
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
            continue

        prompt = f"{context}{TASK_PROMPT}"

        content = []
        for img in frames:
            content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(
            text=[text], images=frames, return_tensors='pt',
            padding=True).to('cuda')

        start_time = time.monotonic()
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=64, do_sample=False)
        elapsed = time.monotonic() - start_time

        generated = outputs[0][inputs['input_ids'].shape[-1]:]
        answer = processor.decode(generated, skip_special_tokens=True).strip()

        print(f"[{i:2d}] GT: {gt}")
        print(f"     Qwen2.5-VL: {answer}")
        print(f"     time: {elapsed:.1f}s")
        print()
        import sys; sys.stdout.flush()

if __name__ == "__main__":
    main()
