#!/usr/bin/env python3
"""
query_stream_ad.py — QueryStream-style streaming AD generation.

Implements QueryStream (ICLR 2026) for Audio Description:
  - QDP: Query-Aware Differential Pruning (semantic + temporal token filtering)
  - RTAR: Relevance-Triggered Active Response (dual-gated response scheduling)

Streaming constraint: only past + current frames are visible (causal mask).
Uses CLIP (ViT-B/32) for patch-level feature extraction — training-free, plug-and-play.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

# ── Default Config ───────────────────────────────────────────
DEFAULT_CLIP_MODEL = "openai/clip-vit-base-patch32"
DEFAULT_QUERY = (
    "Describe what is happening in this clip concisely. "
    "Focus on visible actions, movements, and expressions. "
    "If character names are mentioned in the context, use them "
    "(e.g. 'Don Vito Corleone walks...' not 'A man walks...'). "
    "Do not quote dialogue."
)
FPS: float = 1.0
FRAME_SIZE: int = 224
TAU_RELEVANCE: float = 0.15
TAU_DENSITY: float = 0.15
EMA_ALPHA: float = 0.3
MIN_SEGMENT_SEC: float = 3.0
MAX_SEGMENT_SEC: float = 30.0
# ─────────────────────────────────────────────────────────────


def _extract_frames(
    video_path: Path,
    start_sec: float,
    end_sec: float,
    output_dir: Path,
    fps: float = FPS,
    size: int = FRAME_SIZE,
) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pattern = output_dir / "frame_%06d.jpg"
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{start_sec:.3f}",
        "-t", f"{end_sec - start_sec:.3f}",
        "-i", str(video_path),
        "-vf", f"fps={fps},scale={size}:{size}",
        str(pattern),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return sorted(output_dir.glob("frame_*.jpg"))


class StreamCLIPEncoder:
    """Lightweight CLIP encoder for patch-level feature extraction."""

    def __init__(self, model_name: str = DEFAULT_CLIP_MODEL, device: str = "cuda:0"):
        from transformers import CLIPModel, CLIPProcessor
        self.device = device
        self.model = CLIPModel.from_pretrained(model_name).to(device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name, truncation=True, max_length=77)
        self._query_embed: Optional[torch.Tensor] = None
        self._embed_dim: Optional[int] = None

    @torch.no_grad()
    def set_query(self, query_text: str) -> torch.Tensor:
        inputs = self.processor(text=[query_text], return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        text_outputs = self.model.text_model(**inputs)
        pooled = text_outputs.pooler_output
        self._query_embed = self.model.text_projection(pooled)
        self._query_embed = F.normalize(self._query_embed, dim=-1)
        return self._query_embed

    @property
    def query_embed(self) -> torch.Tensor:
        if self._query_embed is None:
            self.set_query(DEFAULT_QUERY)
        return self._query_embed

    @torch.no_grad()
    def encode_frame(
        self, frame_path: Path
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        from PIL import Image
        image = Image.open(frame_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)

        vision_outputs = self.model.vision_model(pixel_values=pixel_values)
        patch_hidden = vision_outputs.last_hidden_state[:, 1:, :]

        patch_proj = self.model.visual_projection(patch_hidden)
        patch_proj = F.normalize(patch_proj, dim=-1)

        cls_pooled = vision_outputs.pooler_output
        frame_proj = self.model.visual_projection(cls_pooled)
        frame_proj = F.normalize(frame_proj, dim=-1)

        return patch_proj, frame_proj

    @torch.no_grad()
    def encode_frames_batch(
        self,
        frame_paths: List[Path],
        batch_size: int = 64,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        from PIL import Image

        all_patch_proj: List[torch.Tensor] = []
        all_frame_proj: List[torch.Tensor] = []

        total = len(frame_paths)
        for i in range(0, total, batch_size):
            batch_paths = frame_paths[i:i + batch_size]
            images = [Image.open(p).convert("RGB") for p in batch_paths]
            inputs = self.processor(images=images, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(self.device)

            vision_outputs = self.model.vision_model(pixel_values=pixel_values)
            patch_hidden = vision_outputs.last_hidden_state[:, 1:, :]

            patch_proj = self.model.visual_projection(patch_hidden)
            patch_proj = F.normalize(patch_proj, dim=-1)

            cls_pooled = vision_outputs.pooler_output
            frame_proj = self.model.visual_projection(cls_pooled)
            frame_proj = F.normalize(frame_proj, dim=-1)

            all_patch_proj.append(patch_proj.detach().cpu())
            all_frame_proj.append(frame_proj.detach().cpu())

            if self.device.startswith("cuda"):
                torch.cuda.empty_cache()

        patch_proj_all = torch.cat(all_patch_proj, dim=0)
        frame_proj_all = torch.cat(all_frame_proj, dim=0)
        self._embed_dim = patch_proj_all.shape[-1]
        return patch_proj_all, frame_proj_all


class QDPFilter:
    """
    Query-Aware Differential Pruning.

    For each frame:
      M_sem: cosine_sim(patch_proj, query_embed) > frame_mean_similarity
      M_temp: |patch_hidden - EMA_history| > τ_temp

    Keep = M_sem AND M_temp (both conditions must pass)
    """

    def __init__(
        self,
        clip_encoder: StreamCLIPEncoder,
        tau_relevance: float = TAU_RELEVANCE,
        tau_novelty: float = 0.1,
        ema_alpha: float = EMA_ALPHA,
    ):
        self.encoder = clip_encoder
        self.tau_relevance = tau_relevance
        self.tau_novelty = tau_novelty
        self.ema_alpha = ema_alpha
        self._ema_history: Optional[torch.Tensor] = None
        self._patch_dim: Optional[int] = None

    def reset(self):
        self._ema_history = None

    @torch.no_grad()
    def process_frame(
        self, frame_path: Path
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        patch_proj, frame_proj = self.encoder.encode_frame(frame_path)
        query_embed = self.encoder.query_embed

        sim_patch_query = torch.matmul(patch_proj, query_embed.T).squeeze(-1)
        frame_mean_sim = sim_patch_query.mean()
        M_sem = sim_patch_query > frame_mean_sim

        if self._ema_history is None:
            self._ema_history = patch_proj.detach().clone()
            self._patch_dim = patch_proj.shape[-1]
            M_temp = torch.ones(patch_proj.shape[1], dtype=torch.bool, device=patch_proj.device)
        else:
            if patch_proj.shape[1] != self._ema_history.shape[1]:
                min_patches = min(patch_proj.shape[1], self._ema_history.shape[1])
                patch_proj = patch_proj[:, :min_patches, :]
                self._ema_history = self._ema_history[:, :min_patches, :]
                M_sem = M_sem[:min_patches]

            diff = torch.norm(patch_proj - self._ema_history, dim=-1).squeeze(0)
            M_temp = diff > self.tau_novelty

            self._ema_history = (
                self.ema_alpha * patch_proj + (1 - self.ema_alpha) * self._ema_history
            )

        keep_mask = M_sem & M_temp
        keep_rate = keep_mask.float().mean().item()

        frame_relevance = sim_patch_query.max().item()

        return keep_mask, patch_proj, keep_rate


class RTARScheduler:
    """
    Relevance-Triggered Active Response — dual-gated response scheduler.

    For AD (universal query), uses RELATIVE spike detection:
      Trigger when R_t or D_t exceeds running_mean + n_sigma * running_std,
      meaning a significant deviation from recent history (scene change).

    Also enforces min/max segment length bounds.
    """

    def __init__(
        self,
        clip_encoder: StreamCLIPEncoder,
        tau_relevance: float = TAU_RELEVANCE,
        tau_density: float = TAU_DENSITY,
        min_segment_sec: float = MIN_SEGMENT_SEC,
        max_segment_sec: float = MAX_SEGMENT_SEC,
        fps: float = FPS,
        spike_sigma: float = 2.0,
        history_window: int = 30,
    ):
        self.encoder = clip_encoder
        self.tau_relevance = tau_relevance
        self.tau_density = tau_density
        self.spike_sigma = spike_sigma
        self.min_frames = int(min_segment_sec * fps)
        self.max_frames = int(max_segment_sec * fps)
        self.fps = fps

        self._frame_count: int = 0
        self._segment_start_idx: int = 0
        self._last_trigger_frame: int = 0

        self._r_history: Deque[float] = deque(maxlen=history_window)
        self._d_history: Deque[float] = deque(maxlen=history_window)

    def reset(self):
        self._frame_count = 0
        self._segment_start_idx = 0
        self._last_trigger_frame = 0
        self._r_history.clear()
        self._d_history.clear()

    @torch.no_grad()
    def step(
        self, frame_path: Path, keep_rate: float
    ) -> Tuple[bool, Optional[Tuple[int, int]]]:
        _, frame_proj = self.encoder.encode_frame(frame_path)
        query_embed = self.encoder.query_embed
        R_t = torch.matmul(frame_proj, query_embed.T).squeeze(-1).item()
        D_t = keep_rate

        self._r_history.append(R_t)
        self._d_history.append(D_t)

        trigger = False
        segment_range: Optional[Tuple[int, int]] = None

        seg_len = self._frame_count - self._segment_start_idx

        if seg_len >= self.min_frames:
            r_arr = np.array(self._r_history) if len(self._r_history) >= 3 else np.array([R_t])
            d_arr = np.array(self._d_history) if len(self._d_history) >= 3 else np.array([D_t])

            r_mean = float(np.mean(r_arr))
            r_std = float(np.std(r_arr)) if len(r_arr) > 1 else 0.01
            d_mean = float(np.mean(d_arr))
            d_std = float(np.std(d_arr)) if len(d_arr) > 1 else 0.01

            r_threshold = r_mean + self.spike_sigma * r_std
            d_threshold = d_mean + self.spike_sigma * d_std

            r_spike = R_t > r_threshold
            d_spike = D_t > d_threshold

            if d_spike:
                trigger = True

        if not trigger and seg_len >= self.max_frames:
            trigger = True

        if trigger:
            segment_range = (self._segment_start_idx, self._frame_count + 1)
            self._segment_start_idx = self._frame_count + 1
            self._last_trigger_frame = self._frame_count

        self._frame_count += 1
        return trigger, segment_range

    def force_flush(self) -> Optional[Tuple[int, int]]:
        if self._frame_count > self._segment_start_idx:
            sr = (self._segment_start_idx, self._frame_count)
            self._segment_start_idx = self._frame_count
            return sr
        return None


class QueryStreamADGenerator:
    """
    Streaming AD generator using QueryStream principles:
      QDP filters frames on-the-fly
      RTAR decides when to trigger AD generation
      Triggers Video-LLaMA only at opportune moments
    """

    def __init__(
        self,
        engine,
        video_path: Path,
        xlsx_path,
        output_dir: Path,
        gpu_id: int = 0,
        clip_model: str = DEFAULT_CLIP_MODEL,
        query_text: str = DEFAULT_QUERY,
        tau_relevance: float = TAU_RELEVANCE,
        tau_density: float = TAU_DENSITY,
        tau_novelty: float = 0.1,
        ema_alpha: float = EMA_ALPHA,
        fps: float = FPS,
        temperature: float = 0.2,
        max_new_tokens: int = 256,
        min_segment_sec: float = MIN_SEGMENT_SEC,
        max_segment_sec: float = MAX_SEGMENT_SEC,
        spike_sigma: float = 2.0,
        history_window: int = 30,
        vad_silence_db: float = -30.0,
        vad_min_silence_dur: float = 1.5,
    ):
        self.engine = engine
        self.video_path = video_path
        self.xlsx_path = xlsx_path
        self.output_dir = output_dir
        self.device = f"cuda:{gpu_id}"
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.fps = fps
        self._is_script_free = xlsx_path is None

        self._dialogue_by_time: List[Tuple[float, float, str, str, str]] = []
        self._face_avatars_by_name: Dict[str, List[Path]] = {}

        if xlsx_path is not None:
            from batch_generate_ad import (
                parse_dialogue_rows,
                build_gap_context,
                DialogueRow,
            )
            rows = parse_dialogue_rows(xlsx_path)
            for r in rows:
                self._dialogue_by_time.append(
                    (r.start_sec, r.end_sec, r.dialog, r.characters, r.location)
                )
        else:
            from vad_gap_detector import detect_gaps_from_video, AudioGap
            gaps, _scene_bounds = detect_gaps_from_video(
                video_path=video_path,
                min_gap_sec=0.1,
                max_gap_sec=999999,
                silence_threshold_db=vad_silence_db,
                min_silence_dur=vad_min_silence_dur,
                use_scene_detect=False,
            )
            self._vad_speech_segments: List[Tuple[float, float]] = []
            prev_end = 0.0
            for g in gaps:
                if g.gap_start_sec > prev_end + 0.3:
                    self._vad_speech_segments.append((prev_end, g.gap_start_sec))
                prev_end = g.gap_end_sec
            self._dialogue_by_time = [
                (s, e, "", "", "") for s, e in self._vad_speech_segments
            ]
            print(f"  [VAD] Detected {len(self._vad_speech_segments)} speech segments from audio")

        if not self._is_script_free:
            try:
                from face_gallery import load_gallery, lookup_face_images
                _, gallery_people = load_gallery(video_path.stem)
                if gallery_people is not None:
                    all_chars: set = set()
                    for _, _, _, chars_str, _ in self._dialogue_by_time:
                        for c in chars_str.split(","):
                            c = c.strip()
                            if c:
                                all_chars.add(c)
                    for char_name in sorted(all_chars):
                        avatars = lookup_face_images(
                            [char_name], gallery_people, video_path.stem,
                        )
                        if avatars:
                            self._face_avatars_by_name[char_name.lower()] = avatars
                    if self._face_avatars_by_name:
                        print(f"  [Face] {len(self._face_avatars_by_name)} character avatars loaded")
            except Exception:
                pass

        print(f"  CLIP model: {clip_model}")
        self.clip_encoder = StreamCLIPEncoder(model_name=clip_model, device=self.device)
        self.clip_encoder.set_query(query_text)

        self.qdp = QDPFilter(
            self.clip_encoder,
            tau_relevance=tau_relevance,
            tau_novelty=tau_novelty,
            ema_alpha=ema_alpha,
        )
        self.rtar = RTARScheduler(
            self.clip_encoder,
            tau_relevance=tau_relevance,
            tau_density=tau_density,
            min_segment_sec=min_segment_sec,
            max_segment_sec=max_segment_sec,
            spike_sigma=spike_sigma,
            history_window=history_window,
            fps=fps,
        )

        self._frame_paths: List[Path] = []

    def _align_to_dialogue_gap(
        self, start_sec: float, end_sec: float, min_gap_sec: float = 3.0
    ) -> Tuple[float, float, bool]:
        intervals = sorted(self._dialogue_by_time, key=lambda x: (x[0], x[1]))
        if not intervals:
            return end_sec, end_sec, False

        last_overlap_end = start_sec
        for ds, de, *_ in intervals:
            if de > start_sec and ds < end_sec:
                if de > last_overlap_end:
                    last_overlap_end = de

        if last_overlap_end < end_sec:
            return end_sec, end_sec, False

        for i in range(len(intervals) - 1):
            prev_end = intervals[i][1]
            next_start = intervals[i + 1][0]
            gap_dur = next_start - prev_end
            if prev_end >= last_overlap_end and gap_dur >= min_gap_sec:
                return prev_end, prev_end, True

        return end_sec, end_sec, False

    @staticmethod
    def _merge_overlapping_segments(
        raw_segs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        if not raw_segs:
            return []
        merged: List[Dict[str, Any]] = [dict(raw_segs[0])]
        for seg in raw_segs[1:]:
            prev = merged[-1]
            if seg["s_idx"] < prev["e_idx"]:
                prev["e_idx"] = max(prev["e_idx"], seg["e_idx"])
                prev["end_sec"] = max(prev["end_sec"], seg["end_sec"])
                prev["avg_keep"] = max(prev["avg_keep"], seg["avg_keep"])
                prev["aligned"] = prev["aligned"] or seg["aligned"]
                if seg.get("output_time") is not None:
                    prev["output_time"] = seg["output_time"]
            else:
                merged.append(dict(seg))
        return merged

    def _get_context_for_range(
        self, start_sec: float, end_sec: float, output_time_sec: Optional[float] = None
    ) -> str:
        if self._is_script_free:
            parts: List[str] = []
            parts.append("[Scene context]")
            parts.append(
                f"This is a silent pause between conversations "
                f"({end_sec - start_sec:.1f}s)."
            )
            parts.append(
                "\n[Instructions]\n"
                "Describe only what is VISIBLE on screen during this gap. "
                "Focus on actions, gestures, facial expressions, camera movements, "
                "and environmental changes."
            )
            return "\n".join(parts)

        ctx_before: List[str] = []
        ctx_after: List[str] = []
        chars_set: set = set()
        location = ""

        for ds, de, dialog, chars_str, loc in self._dialogue_by_time:
            if de <= start_sec:
                ctx_before.append(dialog)
                if len(ctx_before) > 5:
                    ctx_before.pop(0)
                for c in chars_str.split(","):
                    c = c.strip()
                    if c:
                        chars_set.add(c)
                if loc:
                    location = loc
            elif ds >= end_sec:
                ctx_after.append(dialog)
                if len(ctx_after) >= 5:
                    break

        parts: List[str] = []
        parts.append("[Scene context]")
        if location:
            parts.append(f"Location: {location}")
        if chars_set:
            parts.append(f"Characters: {', '.join(sorted(chars_set))}")
        if ctx_before:
            parts.append("\n[Dialogue before the gap]")
            for d in ctx_before[-5:]:
                parts.append(f"- {d}")
        if ctx_after:
            parts.append("\n[Dialogue after the gap]")
            for d in ctx_after[:5]:
                parts.append(f"- {d}")

        if output_time_sec is not None and abs(output_time_sec - end_sec) > 0.5:
            parts.append(
                f"\n[Timing]\n"
                f"This segment spans {end_sec - start_sec:.1f}s. "
                f"The AD will be spoken at {output_time_sec - start_sec:.1f}s from now "
                f"during a dialogue gap. Describe the entire content from {start_sec:.0f}s "
                f"to {output_time_sec:.0f}s comprehensively."
            )
        else:
            parts.append(
                f"\n[Gap description]\n"
                f"This is a silent interval ({end_sec - start_sec:.1f}s) between dialogue segments."
            )
        return "\n".join(parts)

    def _chars_near_time(self, start_sec: float, end_sec: float, window: float = 10.0) -> List[str]:
        chars_set: set = set()
        for ds, de, _, chars_str, _ in self._dialogue_by_time:
            if de >= start_sec - window and ds <= end_sec + window:
                for c in chars_str.split(","):
                    c = c.strip()
                    if c:
                        chars_set.add(c)
        return sorted(chars_set)

    def run(self) -> Dict[str, Any]:
        movie_name = self.video_path.stem
        print(f"\n{'='*60}")
        print(f"QueryStream AD: {movie_name}")
        print(f"{'='*60}")

        total_duration_sec = self._get_video_duration()
        print(f"  Duration: {total_duration_sec:.0f}s, FPS: {self.fps}")

        self.qdp.reset()
        self.rtar.reset()

        frames_dir = self.output_dir / f".tmp_qstream_{movie_name}"
        frames_dir.mkdir(parents=True, exist_ok=True)

        print(f"  Extracting frames ...")
        t0 = time.monotonic()
        all_frames = _extract_frames(
            self.video_path, 0.0, total_duration_sec, frames_dir, fps=self.fps
        )
        print(f"    {len(all_frames)} frames in {time.monotonic() - t0:.1f}s")

        self._frame_paths = all_frames
        raw_segments: List[Dict[str, Any]] = []
        total_keep_rates: List[float] = []
        dialogue_overlaps_fixed: int = 0

        print(f"  Streaming through {len(all_frames)} frames ...")
        stream_t0 = time.monotonic()
        frame_idx = 0

        for frame_idx, fp in enumerate(all_frames):
            keep_mask, _, keep_rate = self.qdp.process_frame(fp)
            total_keep_rates.append(keep_rate)

            triggered, seg_range = self.rtar.step(fp, keep_rate)

            if triggered and seg_range is not None:
                s_idx, e_idx = seg_range
                s_idx = max(0, int(s_idx))
                e_idx = min(len(all_frames), int(e_idx))
                start_sec = s_idx / self.fps
                end_sec = e_idx / self.fps

                new_end, output_time, aligned = self._align_to_dialogue_gap(
                    start_sec, end_sec
                )
                if aligned:
                    dialogue_overlaps_fixed += 1
                    e_idx = min(len(all_frames), int(output_time * self.fps))
                    end_sec = new_end

                avg_keep = (
                    np.mean(total_keep_rates[s_idx:e_idx]) if s_idx < e_idx else 0
                )

                raw_segments.append({
                    "s_idx": s_idx, "e_idx": e_idx,
                    "start_sec": start_sec, "end_sec": end_sec,
                    "avg_keep": avg_keep,
                    "aligned": aligned, "output_time": output_time if aligned else None,
                })

        remaining = self.rtar.force_flush()
        if remaining is not None:
            s_idx = max(0, int(remaining[0]))
            e_idx = min(len(all_frames), int(remaining[1]))
            start_sec = s_idx / self.fps
            end_sec = e_idx / self.fps

            new_end, output_time, aligned = self._align_to_dialogue_gap(
                start_sec, end_sec
            )
            if aligned:
                dialogue_overlaps_fixed += 1
                e_idx = min(len(all_frames), int(output_time * self.fps))
                end_sec = new_end

            avg_keep = np.mean(total_keep_rates[s_idx:e_idx]) if s_idx < e_idx else 0
            raw_segments.append({
                "s_idx": s_idx, "e_idx": e_idx,
                "start_sec": start_sec, "end_sec": end_sec,
                "avg_keep": avg_keep,
                "aligned": aligned, "output_time": output_time if aligned else None,
                "final": True,
            })

        merged_segs = self._merge_overlapping_segments(raw_segments)
        print(f"  Raw: {len(raw_segments)} → Merged: {len(merged_segs)} segments")

        segments: List[Dict[str, Any]] = []
        for i, mseg in enumerate(merged_segs):
            s_idx, e_idx = mseg["s_idx"], mseg["e_idx"]
            seg_frames = all_frames[s_idx:e_idx]
            dur = mseg["end_sec"] - mseg["start_sec"]
            tag = " [ALIGNED]" if mseg["aligned"] else ""

            print(
                f"  [Seg {i+1}/{len(merged_segs)}] "
                f"frames {s_idx}-{e_idx} ({dur:.1f}s) "
                f"keep_rate={mseg['avg_keep']:.3f}{tag}"
            )

            entry = self._generate_ad_for_segment(
                seg_frames, s_idx, e_idx,
                mseg["start_sec"], mseg["end_sec"], mseg["avg_keep"],
                output_time_sec=mseg["output_time"],
            )
            if entry:
                segments.append(entry)

        stream_elapsed = time.monotonic() - stream_t0
        inference_total = sum(
            e.get("inference_time_sec", 0) for e in segments
        )
        avg_keep = np.mean(total_keep_rates) if total_keep_rates else 0
        total_proc = stream_elapsed + inference_total
        time_per_sec = total_proc / total_duration_sec if total_duration_sec > 0 else 0
        print(
            f"  Done. {len(segments)} segments, "
            f"{dialogue_overlaps_fixed} dialogue-aligned, "
            f"avg keep_rate={avg_keep:.3f}, "
            f"preproc={stream_elapsed:.1f}s, "
            f"infer={inference_total:.1f}s, "
            f"total={total_proc:.1f}s ({time_per_sec:.2f}s per video sec)"
        )

        import shutil
        try:
            shutil.rmtree(str(frames_dir), ignore_errors=True)
        except OSError:
            pass

        result = {
            "movie": movie_name,
            "video_path": str(self.video_path),
            "method": "QueryStream",
            "params": {
                "tau_novelty": self.qdp.tau_novelty,
                "ema_alpha": self.qdp.ema_alpha,
                "spike_sigma": self.rtar.spike_sigma,
                "min_segment_sec": self.rtar.min_frames / self.rtar.fps,
                "max_segment_sec": self.rtar.max_frames / self.rtar.fps,
                "fps": self.fps,
            },
            "total_frames": len(all_frames),
            "video_duration_sec": round(total_duration_sec, 1),
            "avg_keep_rate": round(avg_keep, 4),
            "preprocess_time_sec": round(stream_elapsed, 1),
            "inference_total_time_sec": round(inference_total, 1),
            "total_time_sec": round(total_proc, 1),
            "time_per_video_sec": round(time_per_sec, 3),
            "dialogue_overlaps_fixed": dialogue_overlaps_fixed,
            "total_segments": len(segments),
            "ad_entries": segments,
        }

        out_file = self.output_dir / f"{movie_name}_ad_output.json"
        with out_file.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"  Saved: {out_file}")

        return result

    def _generate_ad_for_segment(
        self,
        seg_frames: List[Path],
        s_idx: int,
        e_idx: int,
        start_sec: float,
        end_sec: float,
        avg_keep: float,
        output_time_sec: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        if len(seg_frames) < 2:
            return None

        clip_path = self.output_dir / f"qs_seg_{s_idx:05d}_{e_idx:05d}.mp4"
        duration = end_sec - start_sec
        if duration < 0.5:
            return None

        clip_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_img_list = self.output_dir / f".tmp_imglist_{s_idx}.txt"
        with tmp_img_list.open("w") as f:
            for fp in seg_frames:
                f.write(f"file '{fp.absolute()}'\n")

        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "concat", "-safe", "0", "-r", str(self.fps),
            "-i", str(tmp_img_list),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-vf", f"fps={self.fps}",
            str(clip_path),
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError:
            if tmp_img_list.exists():
                tmp_img_list.unlink()
            return None

        if tmp_img_list.exists():
            tmp_img_list.unlink()

        context_text = self._get_context_for_range(start_sec, end_sec, output_time_sec)

        face_avatars: List[Path] = []
        nearby_chars: List[str] = []
        if self._face_avatars_by_name and not self._is_script_free:
            nearby_chars = self._chars_near_time(start_sec, end_sec)
            for ch in nearby_chars:
                avs = self._face_avatars_by_name.get(ch.lower(), [])
                for a in avs:
                    if a not in face_avatars:
                        face_avatars.append(a)

        try:
            ad_text, inference_time, _ = self.engine.infer_one_segment(
                clip_path=clip_path,
                context_text=context_text,
                task_prompt=DEFAULT_QUERY,
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens,
                face_avatars=face_avatars if face_avatars else None,
                character_names=nearby_chars if nearby_chars else None,
            )
        except Exception as exc:
            print(f"    Inference ERROR: {exc}")
            ad_text = f"[ERROR: {exc}]"
            inference_time = 0.0

        if clip_path.exists():
            clip_path.unlink()

        return {
            "gap_id": s_idx,
            "scene_index": "QS",
            "location": "",
            "gap_start_time": f"{int(start_sec//3600):02d}:{int((start_sec%3600)//60):02d}:{start_sec%60:06.3f}",
            "gap_end_time": f"{int(end_sec//3600):02d}:{int((end_sec%3600)//60):02d}:{end_sec%60:06.3f}",
            "gap_start_sec": round(start_sec, 3),
            "gap_end_sec": round(end_sec, 3),
            "gap_duration_sec": round(duration, 1),
            "output_time_sec": round(output_time_sec, 3) if output_time_sec is not None else None,
            "characters": [],
            "context_before": [],
            "context_after": [],
            "ad_text": ad_text,
            "inference_time_sec": round(inference_time, 1),
            "qs_keep_rate": round(avg_keep, 4),
            "qs_frame_range": [s_idx, e_idx],
        }

    def _get_video_duration(self) -> float:
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(self.video_path),
        ]
        try:
            out = subprocess.check_output(cmd, text=True)
            return float(out.strip())
        except Exception:
            return 7200.0


def generate_with_querystream(
    engine,
    video_path: Path,
    xlsx_path,
    output_dir: Path,
    gpu_id: int = 0,
    tau_relevance: float = TAU_RELEVANCE,
    tau_density: float = TAU_DENSITY,
    tau_novelty: float = 0.1,
    ema_alpha: float = EMA_ALPHA,
    fps: float = FPS,
    temperature: float = 0.2,
    max_new_tokens: int = 256,
    vad_silence_db: float = -30.0,
    vad_min_silence_dur: float = 1.5,
) -> Dict[str, Any]:
    generator = QueryStreamADGenerator(
        engine=engine,
        video_path=video_path,
        xlsx_path=xlsx_path,
        output_dir=output_dir,
        gpu_id=gpu_id,
        tau_relevance=tau_relevance,
        tau_density=tau_density,
        tau_novelty=tau_novelty,
        ema_alpha=ema_alpha,
        fps=fps,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        vad_silence_db=vad_silence_db,
        vad_min_silence_dur=vad_min_silence_dur,
    )
    return generator.run()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="QueryStream AD generation")
    parser.add_argument("--video", required=True)
    parser.add_argument("--xlsx", default=None, help="Optional xlsx script. If omitted, VAD-based gap detection is used.")
    parser.add_argument("--output-dir", default="/tmp/qs_test")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--tau-relevance", type=float, default=TAU_RELEVANCE)
    parser.add_argument("--tau-density", type=float, default=TAU_DENSITY)
    parser.add_argument("--tau-novelty", type=float, default=0.1)
    parser.add_argument("--ema-alpha", type=float, default=EMA_ALPHA)
    parser.add_argument("--fps", type=float, default=FPS)
    parser.add_argument("--vad-silence-db", type=float, default=-30.0)
    parser.add_argument("--vad-min-silence-dur", type=float, default=1.5)
    args = parser.parse_args()

    from ad_engine import build_ad_engine
    engine = build_ad_engine(gpu_id=args.gpu)

    xlsx_path = Path(args.xlsx) if args.xlsx else None

    result = generate_with_querystream(
        engine=engine,
        video_path=Path(args.video),
        xlsx_path=xlsx_path,
        output_dir=Path(args.output_dir),
        gpu_id=args.gpu,
        tau_relevance=args.tau_relevance,
        tau_density=args.tau_density,
        tau_novelty=args.tau_novelty,
        ema_alpha=args.ema_alpha,
        fps=args.fps,
    )
    print(f"\nResult: {result['total_segments']} segments, avg keep_rate={result['avg_keep_rate']:.4f}")
