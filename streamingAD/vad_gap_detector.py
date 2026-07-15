#!/usr/bin/env python3
"""
vad_gap_detector.py — Script-free dialogue gap detection via audio analysis.

Uses ffmpeg's silencedetect filter to find silent intervals in the video's
audio track. No aligned script / xlsx needed — works on any raw video.

Also provides optional visual scene-boundary detection via ffmpeg's
scene-change filter, which helps identify narrative boundaries independently
of the audio.

Usage:
    from vad_gap_detector import detect_gaps_from_video

    gaps = detect_gaps_from_video(
        video_path=Path("movie.mp4"),
        min_gap_sec=4.0,
        silence_threshold_db=-30,
        min_silence_dur=1.5,
    )
"""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── ffmpeg silencedetect defaults ─────────────────────────────
DEFAULT_SILENCE_THRESHOLD_DB: float = -30.0
DEFAULT_MIN_SILENCE_DUR_SEC: float = 1.5
DEFAULT_MIN_GAP_SEC: float = 4.0
DEFAULT_MAX_GAP_SEC: float = 60.0
DEFAULT_SCENE_THRESHOLD: float = 0.4


@dataclass
class AudioGap:
    gap_id: int
    gap_start_sec: float
    gap_end_sec: float
    gap_duration_sec: float
    audio_before_end_sec: float
    audio_after_start_sec: float


@dataclass
class VideoSceneBoundary:
    timestamp_sec: float
    score: float


def _run_ffmpeg_silence_detect(
    video_path: Path,
    silence_threshold_db: float = DEFAULT_SILENCE_THRESHOLD_DB,
    min_silence_dur: float = DEFAULT_MIN_SILENCE_DUR_SEC,
    ffmpeg_bin: str = "ffmpeg",
) -> List[Dict[str, float]]:
    cmd = [
        ffmpeg_bin, "-hide_banner", "-y",
        "-i", str(video_path),
        "-af", f"silencedetect=n={silence_threshold_db}dB:d={min_silence_dur}",
        "-f", "null", "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    stderr = proc.stderr

    events: List[Dict[str, float]] = []
    pattern = re.compile(
        r"(silence_start|silence_end):\s*([\d]+\.?[\d]*)"
    )
    for m in pattern.finditer(stderr):
        evt_type = m.group(1)
        ts = float(m.group(2))
        events.append({"type": evt_type, "timestamp_sec": ts})

    return events


def _silence_events_to_gaps(
    events: List[Dict[str, float]],
    min_gap_sec: float = DEFAULT_MIN_GAP_SEC,
    max_gap_sec: float = DEFAULT_MAX_GAP_SEC,
) -> List[AudioGap]:
    gaps: List[AudioGap] = []
    gap_id = 0

    if not events:
        return gaps

    i = 0
    while i < len(events):
        if events[i]["type"] == "silence_start":
            gap_start = events[i]["timestamp_sec"]
            gap_end: Optional[float] = None
            for j in range(i + 1, len(events)):
                if events[j]["type"] == "silence_end":
                    gap_end = events[j]["timestamp_sec"]
                    i = j
                    break
            if gap_end is None:
                break

            duration = gap_end - gap_start
            if duration < min_gap_sec:
                i += 1
                continue
            if duration > max_gap_sec:
                i += 1
                continue

            gaps.append(AudioGap(
                gap_id=gap_id,
                gap_start_sec=gap_start,
                gap_end_sec=gap_end,
                gap_duration_sec=duration,
                audio_before_end_sec=gap_start,
                audio_after_start_sec=gap_end,
            ))
            gap_id += 1
        i += 1

    return gaps


def _detect_scene_boundaries(
    video_path: Path,
    scene_threshold: float = DEFAULT_SCENE_THRESHOLD,
    ffmpeg_bin: str = "ffmpeg",
) -> List[VideoSceneBoundary]:
    cmd = [
        ffmpeg_bin, "-hide_banner", "-y",
        "-i", str(video_path),
        "-vf", f"select='gt(scene\\,{scene_threshold})',metadata=print",
        "-f", "null", "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    stderr = proc.stderr

    boundaries: List[VideoSceneBoundary] = []
    pattern = re.compile(
        r"pts_time:([\d]+\.?[\d]*).*?lavfi\.scene_score=([\d]+\.?[\d]*)",
        re.DOTALL,
    )

    last_ts = -99.0
    for m in pattern.finditer(stderr):
        ts = float(m.group(1))
        score = float(m.group(2))
        if ts - last_ts > 2.0:
            boundaries.append(VideoSceneBoundary(timestamp_sec=ts, score=score))
            last_ts = ts

    return boundaries


def _merge_gaps_with_scenes(
    gaps: List[AudioGap],
    scene_boundaries: List[VideoSceneBoundary],
) -> List[AudioGap]:
    if not scene_boundaries:
        return gaps

    scene_ts = sorted(b.timestamp_sec for b in scene_boundaries)

    merged: List[AudioGap] = []
    gap_id = 0
    for gap in gaps:
        start = gap.gap_start_sec
        end = gap.gap_end_sec

        cuts_inside = [t for t in scene_ts if start < t < end]
        cuts_inside.sort()

        if not cuts_inside:
            merged.append(AudioGap(
                gap_id=gap_id,
                gap_start_sec=start,
                gap_end_sec=end,
                gap_duration_sec=end - start,
                audio_before_end_sec=gap.audio_before_end_sec,
                audio_after_start_sec=gap.audio_after_start_sec,
            ))
            gap_id += 1
        else:
            prev = start
            for cut in cuts_inside:
                dur = cut - prev
                if dur >= DEFAULT_MIN_GAP_SEC:
                    merged.append(AudioGap(
                        gap_id=gap_id,
                        gap_start_sec=prev,
                        gap_end_sec=cut,
                        gap_duration_sec=dur,
                        audio_before_end_sec=gap.audio_before_end_sec,
                        audio_after_start_sec=gap.audio_after_start_sec,
                    ))
                    gap_id += 1
                prev = cut
            dur = end - prev
            if dur >= DEFAULT_MIN_GAP_SEC:
                merged.append(AudioGap(
                    gap_id=gap_id,
                    gap_start_sec=prev,
                    gap_end_sec=end,
                    gap_duration_sec=dur,
                    audio_before_end_sec=gap.audio_before_end_sec,
                    audio_after_start_sec=gap.audio_after_start_sec,
                ))
                gap_id += 1

    return merged


def detect_gaps_from_video(
    video_path: Path,
    min_gap_sec: float = DEFAULT_MIN_GAP_SEC,
    max_gap_sec: float = DEFAULT_MAX_GAP_SEC,
    silence_threshold_db: float = DEFAULT_SILENCE_THRESHOLD_DB,
    min_silence_dur: float = DEFAULT_MIN_SILENCE_DUR_SEC,
    use_scene_detect: bool = True,
    scene_threshold: float = DEFAULT_SCENE_THRESHOLD,
    ffmpeg_bin: str = "ffmpeg",
) -> Tuple[List[AudioGap], List[VideoSceneBoundary]]:
    events = _run_ffmpeg_silence_detect(
        video_path=video_path,
        silence_threshold_db=silence_threshold_db,
        min_silence_dur=min_silence_dur,
        ffmpeg_bin=ffmpeg_bin,
    )

    gaps = _silence_events_to_gaps(
        events,
        min_gap_sec=min_gap_sec,
        max_gap_sec=max_gap_sec,
    )

    scene_boundaries: List[VideoSceneBoundary] = []
    if use_scene_detect:
        scene_boundaries = _detect_scene_boundaries(
            video_path=video_path,
            scene_threshold=scene_threshold,
            ffmpeg_bin=ffmpeg_bin,
        )
        gaps = _merge_gaps_with_scenes(gaps, scene_boundaries)

    return gaps, scene_boundaries


def extract_clip(
    video_path: Path,
    start_sec: float,
    end_sec: float,
    output_path: Path,
    ffmpeg_bin: str = "ffmpeg",
) -> bool:
    duration = end_sec - start_sec
    if duration <= 0:
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg_bin, "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{start_sec:.3f}",
        "-i", str(video_path),
        "-t", f"{duration:.3f}",
        "-map", "0:v:0?", "-map", "0:a:0?",
        "-c", "copy",
        "-avoid_negative_ts", "make_zero",
        str(output_path),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError:
        return False


def _prepare_fast_video(
    src_video: Path,
    cache_dir: Path,
    ffmpeg_bin: str = "ffmpeg",
) -> Path:
    fast_video = cache_dir / f"fast_{src_video.stem}.mp4"
    if fast_video.exists():
        return fast_video

    cmd = [
        ffmpeg_bin, "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(src_video),
        "-map", "0:v:0?", "-map", "0:a:0?",
        "-c", "copy",
        "-movflags", "+faststart",
        str(fast_video),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return fast_video


def _sec_to_timestamp(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"
