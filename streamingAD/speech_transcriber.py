#!/usr/bin/env python3
"""
speech_transcriber.py — Whisper-based speech transcription + VAD speaker segmentation.

Transcribes the audio track of a movie using OpenAI Whisper. Combined with
VAD silence detection, it segments speech into speaker turns without needing
any aligned script.

No speaker diarization model (pyannote) is needed. Instead, we:
  1. Run VAD (ffmpeg silencedetect) to find all speech intervals
  2. Run Whisper to transcribe the full audio with word-level timestamps
  3. Map Whisper segments onto VAD speech intervals → "speaker turns"
  4. Each speaker turn is labeled as SPEAKER-0, SPEAKER-1, ... sequentially

This gives the AD model access to "what is being said around the gap"
without knowing character names.

Model files are downloaded to /mnt/disk1new/ylz/newAD/models/whisper/.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WHISPER_MODEL_DIR = PROJECT_ROOT / "models" / "whisper"
DEFAULT_WHISPER_MODEL = "turbo"
DEFAULT_WHISPER_DEVICE = "cuda"

WHISPER_MODEL_SIZES = [
    "tiny", "tiny.en", "base", "base.en", "small", "small.en",
    "medium", "medium.en", "large-v1", "large-v2", "large-v3",
    "large", "turbo",
]


@dataclass
class SpeakerTurn:
    turn_id: int
    speaker_label: str
    start_sec: float
    end_sec: float
    text: str
    is_dialogue: bool = True


@dataclass
class TranscriptionResult:
    turns: List[SpeakerTurn]
    full_text: str
    model_name: str
    model_size: str
    language: str
    duration_sec: float
    process_time_sec: float


def _load_whisper_model(
    model_size: str = DEFAULT_WHISPER_MODEL,
    model_dir: Optional[Path] = None,
    device: str = DEFAULT_WHISPER_DEVICE,
):
    import whisper

    if model_dir is None:
        model_dir = DEFAULT_WHISPER_MODEL_DIR
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("WHISPER_CACHE_DIR", str(model_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(model_dir.parent))

    print(f"[Whisper] Loading model '{model_size}' to {device} ...")
    t0 = time.monotonic()
    model = whisper.load_model(model_size, device=device, download_root=str(model_dir))
    elapsed = time.monotonic() - t0
    print(f"[Whisper] Model loaded in {elapsed:.1f}s")
    return model


def _extract_audio(
    video_path: Path,
    output_path: Path,
    start_sec: Optional[float] = None,
    end_sec: Optional[float] = None,
    sample_rate: int = 16000,
    ffmpeg_bin: str = "ffmpeg",
) -> Path:
    cmd = [
        ffmpeg_bin, "-hide_banner", "-loglevel", "error", "-y",
    ]
    if start_sec is not None:
        cmd += ["-ss", f"{start_sec:.3f}"]
    cmd += ["-i", str(video_path)]
    if end_sec is not None and start_sec is not None:
        cmd += ["-t", f"{end_sec - start_sec:.3f}"]
    cmd += [
        "-vn",
        "-ac", "1",
        "-ar", str(sample_rate),
        "-sample_fmt", "s16",
        str(output_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return output_path


def transcribe_audio(
    audio_path: Path,
    model,
    language: Optional[str] = None,
    word_timestamps: bool = True,
) -> Dict[str, Any]:
    options = {
        "word_timestamps": word_timestamps,
        "verbose": False,
    }
    if language:
        options["language"] = language

    result = model.transcribe(str(audio_path), **options)
    return result


def _get_speech_intervals_from_vad(
    video_path: Path,
    silence_threshold_db: float = -30.0,
    min_silence_dur: float = 1.0,
    ffmpeg_bin: str = "ffmpeg",
) -> List[Tuple[float, float]]:
    from vad_gap_detector import _run_ffmpeg_silence_detect

    events = _run_ffmpeg_silence_detect(
        video_path=video_path,
        silence_threshold_db=silence_threshold_db,
        min_silence_dur=min_silence_dur,
        ffmpeg_bin=ffmpeg_bin,
    )

    speech_intervals: List[Tuple[float, float]] = []
    prev_end = 0.0

    silence_starts: List[float] = []
    silence_ends: List[float] = []
    i = 0
    while i < len(events):
        if events[i]["type"] == "silence_start":
            ss = events[i]["timestamp_sec"]
            if i + 1 < len(events) and events[i + 1]["type"] == "silence_end":
                se = events[i + 1]["timestamp_sec"]
                silence_starts.append(ss)
                silence_ends.append(se)
                i += 2
            else:
                i += 1
        else:
            i += 1

    if not silence_starts:
        return [(0.0, 999999.0)]

    if silence_starts[0] > 0.3:
        speech_intervals.append((0.0, silence_starts[0]))

    for j in range(len(silence_starts) - 1):
        speech_start = silence_ends[j]
        speech_end = silence_starts[j + 1]
        if speech_end - speech_start > 0.3:
            speech_intervals.append((speech_start, speech_end))

    if silence_ends:
        speech_intervals.append((silence_ends[-1], 999999.0))

    return speech_intervals


def _events_to_speech_intervals(
    events: List[Dict[str, float]],
) -> List[Tuple[float, float]]:
    starts: List[float] = []
    ends: List[float] = []
    for e in events:
        if e["type"] == "silence_start":
            starts.append(e["timestamp_sec"])
        else:
            ends.append(e["timestamp_sec"])
    starts.sort()
    ends.sort()

    intervals: List[Tuple[float, float]] = []
    prev = 0.0
    for si in range(min(len(starts), len(ends))):
        ss = starts[si]
        se = ends[si]
        if ss > prev + 0.3:
            intervals.append((prev, ss))
        prev = se
    if prev < 999999:
        intervals.append((prev, 999999.0))
    return intervals


def _map_segments_to_speech_intervals(
    whisper_segments: List[Dict[str, Any]],
    speech_intervals: List[Tuple[float, float]],
) -> List[SpeakerTurn]:
    turns: List[SpeakerTurn] = []
    turn_id = 0

    for si_idx, (si_start, si_end) in enumerate(speech_intervals):
        seg_texts: List[Tuple[float, str]] = []

        for seg in whisper_segments:
            seg_start = float(seg.get("start", 0))
            seg_end = float(seg.get("end", 0))
            seg_text = str(seg.get("text", "")).strip()

            if not seg_text:
                continue

            if seg_start >= si_start and seg_end <= si_end:
                seg_texts.append((seg_start, seg_text))
            elif seg_start >= si_start and seg_start < si_end:
                seg_texts.append((seg_start, seg_text))
            elif seg_end > si_start and seg_end <= si_end:
                seg_texts.append((seg_start, seg_text))

        seg_texts.sort(key=lambda x: x[0])
        combined = " ".join(t for _, t in seg_texts).strip()

        if combined:
            speaker_label = f"SPEAKER-{turn_id % 8}"
            turns.append(SpeakerTurn(
                turn_id=turn_id,
                speaker_label=speaker_label,
                start_sec=si_start,
                end_sec=si_end,
                text=combined,
                is_dialogue=True,
            ))
            turn_id += 1

    return turns


def transcribe_video(
    video_path: Path,
    model_size: str = DEFAULT_WHISPER_MODEL,
    model_dir: Optional[Path] = None,
    device: str = DEFAULT_WHISPER_DEVICE,
    language: Optional[str] = None,
    silence_threshold_db: float = -30.0,
    min_silence_dur: float = 1.0,
    max_duration_sec: Optional[float] = None,
    tmp_dir: Optional[Path] = None,
    silence_events: Optional[List[Dict[str, float]]] = None,
) -> TranscriptionResult:
    if model_dir is None:
        model_dir = DEFAULT_WHISPER_MODEL_DIR
    model_dir = Path(model_dir)

    if tmp_dir is None:
        tmp_dir = Path("/mnt/disk1new/ylz/newAD/.tmp_whisper")
    tmp_dir = Path(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    t_total_start = time.monotonic()

    print(f"[Whisper] Processing: {video_path.name}")

    model = _load_whisper_model(
        model_size=model_size,
        model_dir=model_dir,
        device=device,
    )

    audio_path = tmp_dir / f"{video_path.stem}_audio.wav"
    print(f"[Whisper] Extracting audio to {audio_path} ...")
    t0 = time.monotonic()
    _extract_audio(video_path, audio_path, start_sec=0.0, end_sec=max_duration_sec)
    audio_extract_time = time.monotonic() - t0
    print(f"[Whisper] Audio extracted in {audio_extract_time:.1f}s")

    total_duration = float(
        subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(audio_path)],
            text=True,
        ).strip()
    )

    print(f"[Whisper] Transcribing {total_duration:.0f}s of audio ...")
    t0 = time.monotonic()
    result = transcribe_audio(audio_path, model=model, language=language)
    transcribe_time = time.monotonic() - t0
    print(f"[Whisper] Transcription done in {transcribe_time:.1f}s")

    detected_lang = result.get("language", "unknown")
    segments = result.get("segments", [])

    print(f"[Whisper] Language: {detected_lang}, {len(segments)} segments")

    if silence_events is not None:
        print(f"[VAD] Using pre-computed {len(silence_events)} silence events")
        speech_intervals = _events_to_speech_intervals(silence_events)
        vad_time = 0.0
    else:
        print(f"[VAD] Detecting speech intervals ...")
        t0 = time.monotonic()
        speech_intervals = _get_speech_intervals_from_vad(
            video_path=video_path,
            silence_threshold_db=silence_threshold_db,
            min_silence_dur=min_silence_dur,
        )
        vad_time = time.monotonic() - t0
        print(f"[VAD] Found {len(speech_intervals)} speech intervals in {vad_time:.1f}s")

    turns = _map_segments_to_speech_intervals(segments, speech_intervals)
    print(f"[Speaker] Mapped to {len(turns)} speaker turns")

    full_text = " ".join(
        seg.get("text", "").strip() for seg in segments
    )

    if audio_path.exists():
        audio_path.unlink()

    total_time = time.monotonic() - t_total_start

    return TranscriptionResult(
        turns=turns,
        full_text=full_text,
        model_name="openai/whisper",
        model_size=model_size,
        language=detected_lang,
        duration_sec=total_duration,
        process_time_sec=total_time,
    )


def turns_near_timestamp(
    turns: List[SpeakerTurn],
    timestamp_sec: float,
    window_before_sec: float = 30.0,
    window_after_sec: float = 15.0,
    max_turns: int = 10,
) -> List[SpeakerTurn]:
    nearby: List[SpeakerTurn] = []
    for t in turns:
        if t.end_sec <= timestamp_sec and t.end_sec >= timestamp_sec - window_before_sec:
            nearby.append(t)
        elif t.start_sec >= timestamp_sec and t.start_sec <= timestamp_sec + window_after_sec:
            nearby.append(t)

    nearby.sort(key=lambda t: t.start_sec)

    before = [t for t in nearby if t.end_sec <= timestamp_sec]
    after = [t for t in nearby if t.start_sec >= timestamp_sec]

    before = before[-max_turns:]
    after = after[:max_turns]

    result: List[SpeakerTurn] = []
    seen: set = set()
    for t in before + after:
        if t.turn_id not in seen:
            result.append(t)
            seen.add(t.turn_id)
    result.sort(key=lambda t: t.start_sec)
    return result


def turns_to_context_text(
    turns: List[SpeakerTurn],
    gap_start_sec: float,
    gap_end_sec: float,
) -> str:
    parts: List[str] = []
    parts.append("[Scene context from speech transcription]")

    before_turns = [t for t in turns if t.end_sec <= gap_start_sec]
    after_turns = [t for t in turns if t.start_sec >= gap_end_sec]

    if before_turns:
        parts.append("\n[What was being said before the gap]")
        for t in before_turns[-8:]:
            parts.append(f"{t.speaker_label}: \"{t.text}\"")

    if after_turns:
        parts.append("\n[What is being said after the gap]")
        for t in after_turns[:5]:
            parts.append(f"{t.speaker_label}: \"{t.text}\"")

    if not before_turns and not after_turns:
        parts.append("\n(No transcribed dialogue near this gap)")

    parts.append(
        f"\n[Gap description]\n"
        f"This is a silent pause between conversations "
        f"({gap_end_sec - gap_start_sec:.1f}s). "
        f"Describe what is VISIBLE on screen during this gap."
    )

    return "\n".join(parts)
