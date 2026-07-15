#!/usr/bin/env python3
"""
streaming_ad.py — Streaming AD Gradio interface.

- Input video path + optional script path
- Click "Preprocess & Load":
    - Aligned movies (25): load instantly
    - New movies with xlsx: run pipeline (parse xlsx → ffmpeg → face align)
    - New movies without xlsx: prompt to provide one
- Timeline slider → Video-LLaMA inference → AD output
"""

from __future__ import annotations

import argparse
import json as _json
import re
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Generator, Dict

import gradio as gr

import gradio.queueing as _grq
import asyncio as _asyncio

if hasattr(_grq, "PredictBody"):
    _orig_get_message = _grq.Queue.get_message

    async def _patched_get_message(self, event, timeout=5):
        try:
            data = await _asyncio.wait_for(event.websocket.receive_json(), timeout=timeout)
            data.setdefault("event_id", "")
            data.setdefault("event_data", None)
            try:
                return _grq.PredictBody(**data), True
            except Exception:
                import pydantic
                data.setdefault("batched", False)
                return _grq.PredictBody(**data), True
        except _asyncio.TimeoutError:
            await self.clean_event(event)
            return None, False
        except Exception as exc:
            print(f"[queue] get_message error: {exc}", flush=True)
            await self.clean_event(event)
            return None, False

    _grq.Queue.get_message = _patched_get_message

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STREAMING_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(STREAMING_ROOT))

_PREPROCESS_LOCK = threading.Lock()
_PREPROCESS_STATE: Dict[str, Any] = {}

from segment_db import (
    load_segment_db,
    extract_face_data,
    to_float,
    canonical,
    SegmentDB,
)
from context_builder import (
    build_prompt_context,
    build_task_prompt,
    AVAILABLE_MODULES,
    AVAILABLE_PLOT_SUBMODULES,
)
from ad_engine import build_ad_engine
from preprocess_pipeline import run_full_preprocess, DEFAULT_ALIGNED_JSON

AD_CLIPS_DIR = Path("/mnt/disk1new/ylz/newAD/Step04_RunTest/ad_clips_final")
FACE_JSON_ROOT = Path("/mnt/disk1new/ylz/newAD/Step04_RunTest/step04_03_face_align/json")
FINAL_BY_MOVIE_DIR = Path("/mnt/disk1new/ylz/newAD/Step04_RunTest/step04_final_by_movie_new")

EXAMPLE_INSTRUCTIONS = [
    "增加更多动作解读，描述肢体语言",
    "多描述一下主要人物的表情变化",
    "关注镜头运动和场景切换",
    "用更简洁的语言描述，不要超过一句话",
    "加入环境氛围的描述",
    "重点关注人物之间的互动",
]


def parse_clip_stem(ad_id: str, clip_index: Any, ad_order: int) -> str:
    match = re.search(r"__clip(\d+)__ad(\d+)$", str(ad_id or ""))
    if match:
        return f"clip{match.group(1).zfill(4)}_ad{match.group(2).zfill(4)}"
    return f"clip{int(to_float(clip_index, 0)):04d}_ad{ad_order:04d}"


def format_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def _resolve_dir(base: Path, movie_title: str) -> Path | None:
    if base.is_dir():
        return base
    for p in base.parent.iterdir():
        if p.is_dir() and canonical(movie_title) in canonical(p.name):
            return p
    return None


def _extract_title_from_path(video_path: str) -> str:
    p = Path(video_path)
    t = p.stem
    t = re.sub(r"^IMDB-\d+-", "", t)
    t = re.sub(r"\.[^.]+$", "", t)
    return t.strip()


def _lookup_aligned_title(title: str) -> str:
    aligned_path = Path(DEFAULT_ALIGNED_JSON)
    if not aligned_path.is_file():
        return ""
    with aligned_path.open() as f:
        items = _json.load(f).get("items", [])
    all_titles = set(i.get("movie_title", "").strip() for i in items if isinstance(i, dict))
    ct = canonical(title)
    for at in all_titles:
        if canonical(at) == ct:
            return at
    return ""


def _sanitize_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_ " else "_" for ch in str(name or "")).strip()


def _check_movie_ready(movie_title: str) -> dict:
    seg_prefix = _sanitize_name(movie_title).replace(" ", "_")
    seg_json = FINAL_BY_MOVIE_DIR / f"{seg_prefix}.json"
    has_seg = seg_json.is_file()

    seg_count = 0
    if has_seg:
        import json as _j
        with seg_json.open() as _f:
            seg_count = len(_j.load(_f).get("ad_segments", []))

    clips_dir = _resolve_dir(AD_CLIPS_DIR / movie_title, movie_title)
    has_clips = False
    clip_count = 0
    if clips_dir is not None:
        clip_count = len(list(clips_dir.glob("*.mp4")))
        has_clips = clip_count > 0

    # Only consider truly ready if clips match segment count
    all_ready = has_seg and has_clips and clip_count >= seg_count

    face_dir = _resolve_dir(FACE_JSON_ROOT / movie_title, movie_title)
    has_face = face_dir is not None and any(face_dir.glob("*.json"))

    return {
        "has_seg": has_seg,
        "has_clips": has_clips,
        "has_face": has_face,
        "all_ready": all_ready,
        "seg_json": str(seg_json) if has_seg else None,
        "clips_dir": str(clips_dir) if has_clips else None,
        "face_dir": str(face_dir) if has_face else None,
        "seg_count": seg_count,
        "clip_count": clip_count,
    }


def _build_movie_state(movie_title: str) -> dict:
    seg_db = load_segment_db(movie_title, final_by_movie_dir=FINAL_BY_MOVIE_DIR)
    clips_dir = _resolve_dir(AD_CLIPS_DIR / movie_title, movie_title)
    face_dir = _resolve_dir(FACE_JSON_ROOT / movie_title, movie_title)
    total_dur = seg_db.total_duration
    if seg_db.segments:
        min_sec = min(to_float(s.get("ad_movie_start_sec"), 0.0) for s in seg_db.segments)
        max_sec = max(to_float(s.get("ad_movie_end_sec"), 0.0) for s in seg_db.segments)
    else:
        min_sec = 0.0
        max_sec = total_dur
    return {
        "movie_title": movie_title,
        "seg_db": seg_db,
        "clips_dir": str(clips_dir) if clips_dir else None,
        "face_dir": str(face_dir) if face_dir else None,
        "total_dur": total_dur,
        "min_sec": min_sec,
        "max_sec": max_sec,
        "seg_count": len(seg_db.segments),
    }


def _ui_dict_to_updates(st: dict):
    hdr = (
        f"<h2 style='text-align:center'>Streaming Audio Description</h2>"
        f"<p style='text-align:center;color:#888'>"
        f"Movie: <b>{st['movie_title']}</b> | "
        f"AD: {format_time(st['min_sec'])} → {format_time(st['max_sec'])} | "
        f"Model: Video-LLaMA | {st['seg_count']} segments | "
        f"{'Face: ✓' if st['face_dir'] else 'Face: N/A'}"
        f"</p>"
    )
    slider_upd = gr.update(
        minimum=st["min_sec"], maximum=st["total_dur"], value=st["min_sec"], interactive=True
    )
    return hdr, format_time(st["min_sec"]), slider_upd


def create_gr_app(args: argparse.Namespace) -> gr.Blocks:
    gpu_id = int(args.gpu_id)

    print(f"[init] Loading Video-LLaMA model on GPU {gpu_id} ...")
    engine = build_ad_engine(gpu_id=gpu_id)
    print("[init] Model ready.")

    def on_video_path_change(video_path: str):
        vp = (video_path or "").strip()
        if not vp:
            return "", ""
        raw_title = _extract_title_from_path(vp)
        matched = _lookup_aligned_title(raw_title)
        if matched:
            return matched, f"✓ Aligned data found — no script needed"
        return raw_title, f"⚠ New movie — provide xlsx script path below"

    def process_and_load(video_path: str, movie_title_input: str, xlsx_path: str):
        global _PREPROCESS_STATE
        vp = (video_path or "").strip().strip('"\'')
        t = (movie_title_input or "").strip().strip('"\'')
        xp = (xlsx_path or "").strip().strip('"\'')

        if not t and vp:
            t = _extract_title_from_path(vp)

        if not t:
            return {}, "", "00:00", gr.update(), "Enter a video path and movie title."

        matched = _lookup_aligned_title(t)
        is_new = not matched
        if is_new and not xp:
            return {}, "", "00:00", gr.update(), gr.update(
                value=f"'{t}' has no aligned data. Provide an Excel script path.\n"
                      f"Example: /mnt/disk1new/storyvideo/Alignedscript/Movie/IMDB-211-Harry Potter 1.xlsx"
            )

        effective_title = matched if matched else t
        print(f"[process] title={effective_title}, video={vp or 'N/A'}, xlsx={xp or 'N/A'}, aligned={not is_new}")

        ready = _check_movie_ready(effective_title)
        if ready["all_ready"]:
            st = _build_movie_state(effective_title)
            hdr, ts, slider_upd = _ui_dict_to_updates(st)
            return st, hdr, ts, slider_upd, (
                f"Already preprocessed. Loaded {st['seg_count']} segments from {ready['clips_dir']}"
            )

        # Guard: prevent duplicate preprocessing
        with _PREPROCESS_LOCK:
            if _PREPROCESS_STATE.get("active"):
                return {}, "", "00:00", gr.update(), \
                    _PREPROCESS_STATE.get("summary", "") + "\n\n⛔ Already preprocessing. Please wait."

        status_summary = (
            f"**Preprocessing: {effective_title}**\n\n"
            f"Segment → {'✅' if ready['has_seg'] else '⬜'} ({ready.get('seg_count', 0)} segs)\n"
            f"Clips   → {'✅' if ready['has_clips'] else '⬜'} ({ready.get('clip_count', 0)}/{ready.get('seg_count', 0)})\n"
            f"Face    → {'✅' if ready['has_face'] else '⬜'}"
        )

        with _PREPROCESS_LOCK:
            _PREPROCESS_STATE = {
                "active": True,
                "movie": effective_title,
                "stage": "starting",
                "pct": 0.0,
                "msg": "Starting ...",
                "done": False,
                "result": None,
                "error": None,
                "summary": status_summary,
            }

        def _cb(stage, pct, msg):
            with _PREPROCESS_LOCK:
                _PREPROCESS_STATE["stage"] = stage
                _PREPROCESS_STATE["pct"] = pct
                _PREPROCESS_STATE["msg"] = msg
            print(f"[preprocess] {stage} {pct*100:.0f}%: {msg}")

        def _bg():
            global _PREPROCESS_STATE
            try:
                result = run_full_preprocess(
                    movie_title=effective_title, progress=_cb, gpu_id=gpu_id,
                    force=False, video_path=vp, xlsx_path=xp,
                )
                with _PREPROCESS_LOCK:
                    _PREPROCESS_STATE["done"] = True
                    _PREPROCESS_STATE["result"] = result
                    _PREPROCESS_STATE["active"] = False
            except Exception as exc:
                traceback.print_exc()
                with _PREPROCESS_LOCK:
                    _PREPROCESS_STATE["done"] = True
                    _PREPROCESS_STATE["error"] = str(exc)
                    _PREPROCESS_STATE["active"] = False

        threading.Thread(target=_bg, daemon=True).start()
        return {}, "", "00:00", gr.update(), status_summary + "\n\n⏳ Processing started — progress updates every 2s"

    def _poll_progress():
        global _PREPROCESS_STATE
        with _PREPROCESS_LOCK:
            st = dict(_PREPROCESS_STATE)

        if not st:
            return gr.update(), gr.update(), gr.update(), gr.update(), gr.update()

        if not st.get("active"):
            if st.get("done") and st.get("result"):
                r = st["result"]
                if r.get("status") == "ready":
                    movie = _build_movie_state(st["movie"])
                    hdr, ts, _slider = _ui_dict_to_updates(movie)
                    trigger_val = int(time.monotonic() * 1000)
                    with _PREPROCESS_LOCK:
                        _PREPROCESS_STATE.clear()
                    return movie, hdr, ts, (
                        f"✅ Done! {movie['seg_count']} segments ready.\n"
                        f"{format_time(movie['min_sec'])} → {format_time(movie['max_sec'])}"
                    ), trigger_val
                else:
                    msgs = "\n".join(r.get("messages", []))
                    return gr.update(), gr.update(), gr.update(), st.get("summary", "") + f"\n\n❌ Failed:\n{msgs}", gr.update()
            if st.get("error"):
                return gr.update(), gr.update(), gr.update(), st.get("summary", "") + f"\n\n❌ Failed: {st['error']}", gr.update()
            return gr.update(), gr.update(), gr.update(), gr.update(), gr.update()

        stage_label = {
            "step04_01": "① Parse Excel",
            "step04_02": "② Split Clips",
            "step04_03": "③ Face Align"
        }.get(st["stage"], st["stage"])
        pct_str = f"{int(st['pct']*100):d}%"
        msg = (st.get("msg", "") or "").split("\n")[0][:120]
        line = f"\n⏳ [{stage_label}] {pct_str} — {msg}"
        return gr.update(), gr.update(), gr.update(), st.get("summary", "") + line, gr.update()

    def _on_slider_ready(st):
        if not st or not st.get("clips_dir"):
            return gr.update(), gr.update(), gr.update()
        hdr, ts, slider_upd = _ui_dict_to_updates(st)
        return hdr, ts, slider_upd

    def generate_ad(time_sec, instruction, temperature, movie_state, chat_history):
        try:
            st = movie_state
            if not isinstance(st, dict):
                chat_history.append(["Error", f"Movie state is invalid: {type(st).__name__}"])
                return chat_history, movie_state
            if not st or not st.get("clips_dir"):
                chat_history.append(["Error", "No movie loaded. Preprocess & Load first."])
                return chat_history, movie_state

            seg_db = st["seg_db"]
            clips_dir = Path(st["clips_dir"])
            face_dir = Path(st["face_dir"]) if st.get("face_dir") else None

            start_total = time.monotonic()
            t_sec = float(time_sec)
            cur_seg = seg_db.current_segment(t_sec)
            if cur_seg is None:
                chat_history.append([f"t={format_time(t_sec)}", "No AD segment near this time."])
                return chat_history, movie_state

            ad_id = str(cur_seg.get("ad_id", "")).strip()
            seg_idx = seg_db.segments.index(cur_seg)
            clip_stem = parse_clip_stem(ad_id=ad_id, clip_index=cur_seg.get("clip_index"), ad_order=seg_idx + 1)
            clip_path = clips_dir / f"{clip_stem}.mp4"

            face_json = face_dir / f"{clip_stem}.json" if face_dir else None
            face_matches, _ = extract_face_data(face_json, max_face_records=4)

            context_text, _ = build_prompt_context(
                segment=cur_seg,
                modules=AVAILABLE_MODULES,
                plot_submodules=AVAILABLE_PLOT_SUBMODULES,
                face_matches=face_matches,
                max_description_lines=5,
                max_dialog_lines=8,
            )
            task_prompt = build_task_prompt(custom_instruction=str(instruction or "").strip())

            seg_start = to_float(cur_seg.get("ad_movie_start_sec"), 0.0)
            seg_end = to_float(cur_seg.get("ad_movie_end_sec"), 0.0)
            ref_ad = str(cur_seg.get("cmdqa", {}).get("text", "")).strip()
            label = str(instruction or "").strip() or "default"
            user_msg = (
                f"Time: {format_time(seg_start)} → {format_time(seg_end)}\n"
                f"Seg {seg_idx+1}/{len(seg_db.segments)} | {label}"
            )

            if not clip_path.is_file():
                chat_history.append([user_msg, f"Clip not found: {clip_path}"])
                return chat_history, movie_state

            print(f"[generate_ad] clip={clip_stem}, temp={temperature}", flush=True)

            ad_text, elapsed_model, raw_prompt = engine.infer_one_segment(
                clip_path=clip_path,
                context_text=context_text,
                task_prompt=task_prompt,
                temperature=float(temperature),
            )

            total_elapsed = time.monotonic() - start_total
            fp = raw_prompt.replace("<ImageHere>", "[IMG]").replace("<VideoHere>", "[VID]")
            print(f"\n{'='*60}\nFULL PROMPT ({len(fp)} chars):\n{fp}\nGENERATED: {ad_text}\n{'='*60}\n")

            lines = [f"**Time**: {total_elapsed:.1f}s (inference {elapsed_model:.1f}s)"]
            if ref_ad:
                lines.append(f"**Reference AD**: {ref_ad}")
            lines.append(f"**Generated AD**: {ad_text}")
            chat_history.append([user_msg, "\n".join(lines)])
            return chat_history, movie_state
        except Exception as exc:
            traceback.print_exc()
            chat_history.append(["Error", f"{type(exc).__name__}: {exc}"])
            return chat_history, movie_state

    def clear_chat():
        return []

    # ---- UI ----
    demo = gr.Blocks(
        theme=gr.themes.Soft(primary_hue="orange", radius_size="lg"),
        css="""
        #ad-submit-btn {
            background: linear-gradient(135deg, #FF7C00 0%, #E66A00 100%) !important;
            border: none !important; color: white !important;
            height: 56px !important; font-size: 18px !important;
            font-weight: bold !important; border-radius: 12px !important;
        }
        #ad-submit-btn:hover { opacity: 0.9; }
        #instruction-box textarea { font-size: 15px !important; line-height: 1.6 !important; }
        #chatbot-area { height: 480px; }
        #process-btn { height: 44px !important; }
        """,
        title="Streaming AD",
    )

    with demo:
        movie_state = gr.State({})
        header_md = gr.Markdown(
            "<h2 style='text-align:center'>Streaming Audio Description</h2>"
            "<p style='text-align:center;color:#888'>Enter paths and click Preprocess &amp; Load</p>"
        )

        gr.Markdown("### Video Source")
        with gr.Row():
            video_path_input = gr.Textbox(
                label="Video File Path",
                placeholder="/mnt/disk1new/storyvideo/Movie/IMDB-211-Harry Potter 1.mkv",
            )
            movie_title_input = gr.Textbox(
                label="Movie Title (auto-detected)",
                placeholder="Harry Potter 1",
            )
        with gr.Row():
            xlsx_path_input = gr.Textbox(
                label="Script Path (xlsx, only for new movies — leave empty for aligned movies)",
                placeholder="/mnt/disk1new/storyvideo/Alignedscript/Movie/IMDB-211-Harry Potter 1.xlsx",
            )
            script_hint = gr.Textbox(
                label="", value="", interactive=False, lines=1,
            )
            process_btn = gr.Button("Preprocess & Load", elem_id="process-btn", variant="secondary")

        video_path_input.change(
            fn=on_video_path_change,
            inputs=[video_path_input],
            outputs=[movie_title_input, script_hint],
        )

        status_text = gr.Markdown(value="Waiting for input...")

        with gr.Row():
            with gr.Column(scale=5):
                gr.Markdown("### Timeline")
                time_slider = gr.Slider(
                    minimum=0.0, maximum=100.0, value=0.0, step=1.0,
                    label="Position (seconds)", interactive=True,
                )
                time_display = gr.Textbox(value="00:00", label="Current Time", interactive=False)
                time_slider.change(
                    fn=lambda v: format_time(float(v)),
                    inputs=[time_slider], outputs=[time_display],
                )

                gr.Markdown("### Quick Jump")
                with gr.Row():
                    for label, fn in [
                        ("-10s", lambda v: max(0, float(v) - 10)),
                        ("+10s", lambda v: float(v) + 10),
                        ("Start", lambda v: 0),
                    ]:
                        gr.Button(label).click(fn=fn, inputs=[time_slider], outputs=[time_slider])

            with gr.Column(scale=5):
                chatbot = gr.Chatbot(label="AD Generation Log", elem_id="chatbot-area", height=480)

        with gr.Row():
            instruction_input = gr.Textbox(
                label="Custom Instruction (leave empty for default)",
                placeholder="e.g. Focus more on character expressions...",
                elem_id="instruction-box", lines=2,
            )

        gr.Markdown("### Quick Templates")
        with gr.Row():
            for ex in EXAMPLE_INSTRUCTIONS:
                gr.Button(ex).click(fn=lambda t=ex: t, outputs=[instruction_input])

        with gr.Row():
            gr.Button("Clear Log").click(fn=clear_chat, outputs=[chatbot])
            temperature_slider = gr.Slider(0.1, 1.5, value=0.2, step=0.05, label="Temperature")
            gr.Button("Generate AD", elem_id="ad-submit-btn", variant="primary").click(
                fn=generate_ad,
                inputs=[time_slider, instruction_input, temperature_slider, movie_state, chatbot],
                outputs=[chatbot, movie_state],
            )

        slider_ready_trigger = gr.Number(value=0, visible=False)

        process_btn.click(
            fn=process_and_load,
            inputs=[video_path_input, movie_title_input, xlsx_path_input],
            outputs=[movie_state, header_md, time_display, time_slider, status_text],
        )

        slider_ready_trigger.change(
            fn=_on_slider_ready,
            inputs=[movie_state],
            outputs=[header_md, time_display, time_slider],
        )

        demo.load(
            fn=_poll_progress,
            outputs=[movie_state, header_md, time_display, status_text, slider_ready_trigger],
            every=2,
        )

    return demo


def parse_args():
    p = argparse.ArgumentParser(description="Streaming AD Generator")
    p.add_argument("--gpu-id", type=int, default=0)
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--share", action="store_true")
    p.add_argument("--server-name", type=str, default="0.0.0.0")
    p.add_argument("--movie-title", type=str, default="")
    p.add_argument("--movie-path", type=str, default="")
    return p.parse_args()


def main():
    a = parse_args()
    demo = create_gr_app(a)
    demo.queue(max_size=4).launch(server_name=a.server_name, server_port=a.port, share=a.share)


if __name__ == "__main__":
    main()
