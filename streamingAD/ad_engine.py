#!/usr/bin/env python3
"""
AD Engine — Video-LLaMA model loading + single-segment inference.

Identical to newAD step04_04_imagehere.py approach:
- Same Config / registry / Chat initialization
- Same conversation flow: context → upload images → upload video → ask task → answer
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import random
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VIDEO_LLAMA_ROOT = PROJECT_ROOT / "Video-LLaMA"
MODELS_ROOT = PROJECT_ROOT / "models"

if str(VIDEO_LLAMA_ROOT) not in sys.path:
    sys.path.insert(0, str(VIDEO_LLAMA_ROOT))

from video_llama.common.config import Config
from video_llama.common.registry import registry
from video_llama.conversation.conversation_video import Chat, default_conversation

DEFAULT_CFG_PATH = VIDEO_LLAMA_ROOT / "eval_configs" / "video_llama_eval_only_vl.yaml"
DEFAULT_LLAMA_DIR = MODELS_ROOT / "llama-2-7b-chat-hf"
DEFAULT_CKPT_FILE = MODELS_ROOT / "ad3_moviellama2_ce_iter14000.pth.tar"


@dataclass
class ADEngine:
    chat: Chat
    device: str

    def select_best_candidate(
        self,
        candidates: List[str],
        selector=None,
        clip_path: Path = None,
    ) -> str:
        """
        Use an LLM to select the best AD candidate.
        - Shuffles candidate order to avoid positional bias
        - Does NOT reveal how candidates were generated (beam vs sample)
        - If selector (Qwen model) is provided, use it; otherwise use Llama
        """
        if not candidates:
            return ""
        if len(candidates) == 1:
            return candidates[0]

        # Shuffle to avoid positional bias
        indexed = list(enumerate(candidates))
        random.shuffle(indexed)
        shuffled_candidates = [c for _, c in indexed]

        # System prompt — defines what makes a good AD
        system_prompt = (
            "You will be provided with several audio descriptions of a video clip.\n"
            "Your task is to choose the one that is most helpful to the blind.\n"
            "Choose something concise and that blind people might be curious about.\n"
            "A good audio description:\n"
            "- Describes specific visible actions (e.g. 'walks', 'kisses', 'opens')\n"
            "- Uses character names if available\n"
            "- Is one short sentence (under 20 words)\n"
            "- Does NOT describe what cannot be seen (thoughts, feelings, sounds)\n"
            "- Does NOT just list who is present without describing actions\n"
            "A bad audio description:\n"
            "- Is too vague (e.g. 'Two people are in a room')\n"
            "- Only describes static poses (e.g. 'sits on a couch')\n"
            "- Misses the main action happening in the clip\n"
            "\n"
            "Reply with ONLY the number of the best description."
        )

        # Few-shot examples
        few_shot_pairs = [
            {
                "candidates": (
                    "Dix Handley lies on a couch, his eyes closed.\n"
                    "Dix Handley lies on a couch.\n"
                    "The woman groggily opens her eyes and sees him standing over her."
                ),
                "answer": "3"
            },
            {
                "candidates": (
                    "Dix Handley sits on a couch.\n"
                    "The blonde saunters over to him and leans in for a kiss.\n"
                    "Doll Conovan sits on a couch, looking at Dix."
                ),
                "answer": "2"
            },
            {
                "candidates": (
                    "Dix Handley opens a safe deposit box.\n"
                    "He opens the safe door and pulls out a stack of cash.\n"
                    "Dix Handley and Alonzo Emmerich are in a room."
                ),
                "answer": "2"
            },
            {
                "candidates": (
                    "Dix Handley looks at Alonzo D.\n"
                    "Dix takes a drag on his cigarette.\n"
                    "Dix Handley and Alonzo Emmerich stand in a room."
                ),
                "answer": "2"
            },
        ]

        # Format shuffled candidates
        cand_text = "\n".join(f"{i+1}. {c}" for i, c in enumerate(shuffled_candidates))

        # Build full prompt
        few_shot_text = ""
        for ex in few_shot_pairs:
            few_shot_text += f"\n{ex['candidates']}\nAnswer: {ex['answer']}\n"

        full_prompt = (
            f"{system_prompt}\n"
            f"\nExamples:\n{few_shot_text}"
            f"\nNow choose the best:\n{cand_text}\n"
            f"Answer:"
        )

        try:
            if selector is not None:
                # Use Qwen selector
                answer = selector.generate(full_prompt, clip_path=clip_path, max_new_tokens=5)
            else:
                # Use Llama (self.model)
                tokenizer = self.chat.model.llama_tokenizer
                llama = self.chat.model.llama_model
                input_text = f"[INST] {full_prompt} [/INST]"
                inputs = tokenizer([input_text], return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = llama.generate(
                        **inputs, max_new_tokens=5, do_sample=False,
                    )
                generated = outputs[0][inputs["input_ids"].shape[-1]:]
                answer = tokenizer.decode(generated, skip_special_tokens=True).strip()

            # Parse the number
            import re
            nums = re.findall(r'\d+', answer)
            if nums:
                shuffled_idx = int(nums[0]) - 1
                if 0 <= shuffled_idx < len(shuffled_candidates):
                    chosen = shuffled_candidates[shuffled_idx]
                    print(f"    [Selector] picked #{shuffled_idx+1}: {chosen[:60]}...")
                    return chosen
        except Exception as e:
            print(f"    [Selector] Error: {e}")

        # Fallback: return first candidate
        return candidates[0]

    def infer_one_segment(
        self,
        clip_path: Path,
        context_text: str,
        task_prompt: str,
        temperature: float = 0.2,
        max_new_tokens: int = 256,
        face_avatars: Optional[List[Path]] = None,
        character_names: Optional[List[str]] = None,
        num_beams: int = 1,
    ) -> Tuple[str, float, str]:
        torch.cuda.empty_cache()

        with torch.no_grad():
            start_time = time.monotonic()

            conv = default_conversation.copy()
            img_list: List[Any] = []

            if str(context_text).strip():
                self.chat.ask(context_text.strip(), conv)

            if face_avatars:
                for i, avatar in enumerate(face_avatars):
                    if avatar.is_file():
                        try:
                            self.chat.upload_img(str(avatar), conv, img_list)
                        except Exception as e:
                            print(f"    upload_avatar[{i}] failed: {e}")
                if character_names:
                    name_labels = ", ".join(character_names)
                    self.chat.ask(f"[These faces belong to: {name_labels}]", conv)

            self.chat.upload_video_without_audio(str(clip_path), conv, img_list)

            self.chat.ask(task_prompt.strip(), conv)

            prompt_text = conv.get_prompt()

            torch.cuda.synchronize(self.device)

            answer, _ = self.chat.answer(
                conv=conv,
                img_list=img_list,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                num_beams=num_beams,
            )

        torch.cuda.empty_cache()

        elapsed = time.monotonic() - start_time
        result = str(answer).replace("</s>", "").strip()
        return result, elapsed, prompt_text


def build_ad_engine(
    gpu_id: int = 0,
    cfg_path: Optional[Path] = None,
    llama_dir: Optional[Path] = None,
    ckpt_file: Optional[Path] = None,
) -> ADEngine:
    cfg_path = Path(cfg_path or DEFAULT_CFG_PATH)
    llama_dir = Path(llama_dir or DEFAULT_LLAMA_DIR)
    ckpt_file = Path(ckpt_file or DEFAULT_CKPT_FILE)

    if not cfg_path.is_file():
        raise FileNotFoundError(f"cfg not found: {cfg_path}")
    if not llama_dir.is_dir():
        raise FileNotFoundError(f"LLaMA dir not found: {llama_dir}")
    if not ckpt_file.is_file():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_file}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available.")
    if gpu_id >= torch.cuda.device_count():
        raise ValueError(f"GPU {gpu_id} not available (max {torch.cuda.device_count() - 1})")

    device = f"cuda:{gpu_id}"
    gpu_name = torch.cuda.get_device_name(gpu_id)
    print(f"[AD Engine] Using GPU {gpu_id}: {gpu_name}")

    class _FakeArgs:
        options = []
    _FakeArgs.cfg_path = str(cfg_path)

    cfg = Config(_FakeArgs)
    model_cfg = cfg.model_cfg
    model_cfg.llama_model = str(llama_dir)
    model_cfg.ckpt = str(ckpt_file)

    model_cls = registry.get_model_class(model_cfg.arch)
    model = model_cls.from_config(model_cfg).to(device)
    model.eval()

    vis_cfg = cfg.datasets_cfg.webvid.vis_processor.train
    vis_processor = registry.get_processor_class(vis_cfg.name).from_config(vis_cfg)

    chat = Chat(model, vis_processor, device=device)
    print("[AD Engine] Model loaded successfully.")

    return ADEngine(chat=chat, device=device)
