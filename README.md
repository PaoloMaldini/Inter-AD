# Inter-AD

This repository contains the runnable code for the interactive audio-description (AD) experiments.

## Included package

- `streamingAD/`: streaming AD generation, interactive experiment runners, and evaluation scripts.
- `streamingAD/face_gallery_data/`: compact face-gallery metadata used by the supplied pipelines.
- `requirements-gemma4.txt`: exact pip package snapshot from the Gemma-4 environment (Python 3.10.20, PyTorch 2.7.1+cu118).

Generated outputs, videos, model checkpoints, Hugging Face caches, and Python bytecode are intentionally excluded. The larger PlotTree feature binary (`plotree_features.pkl`) is also kept as an external data dependency; copy it into `streamingAD/face_gallery_data/` when running pipelines that use face-embedding lookup.

## Environment

For a matching environment:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-gemma4.txt
```

The requirements file captures the existing CUDA 11.8 build. On a machine with a different CUDA version, install a matching PyTorch build and adjust the torch/torchvision/triton/nvidia entries accordingly. The model checkpoints, movie assets, ffmpeg, and any local Video-LLaMA/Gemma code are external assets. Paths in `streamingAD/run.sh` are examples for the original server and should be adjusted on the target machine.

## Run the interactive evaluator

```bash
python -u streamingAD/workplace/eval_interactive.py --help
```

For a full GPU evaluation, provide `--instructed-dir`, `--baseline-dir`, `--output-dir`, and `--judge-gpu`. Add `--segment-judge` to enable the additional local-window LLM checks.
