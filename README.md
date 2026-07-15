# Inter-AD

This repository contains the runnable code for the interactive audio-description (AD) experiments.

## Included package

- `streamingAD/`: streaming AD generation, interactive experiment runners, and evaluation scripts.
- `streamingAD/face_gallery_data/`: compact face-gallery metadata used by the supplied pipelines.

Generated outputs, videos, model checkpoints, Hugging Face caches, and Python bytecode are intentionally excluded. The larger PlotTree feature binary (`plotree_features.pkl`) is also kept as an external data dependency; copy it into `streamingAD/face_gallery_data/` when running pipelines that use face-embedding lookup.

## Environment

The pipelines expect the external Video-LLaMA/Gemma-Qwen model checkpoints and preprocessed movie assets described in `streamingAD/README.md`. Paths in `streamingAD/run.sh` are examples for the original server and should be adjusted on the target machine.

## Run the interactive evaluator

```bash
python -u streamingAD/workplace/eval_interactive.py --help
```

For a full GPU evaluation, provide `--instructed-dir`, `--baseline-dir`, `--output-dir`, and `--judge-gpu`. Add `--segment-judge` to enable the additional local-window LLM checks.
