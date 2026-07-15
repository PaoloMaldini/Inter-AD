#!/usr/bin/env bash
set -euo pipefail

STREAMING_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$STREAMING_DIR/.." && pwd)"
VENV_PYTHON="/mnt/disk6new/wzq/env/videollava/bin/python"

GPU_ID="${GPU_ID:-1}"
PORT="${PORT:-7860}"
SHARE="${SHARE:-}"
MOVIE_PATH="${MOVIE_PATH:-/mnt/disk1new/storyvideo/Movie/IMDB-001-The Shawshank Redemption.mp4}"

export HF_HOME="${HF_HOME:-/tmp/hf_cache}"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

echo "============================================"
echo "  Streaming AD Generator"
echo "  GPU:   ${GPU_ID}"
echo "  Port:  ${PORT}"
echo "  Movie: ${MOVIE_PATH}"
echo "  Env:   ${VENV_PYTHON}"
echo "============================================"

SHARE_FLAG=""
if [ "${SHARE}" = "1" ] || [ "${SHARE}" = "true" ]; then
    SHARE_FLAG="--share"
fi

cd "$PROJECT_ROOT"
exec "$VENV_PYTHON" "$STREAMING_DIR/streaming_ad.py" \
    --gpu-id "$GPU_ID" \
    --port "$PORT" \
    --movie-path "$MOVIE_PATH" \
    $SHARE_FLAG \
    --server-name 0.0.0.0
