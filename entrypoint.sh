#!/bin/bash
# Qwen3.8-27B AWQ-INT4 + DSpark on RTX 3090s (validated presets)
# Env: TP (1|2|4, default 1), PORT (default 8000), DSPARK (1|0, default 1), MODELS_DIR (default /models)
set -e
TP=${TP:-1}
PORT=${PORT:-8000}
DSPARK=${DSPARK:-1}
MODELS=${MODELS_DIR:-/models}
AWQ=$MODELS/Qwen3.8-27B-AWQ-INT4
DRAFT=$MODELS/Qwen3.8-27B-DSpark

echo "[entrypoint] TP=$TP DSPARK=$DSPARK PORT=$PORT models=$MODELS"

# --- download weights if missing (~23 GB target + 2.6 GB draft) ---
if [ ! -f "$AWQ/model.safetensors.index.json" ]; then
  echo "[entrypoint] downloading AWQ target (cyankiwi/Qwen3.8-27B-AWQ-INT4)…"
  hf download cyankiwi/Qwen3.8-27B-AWQ-INT4 --local-dir "$AWQ"
fi
if [ "$DSPARK" = "1" ] && [ ! -f "$DRAFT/model.safetensors" ]; then
  echo "[entrypoint] downloading DSpark draft (RadixArk/Qwen3.8-27B-DSpark)…"
  hf download RadixArk/Qwen3.8-27B-DSpark --local-dir "$DRAFT"
fi

# --- validated presets per GPU count ---
# GDN hybrid memory tuning: TP1 needs a small mamba state pool to fit one 24 GB card
case "$TP:$DSPARK" in
  1:1) TPFLAGS="--mem-fraction-static 0.92 --max-mamba-cache-size 16" ;;
  1:0) TPFLAGS="--mem-fraction-static 0.92 --max-mamba-cache-size 48" ;;
  2:1) TPFLAGS="--mem-fraction-static 0.88 --max-mamba-cache-size 64" ;;
  2:0) TPFLAGS="--mem-fraction-static 0.90 --max-mamba-cache-size 96" ;;
  4:1) TPFLAGS="--mem-fraction-static 0.72" ;;
  4:0) TPFLAGS="--mem-fraction-static 0.80" ;;
  *) echo "TP must be 1, 2 or 4"; exit 1 ;;
esac

SPECFLAGS=""
if [ "$DSPARK" = "1" ]; then
  SPECFLAGS="--speculative-algorithm DSPARK --speculative-draft-model-path $DRAFT --speculative-dspark-block-size 7 --speculative-draft-model-quantization unquant"
fi

echo "[entrypoint] launching sglang…"
exec python3 -m sglang.launch_server \
  --model-path "$AWQ" \
  --tp "$TP" \
  --chunked-prefill-size 4096 \
  --mamba-radix-cache-strategy extra_buffer \
  $TPFLAGS $SPECFLAGS \
  --host 0.0.0.0 --port "$PORT"
