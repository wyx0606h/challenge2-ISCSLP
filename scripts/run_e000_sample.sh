#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ISCSLP_ROOT="${ISCSLP_ROOT:-/root/autodl-tmp/iscslp2026}"
ISCSLP_MODELS="${ISCSLP_MODELS:-$ISCSLP_ROOT/models}"
ISCSLP_ARTIFACTS="${ISCSLP_ARTIFACTS:-$ISCSLP_ROOT/artifacts}"
HF_HOME="${HF_HOME:-$ISCSLP_ROOT/cache/huggingface}"
XDG_CACHE_HOME="${XDG_CACHE_HOME:-$ISCSLP_ROOT/cache/xdg}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

export HF_HOME XDG_CACHE_HOME CUDA_VISIBLE_DEVICES
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p \
  "$ISCSLP_ARTIFACTS/logs/E000_sample" \
  "$ISCSLP_ARTIFACTS/outputs/E000_sample" \
  "$ISCSLP_ARTIFACTS/metrics/E000_sample"

cd "$REPO_DIR"

OUTPUT_PREFIX="$ISCSLP_ARTIFACTS/outputs/E000_sample/text_history_demo"
LOG_PATH="$ISCSLP_ARTIFACTS/logs/E000_sample/run_e000_sample.$(date +%Y%m%d_%H%M%S).log"

python infer/cot_tts_text_history_inference.py \
  --checkpoint_path "$ISCSLP_MODELS/cot_tts_text_history_baseline/checkpoints/global_step_24500" \
  --history_text_path infer/cases/sample_case/history.txt \
  --target_text_path infer/cases/sample_case/target.txt \
  --ref_audio_path infer/cases/sample_case/reference.wav \
  --output_prefix "$OUTPUT_PREFIX" \
  --spark_model_dir "$ISCSLP_MODELS/Spark-TTS" \
  --model_architecture lora \
  --device cuda:0 \
  --do_sample \
  --temperature 0.6 \
  --top_p 0.75 \
  --min_new_tokens 256 \
  --use_ref_global_tokens \
  --num_candidates 1 \
  --rerank_metric none \
  --max_new_tokens 2000 2>&1 | tee "$LOG_PATH"

echo "$OUTPUT_PREFIX" > "$ISCSLP_ARTIFACTS/outputs/E000_sample/latest_prefix.txt"
echo "$LOG_PATH" > "$ISCSLP_ARTIFACTS/logs/E000_sample/latest_log.txt"
