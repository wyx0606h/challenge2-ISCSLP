#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]] && command -v nvidia-smi >/dev/null 2>&1; then
  CUDA_VISIBLE_DEVICES="$(
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
      | sort -t',' -k2n \
      | head -n 1 \
      | cut -d',' -f1 \
      | tr -d ' '
  )"
  export CUDA_VISIBLE_DEVICES
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

cd "${REPO_ROOT}"
mkdir -p infer/results

python infer/cot_tts_text_history_inference.py \
  --checkpoint_path model/cot_tts_text_history_baseline/checkpoints/global_step_24500 \
  --history_text_path infer/cases/sample_case/history.txt \
  --target_text_path infer/cases/sample_case/target.txt \
  --ref_audio_path infer/cases/sample_case/reference.wav \
  --output_prefix infer/results/text_history_demo \
  --spark_model_dir model/Spark-TTS \
  --model_architecture lora \
  --device cuda:0 \
  --do_sample \
  --temperature 0.6 \
  --top_p 0.75 \
  --min_new_tokens 256 \
  --use_ref_global_tokens \
  --num_candidates 1 \
  --rerank_metric none \
  --max_new_tokens 2000
