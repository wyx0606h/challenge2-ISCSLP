# E000 Baseline Reproduction

This document records the first Track 1 official baseline reproduction attempt
on `exp/baseline`.

## Server Paths

The server does not have `/workspace`, so this run uses Git-external storage:

```bash
export REPO_DIR=/root/challenge2-ISCSLP
export ISCSLP_ROOT=/root/autodl-tmp/iscslp2026
export ISCSLP_MODELS=$ISCSLP_ROOT/models
export ISCSLP_ARTIFACTS=$ISCSLP_ROOT/artifacts
export HF_HOME=$ISCSLP_ROOT/cache/huggingface
export XDG_CACHE_HOME=$ISCSLP_ROOT/cache/xdg
```

Large models, logs, generated audio, metrics, caches, and envs stay under
`$ISCSLP_ROOT`, outside Git.

## Environment

The repository does not include the README-advertised `env/spark_infer.yaml`.
The E000 environment was cloned from server base and repaired in place:

```bash
conda create -p "$ISCSLP_ROOT/envs/e000" --clone base -y
PATH="$ISCSLP_ROOT/envs/e000/bin:$PATH" python --version
```

The exported locks are:

```text
env/requirements-e000.txt
env/environment-e000.yml
```

Two local compatibility shims are present in the cloned env `sitecustomize.py`:

- PyTorch 2.1 `_pytree` compatibility for Transformers 4.57.3.
- Compatibility aliases for older Diffusers/Hugging Face Hub constants.

Spark wav2vec2 weights were converted from `pytorch_model.bin` to
`model.safetensors` in the external model directory to avoid the Transformers
torch-load safety gate on PyTorch 2.1.

## Model Placement

Expected external model paths:

```text
$ISCSLP_MODELS/
├── Spark-TTS/
└── cot_tts_text_history_baseline/
    ├── checkpoints/global_step_24500/hf_ckpt/
    ├── model_assets/
    └── veomni_cli.yaml
```

The source zip hashes and extracted file inventory are stored under:

```text
$ISCSLP_ARTIFACTS/hashes/
```

## Sample Command

Use the wrapper to keep outputs outside the repository:

```bash
cd "$REPO_DIR"
PATH="$ISCSLP_ROOT/envs/e000/bin:$PATH" \
ISCSLP_ROOT="$ISCSLP_ROOT" \
CUDA_VISIBLE_DEVICES=0 \
scripts/run_e000_sample.sh
```

The wrapper preserves the official Track 1 inference parameters:

- checkpoint `global_step_24500`;
- `--model_architecture lora`;
- `--temperature 0.6`;
- `--top_p 0.75`;
- `--min_new_tokens 256`;
- `--max_new_tokens 2000`;
- `--use_ref_global_tokens`;
- `--num_candidates 1`;
- `--rerank_metric none`.

## Validation

Validate the generated sample output:

```bash
PATH="$ISCSLP_ROOT/envs/e000/bin:$PATH" \
python tools/validate_track1_outputs.py \
  --prefix "$ISCSLP_ARTIFACTS/outputs/E000_sample/text_history_demo" \
  --expected-sr 16000 \
  --report "$ISCSLP_ARTIFACTS/metrics/E000_sample/validation.json"
```

The official sample has no real `target.wav` and no real CoT label. It supports
format, coverage, token, RTF, VRAM, and sanity checks only. Metrics requiring
target audio or gold CoT are not computed for this sample.

## Parameter Audit

Run:

```bash
PATH="$ISCSLP_ROOT/envs/e000/bin:$PATH" \
CUDA_VISIBLE_DEVICES=0 \
python tools/count_inference_params.py \
  --checkpoint "$ISCSLP_MODELS/cot_tts_text_history_baseline/checkpoints/global_step_24500" \
  --spark-model-dir "$ISCSLP_MODELS/Spark-TTS" \
  --model-architecture lora \
  --device cuda:0 \
  --report "$ISCSLP_ARTIFACTS/metrics/E000_sample/params.json"
```

The E000 audit currently reports the loaded/invoked modules exceed the 1B
parameter limit. See `docs/KNOWN_ISSUES.md`.

## Evidence Locations

```text
$ISCSLP_ARTIFACTS/logs/E000_sample/
$ISCSLP_ARTIFACTS/outputs/E000_sample/
$ISCSLP_ARTIFACTS/metrics/E000_sample/
$ISCSLP_ARTIFACTS/env/
$ISCSLP_ARTIFACTS/hashes/
```
