# Track 1 Server Runbook

Use this document on the GPU server from `exp/baseline`.

## 1. Clone and Select the Mutable Branch

```bash
git clone https://github.com/wyx0606h/challenge2-ISCSLP.git
cd challenge2-ISCSLP
git switch exp/baseline
```

Never run server experiments from `main`.

## 2. Recommended External Storage Layout

Keep data, weights, caches, and outputs outside the repository:

```text
/path/to/iscslp2026/
├── data/
│   └── ISCSLP2026-CoT-TTS/
├── models/
│   ├── Spark-TTS/
│   └── cot_tts_text_history_baseline/
├── artifacts/
│   ├── logs/
│   ├── outputs/
│   ├── metrics/
│   └── submissions/
└── cache/
```

Suggested environment variables:

```bash
export ISCSLP_ROOT=/path/to/iscslp2026
export ISCSLP_DATA=$ISCSLP_ROOT/data/ISCSLP2026-CoT-TTS
export ISCSLP_MODELS=$ISCSLP_ROOT/models
export ISCSLP_ARTIFACTS=$ISCSLP_ROOT/artifacts
export HF_HOME=$ISCSLP_ROOT/cache/huggingface
```

Use symlinks only on `exp/baseline`, never commit them:

```bash
ln -s "$ISCSLP_MODELS" model
```

## 3. Preflight Inventory

Record:

```bash
nvidia-smi
python --version
which python
git rev-parse HEAD
git status --short
df -h
```

Create SHA256 manifests for model bundles and important metadata. Avoid
hashing the entire 16K-hour dataset repeatedly; hash release manifests and
metadata first, then maintain a deterministic file inventory.

## 4. Environment Reconstruction

The imported official commit does not contain the advertised
`env/spark_infer.yaml`. Build the environment on `exp/baseline` and immediately
export a reproducible lock.

Likely runtime families visible in the code include:

- Python;
- PyTorch and torchaudio;
- transformers;
- safetensors;
- NumPy;
- soundfile/libsndfile;
- FlashAttention 2;
- Spark-TTS/BiCodec dependencies;
- VeOmni dependencies;
- PyYAML for LoRA config reading.

Do not guess versions in the final package. Verify by running the official
sample, then export the exact environment.

## 5. Expected Model Layout

```text
model/
├── Spark-TTS/
└── cot_tts_text_history_baseline/
    ├── checkpoints/
    │   └── global_step_24500/
    │       └── hf_ckpt/
    ├── model_assets/
    └── veomni_cli.yaml
```

Prefer the released `hf_ckpt/`. The source fallback for converting DCP
checkpoints references a helper absent from the imported official commit.

## 6. Official Track 1 Smoke Test

```bash
CUDA_VISIBLE_DEVICES=0 bash infer/run_cot_tts_text_history_inference.sh
```

Equivalent core command:

```bash
CUDA_VISIBLE_DEVICES=0 python infer/cot_tts_text_history_inference.py \
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
```

Check at minimum:

- output reasoning is non-empty and sample-specific;
- WAV opens successfully;
- sample rate is 16 kHz unless the official schema later says otherwise;
- samples are finite and not all zero;
- duration is plausible;
- there are no missing semantic audio tokens;
- the metadata records the exact decoding settings.

## 7. Parameter Audit

Count parameters for every inference component after loading:

- text/CoT-TTS backbone and LoRA modules;
- BiCodec/Spark-TTS tokenizer and decoder;
- any speaker encoder;
- any candidate scorer, reranker, denoiser, enhancer, or postprocessor.

Deduplicate shared tensors by storage identity where appropriate and report:

```text
component, total_params, trainable_params, dtype, memory_bytes, invoked
```

The sum of all invoked modules must be strictly below 1,000,000,000. Keep a
safety margin; do not target exactly the boundary.

## 8. RTF Measurement

Until the organizers publish hardware/details, report both:

- steady-state RTF excluding one-time model load;
- end-to-end RTF including launch/load.

For a batch:

```text
RTF = total wall-clock inference time / total generated audio duration
```

Use synchronization around GPU timing, warm up first, and report:

- hardware;
- precision;
- batch size;
- number of samples;
- total generated duration;
- failures/retries;
- median, p90, and aggregate RTF;
- peak VRAM.

Official eligibility requires RTF `<=3.0`.

## 9. Evaluation Harness TODO

After the official sample succeeds, run every candidate through a local Track 1
evaluation harness before trusting any subjective comparison.

The first implementation should provide:

```bash
python tools/eval_track1.py \
  --manifest "$ISCSLP_ARTIFACTS/manifests/track1_fast_dev.jsonl" \
  --output-dir "$ISCSLP_ARTIFACTS/eval/baseline_fast_dev" \
  --checkpoint "$ISCSLP_MODELS/cot_tts_text_history_baseline/checkpoints/global_step_24500" \
  --spark-model-dir "$ISCSLP_MODELS/Spark-TTS" \
  --device cuda:0
```

Expected artifacts:

- per-sample reasoning text and WAV paths;
- output-coverage and audio-validity report;
- bilingual CER/WER proxy report;
- reasoning grounding and non-template checks;
- speech/style proxy features and slice summaries;
- parameter-count report with `<1B` hard gate;
- steady-state and end-to-end RTF report with peak VRAM;
- one Markdown summary that can be copied into `experiments/`.

Keep the manifest immutable once frozen. Baseline and improved systems must use
the same sample IDs, reference audio, context text, target text, decoding
settings policy, and metric versions unless an experiment record explicitly
declares the change.

See `docs/EVALUATION_TODO.md` for the full design checklist.

## 10. Offline Packaging Test

Before submission:

1. Copy the candidate package into a clean directory/container.
2. Disable network access.
3. clear unrelated local caches;
4. run installation;
5. run the full inference adapter;
6. validate every reasoning and WAV output;
7. record archive SHA256 and size.

The package must include all model assets, tokenizers, vocoders, configs, and
auxiliary files required at inference time.

## 11. First Server Session Checklist

- [ ] On `exp/baseline`, not `main`.
- [ ] Correct official model and dataset checksums recorded.
- [ ] GPU, CUDA, driver, Python, PyTorch captured.
- [ ] Official sample succeeds.
- [ ] Small bilingual manifest succeeds.
- [ ] Local Track 1 evaluation harness TODO reviewed and any available checks
      run.
- [ ] Parameter audit complete and `<1B`.
- [ ] RTF measured and `<=3.0`.
- [ ] No network access needed after assets are staged.
- [ ] Environment lock exported.
- [ ] Experiment record updated.
