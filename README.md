# ISCSLP2026 Challenge Baseline

> Project policy for `wyx0606h/challenge2-ISCSLP`: `main` is the frozen
> project root and official-baseline reference. Do not run experiments or
> commit experimental code directly on `main`. All experiments must start
> from an `exp/*` branch; the first one is `exp/baseline`.
>
> Start with [CHALLENGE_PLAN.md](CHALLENGE_PLAN.md), [AGENTS.md](AGENTS.md),
> and [docs/SERVER_RUNBOOK.md](docs/SERVER_RUNBOOK.md).

This repository contains the inference code for two COT-TTS baselines:

- audio-history COT-TTS: the dialogue history is provided as audio
- text-history COT-TTS: the dialogue history is provided as text

The repository includes:

- `infer/`: inference scripts, launch scripts, and sample inputs
- `sparktts/`: Spark-TTS / BiCodec runtime code
- `veomni/`: VeOmni runtime code required by inference
- `env/spark_infer.yaml`: reference conda environment file
- `LICENSE` and `THIRD_PARTY_NOTICES.md`

Large model files are not stored in git. They should be downloaded separately into `model/`.

## 1. Code Overview

Main inference entrypoints:

- `infer/cot_tts_inference.py`: audio-history inference
- `infer/cot_tts_text_history_inference.py`: text-history inference
- `infer/run_cot_tts_inference.sh`: audio-history demo launcher
- `infer/run_cot_tts_text_history_inference.sh`: text-history demo launcher

Bundled sample case:

```text
infer/cases/sample_case/
├── history.txt
├── history.wav
├── reference.wav
└── target.txt
```

- `history.wav` is used by the audio-history baseline.
- `history.txt` is used by the text-history baseline.
- `reference.wav` is the reference speaker audio.
- `target.txt` is the target content to synthesize.

## 2. Environment Setup

If you already have the verified environment, activate it directly:

```bash
conda activate /data/weizhen/envs/spark
```

If you need to recreate the environment from scratch:

```bash
conda env create -f env/spark_infer.yaml
conda activate spark_audio_infer
```

Then enter the repository root:

```bash
cd /path/to/ISCSLP2026-Challenge-baseline
```

## 3. Model Download

Download the following three model bundles and place them under `model/`:

- Spark-TTS / BiCodec assets: [Link](https://drive.google.com/drive/folders/1t6ICBCWwpwPIqTEioNwu3ccTR6f7srC3?usp=sharing)
- Audio-history baseline model: [Link](https://drive.google.com/drive/folders/18QPjHkMwHBsVlyANMxE0oQRj6wbzrwwH?usp=sharing)
- Text-history baseline model: [Link](https://drive.google.com/drive/folders/1xvkuAW7f7PmVB1D6lVyC6wK8AgbA10J9?usp=sharing)

Expected directory layout after download:

```text
ISCSLP2026-Challenge-baseline/
├── model/
│   ├── Spark-TTS/
│   ├── cot_tts_audio_history_baseline/
│   │   ├── checkpoints/
│   │   │   └── global_step_37000/
│   │   │       └── hf_ckpt/
│   │   ├── model_assets/
│   │   └── veomni_cli.yaml
│   └── cot_tts_text_history_baseline/
│       ├── checkpoints/
│       │   └── global_step_24500/
│       │       └── hf_ckpt/
│       ├── model_assets/
│       └── veomni_cli.yaml
├── infer/
├── sparktts/
└── veomni/
```

If you want to test another checkpoint, point the inference script to a different `global_step_*` directory or provide a matching `hf_ckpt/`.

## 4. Data Download

Download the following dataset: [Link](https://huggingface.co/datasets/HKUSTAudio/chall)

This dataset is prepared for the ISCSLP 2026 CoT-TTS Challenge and is designed to support research on context-aware, expressive, and CoT-guided speech generation. It is constructed from speech-rich media sources, including films, TV dramas, radio dramas, and short dramas, where dialogue often contains rich conversational context, speaker interactions, scene changes, and emotional variation. Each sample is organized around a target utterance, together with its preceding dialogue context, reference speech, metadata, and corresponding annotations, so that models can learn to infer an appropriate speaking style from context rather than relying only on the target text.

| Language | Duration | Ratio | Segments |
|---|---:|---:|---:|
| English | ~8.6K h | 54% | ~1.62M |
| Chinese | ~7.4K h | 46% | ~1.38M |
| Total | ~16K h | 100% | ~3.0M |


The released audio files are extracted segments from the original source recordings. To reduce storage usage, only the audio file format has been standardized to FLAC, while other acoustic properties are preserved as much as possible. The audio has not been aggressively standardized or denoised. As a result, different files may have different sampling rates, channel configurations, loudness levels, background sounds, or environmental noise. This design is intentional: the dataset aims to retain realistic acoustic conditions and avoid excessive preprocessing assumptions, allowing users and participants to decide how to process, normalize, enhance, or filter the audio according to their own methods. However, the metadata and annotations provided in this dataset were generated based on normalized and denoised versions of the corresponding audio segments, in order to improve annotation reliability and accuracy.



## 5. Inference Usage

Outputs are written to `infer/results/`. If `CUDA_VISIBLE_DEVICES` is not set, the launch scripts automatically select the currently emptiest GPU.

Run the audio-history baseline on the bundled sample case:

```bash
CUDA_VISIBLE_DEVICES=0 bash infer/run_cot_tts_inference.sh
```

Run the text-history baseline on the bundled sample case:

```bash
CUDA_VISIBLE_DEVICES=0 bash infer/run_cot_tts_text_history_inference.sh
```

The two launch scripts are equivalent to calling the Python entrypoints directly. For example, the text-history baseline can be launched with:

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

To run your own examples, replace the files in `infer/cases/sample_case/` or pass custom paths through the command-line arguments.

## 6.Contact

For questions, issues, or requests related to the dataset, please contact:

- Weizhen Bian: wbian@connect.ust.hk
