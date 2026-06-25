# AGENTS.md

## Mission

This repository is the working project for the **ISCSLP 2026 CoT-TTS
Challenge**, targeting:

- **Track 1**: Text-Context-Aware CoT-TTS.
- **Category**: Parameter-Constrained.
- **Hard inference-time parameter limit**: fewer than 1 billion parameters,
  counting every loaded or invoked module.

The system receives speaker-labeled dialogue history, target text, and
reference speech. It must output both:

1. a sample-specific reasoning analysis of the intended speaking manner; and
2. a speech waveform that preserves the reference timbre and matches the
   inferred context/style.

## Branch Contract

- `main` is the immutable project root.
  - It contains the imported official baseline plus project documentation.
  - Do not train, tune, refactor, or commit experimental changes here.
  - Only an explicitly approved baseline/vendor refresh may change `main`.
- `exp/baseline` is the first executable experiment branch.
  - Reproduce the official Track 1 baseline before changing the model.
  - Record environment, model hashes, parameter counts, RTF, commands, and
    outputs.
- Future experiments use `exp/<short-name>`.
  - One hypothesis per branch.
  - Branch from `exp/baseline` after baseline reproduction, unless the
    experiment deliberately needs a clean `main`.
  - Never merge experimental code back to `main`.
- Stable reusable tooling may be selectively cherry-picked into a dedicated
  integration branch after review. Do not use `main` as the integration
  branch.

See `docs/BRANCHING_STRATEGY.md`.

## Non-Negotiable Challenge Rules

- No cascaded ASR-LLM-TTS inference pipeline.
- No online API or remote model call during inference.
- Official evaluation runs without internet access.
- Official RTF must be `<= 3.0`.
- The constrained category must use `< 1B` total inference-time parameters.
- Frozen modules, vocoders, speech tokenizers, speaker encoders, auxiliary
  models, and post-processing models all count.
- Every external dataset, pretrained model, synthetic-data source,
  augmentation, post-processing method, and auxiliary resource must be
  declared.
- The submission must generate reasoning and waveform outputs for every test
  sample.
- Do not inspect, identify, reconstruct, or trace hidden-test source media.
- Dataset use is non-commercial research only; redistribution and speaker
  identification are prohibited.

## Working Rules for Agents and Contributors

Before changing code:

1. Confirm the current branch. Stop if it is `main`.
2. Read `CHALLENGE_PLAN.md` and the relevant file under `docs/`.
3. State one experiment hypothesis and one primary success metric.
4. Create or update an experiment record under `experiments/`.
5. Preserve an exact baseline command for comparison.

For every experiment, record:

- Git commit and branch.
- Date, owner, host, GPU model, CUDA/driver, Python, PyTorch, and package lock.
- Dataset split and immutable manifest/hash.
- Model/checkpoint hashes and all external resources.
- Trainable parameters and total inference-time parameters.
- Training command, inference command, seed, config, and decoding settings.
- Runtime, peak VRAM, failures, RTF, and output coverage.
- Objective proxies for intelligibility, speaker similarity, style/prosody,
  quality, reasoning quality, and speech-reasoning consistency.
- Conclusion: keep, reject, or investigate.

## Repository Hygiene

- Never commit model weights, downloaded challenge data, generated audio
  batches, credentials, or machine-specific paths.
- Store large artifacts outside Git and reference them through documented
  environment variables or manifests.
- Keep all inference dependencies local and packageable for offline execution.
- Prefer deterministic preprocessing and versioned manifests.
- Do not silently change sample selection, prompt format, audio normalization,
  or evaluation scripts.
- A result without a reproducible command and artifact hashes is not a valid
  experiment result.

## Baseline Caveats Observed at Import

The imported official baseline README refers to
`env/spark_infer.yaml`, but that file is not present in official commit
`46cd111eb90563a2bdebe96daa3a07730b007e34`. The inference code also contains a
DCP-to-HF conversion fallback that expects `scripts/merge_dcp_to_hf.py`, which
is absent in that commit. Prefer the released checkpoints containing
`hf_ckpt/`, and create a verified environment lock on `exp/baseline`.

The official repository also contains generated outputs and Python bytecode.
Treat them as vendor artifacts, not as trustworthy evaluation references.

## Definition of Done for `exp/baseline`

The baseline branch is considered reproduced only when:

- the Track 1 sample runs end-to-end on the target server;
- a small frozen validation manifest runs with 100% output coverage;
- both reasoning text and valid WAV files are produced;
- total inference parameter count is audited and `<1B`;
- RTF is measured with the official-style definition and is `<=3.0`;
- the environment can be recreated offline;
- commands, logs, hashes, metrics, and known issues are documented.
