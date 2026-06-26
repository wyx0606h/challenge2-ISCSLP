# ISCSLP 2026 CoT-TTS Track 1 Detailed Plan

Last verified against the official challenge website: **2026-06-25**.

## 1. Goal and Success Criteria

We will enter:

- Track: **Track 1 — Text-Context-Aware CoT-TTS**
- Leaderboard: **Parameter-Constrained**
- Input: speaker-labeled text dialogue context, target text, reference speech
- Required output: per-sample reasoning analysis plus synthesized waveform

The challenge score combines:

- 30% objective evaluation: speech quality, intelligibility, speaker
  similarity, prosody/expression, and efficiency.
- 20% LLM-based evaluation: contextual understanding, internal logical
  coherence, and informativeness of reasoning.
- 50% human evaluation: contextual coherence, reasoning accuracy,
  informativeness, naturalness, and speech-reasoning consistency.

Our engineering gates are stricter than “the sample runs”:

1. `<1B` total inference-time parameters, including all auxiliary modules.
2. Official-style RTF `<=3.0`.
3. Offline, complete, directly executable inference package.
4. 100% generation coverage on the frozen local validation set.
5. Reproducible reasoning and audio outputs with traceable configs and hashes.

## 2. Official Dates and Immediate Actions

As of 2026-06-25:

| Date | Milestone | Team action |
|---|---|---|
| 2026-06-20 | Website, data, baseline, registration opened | Released |
| **2026-07-04** | **Registration deadline** | Register immediately; do not wait for baseline reproduction |
| 2026-07-10 to 2026-07-18 | First submission round | Submit a valid reproducible baseline package early |
| 2026-07-19 to 2026-07-25 | Final submission round | Submit selected final system |
| **2026-07-25** | **System/model deadline** | Hard internal target: July 23 |
| 2026-07-26 to 2026-08-08 | Objective evaluation | Keep exact submitted artifact frozen |
| 2026-08-03 | Paper deadline | Internal complete draft: July 30 |
| 2026-08-09 to 2026-08-25 | LLM/human evaluation | No artifact drift |
| 2026-08-26 to 2026-08-30 | Final ranking | Archive all records |
| 2026-09-21 | Camera-ready deadline | Update final results |
| 2026-11-14 to 2026-11-17 | ISCSLP 2026, Penang | Prepare challenge presentation if selected |

Immediate checklist:

- [ ] Submit the official team registration form before 2026-07-04.
- [ ] Confirm team name, primary contact, members, affiliation, Track 1, and
      Parameter-Constrained category.
- [ ] Download the official data and model bundles to controlled storage.
- [ ] Record checksums, access date, source URL, and license terms.
- [ ] Reserve at least one target GPU server for uninterrupted reproduction.
- [ ] Agree on an internal artifact owner and submission owner.

## 3. Rules That Shape the Technical Design

### 3.1 Parameter constraint

The limit is not just the text backbone. Count every loaded/invoked component:

- Qwen-style CoT-TTS model;
- Spark-TTS/BiCodec tokenizer and decoder/vocoder;
- speaker-related encoders;
- rerankers, quality models, enhancement modules, and post-processors;
- any separately loaded reasoning or language model.

Do not claim constrained eligibility until a script produces:

- per-module parameter totals;
- deduplicated total inference parameters;
- trainable vs frozen counts;
- evidence of which modules are resident/invoked during inference.

### 3.2 Architecture restrictions

- Do not build a separate ASR -> LLM -> TTS chain.
- Do not call off-the-shelf remote or local modules as a disguised cascade.
- Improvements should stay within an end-to-end context-aware reasoning and
  speech-generation system.
- Candidate generation/reranking is risky if it introduces a separate model:
  it counts toward parameters and may hurt RTF. Start with model-internal or
  low-cost deterministic selection.

### 3.3 Packaging restrictions

- No internet is available during official evaluation.
- All code, checkpoints, tokenizers, vocoders, configs, and auxiliary files
  must be present.
- Provide a Dockerfile, `environment.yml`, `requirements.txt`, or equivalent.
- External resources and preprocessing must be fully declared.

## 4. Baseline Understanding

The official baseline provides two inference paths:

- `infer/cot_tts_text_history_inference.py`: Track 1 text-history inference.
- `infer/cot_tts_inference.py`: Track 2 audio-history inference.

Track 1 baseline defaults:

- checkpoint: `global_step_24500`;
- architecture: LoRA;
- temperature: `0.6`;
- top-p: `0.75`;
- minimum new tokens: `256`;
- maximum new tokens: `2000`;
- reference global tokens enabled;
- one candidate, no reranking;
- expected sample rate: `16 kHz`;
- default dtype: bfloat16;
- default attention: FlashAttention 2.

The Track 1 prompt contains:

- dialogue history inside `<under_start>...<under_end>`;
- target text inside `<text_start>...<text_end>`;
- reference-speaker BiCodec global tokens;
- generation beginning at `<cot_start>`, followed by target audio tokens.

Important import caveats:

- Official source commit: `46cd111eb90563a2bdebe96daa3a07730b007e34`.
- The advertised `env/spark_infer.yaml` is absent.
- The optional DCP conversion helper under `scripts/` is absent.
- Use the released model's existing `hf_ckpt/` first.
- Generated result files and `__pycache__` exist in the official repository;
  do not treat them as clean benchmark evidence.

## 5. Work Plan

### Phase 0 — Governance and assets (June 25-26)

Deliverables:

- registered team;
- frozen `main` and `exp/baseline`;
- challenge/data/model source manifest with SHA256 checksums;
- external-resource declaration started on day one;
- server storage and permissions checked.

Actions:

1. Download the current official Track 1 model, Spark-TTS assets, and dataset.
2. Keep data and weights outside Git.
3. Create a manifest mapping source URL -> local path -> size -> checksum.
4. Read and accept the non-commercial research-use conditions.
5. Do not redistribute or re-host the dataset.

### Phase 1 — Reproduce the official Track 1 baseline (June 25-28)

Deliverables:

- sample-case reasoning and WAV;
- verified environment lock;
- baseline smoke-test script;
- baseline parameter and RTF report;
- initial local validation manifest.

Actions:

1. Recreate the environment on the target server.
2. Prefer a CUDA/PyTorch/FlashAttention combination already supported by the
   server; log exact versions.
3. Run the official sample command unchanged.
4. Validate WAV readability, sample rate, non-zero duration, finite samples,
   and reasoning non-emptiness.
5. Run 20-50 frozen examples spanning Chinese/English, short/long context,
   emotion, noise, and speaker-reference quality.
6. Measure cold-start separately from steady-state RTF.
7. Audit all inference-time parameters.
8. Save logs and environment exports under ignored artifact storage.

Exit gate:

- 100% valid outputs;
- `<1B` audited total;
- RTF `<=3.0`;
- no network use;
- one-command reproducibility.

### Phase 2 — Build reliable data and evaluation pipelines (June 27-July 2)

Dataset facts:

- approximately 16K hours and 3M segments;
- English: about 8.6K hours / 1.62M segments;
- Chinese: about 7.4K hours / 1.38M segments;
- six folders with `metadata.json`, `dialogue_segments/`, and
  `continuous_segments/`;
- Track 1 should use ordered textual `dialog_segments`, target text, reference
  audio, target audio, and optional `cot_text`.

Actions:

1. Build deterministic metadata indexing without copying the dataset.
2. Validate missing files, duplicate IDs, bad audio, empty text, durations,
   sample rates, channels, clipping, and NaN/Inf values.
3. Create leakage-resistant train/dev splits grouped by source media and scene,
   not by random utterance alone.
4. Freeze a small “fast-dev” set and a larger “selection-dev” set.
5. Balance language, context length, emotion, expressive intensity, speaker,
   reference similarity, noise, and duration.
6. Compare `dialogue_segments` and `continuous_segments` only where permitted
   and relevant; Track 1 context remains text.
7. Keep source titles and identities out of reports.

Local metric suite:

- intelligibility: Chinese CER and English WER proxy;
- speaker similarity proxy;
- speech quality/naturalness proxy;
- F0, energy, duration, pause, and speaking-rate statistics;
- emotion/style agreement proxy;
- reasoning checks for specificity, context grounding, coherence, and
  non-template behavior;
- speech-reasoning consistency review;
- RTF, peak VRAM, failure rate, and parameter count.

Automatic metrics are for iteration, not claims of exact leaderboard parity.

Evaluation-script TODOs are tracked in `docs/EVALUATION_TODO.md`. Treat this
as a required companion to baseline reproduction: every baseline or improved
system comparison must run through the same frozen manifests, output validator,
parameter audit, RTF measurement, bilingual slice report, and reasoning/speech
consistency checks before it is considered actionable.

### Phase 3 — Low-risk baseline improvements (June 29-July 8)

Run one controlled hypothesis at a time:

| ID | Hypothesis | Variables | Primary gate |
|---|---|---|---|
| E000 | Official baseline is reproducible | No changes | Coverage, params, RTF |
| E001 | Decoding can improve stability | temperature, top-p, min/max tokens, repetition controls | Quality without RTF/failure regression |
| E002 | Better context serialization improves style reasoning | speaker labels, turn separators, context window, truncation | Reasoning specificity + style |
| E003 | Reference preprocessing improves timbre | silence trimming, channel conversion, loudness policy | Speaker similarity |
| E004 | Data filtering improves training signal | duration, noise, ref similarity, annotation confidence | Selection-dev composite |
| E005 | Balanced bilingual sampling reduces regressions | language/style/context buckets | Worst-slice score |
| E006 | LoRA configuration improves adaptation efficiently | rank, alpha, target modules, schedule | Composite under parameter budget |
| E007 | Reasoning/audio multitask weighting improves consistency | CoT vs audio-token loss weights | Speech-reasoning consistency |
| E008 | Inference optimization preserves quality | compile/cache/batching/attention | RTF and VRAM |

Rules:

- Use fixed seeds and frozen evaluation manifests.
- Do not tune against hidden evaluation data.
- Reject improvements that violate parameters, RTF, offline packaging, or
  output coverage.
- Review Chinese and English separately and report worst-slice behavior.

### Phase 4 — First submission package (July 3-10)

Target: upload a robust baseline as soon as the first round opens on July 10.

Package checklist:

- [ ] checkpoints and all required model assets;
- [ ] Track 1 inference entrypoint;
- [ ] batch input/output adapter matching the official format;
- [ ] reasoning output for every sample;
- [ ] valid waveform for every sample;
- [ ] offline environment installation or container;
- [ ] deterministic launch command;
- [ ] parameter-count report;
- [ ] RTF report and hardware description;
- [ ] external-resource declaration;
- [ ] system description;
- [ ] license notices;
- [ ] clean-room test with network disabled;
- [ ] archive checksum and size.

### Phase 5 — Model selection and final submission (July 10-25)

1. Use first-round feedback to identify validity and bottleneck issues.
2. Freeze candidate systems by July 18.
3. Compare candidates with a predefined weighted local score plus slice and
   failure analysis.
4. Conduct blinded listening and reasoning review.
5. Select one primary and one fallback artifact.
6. Rebuild from a clean checkout with no network.
7. Run full packaging tests and submit by internal deadline July 23.
8. Keep the exact submitted archive immutable.

### Phase 6 — Paper (parallel; complete by August 3)

Document:

- task and constraints;
- end-to-end architecture;
- context and reasoning representation;
- training data and every external resource;
- parameter accounting;
- training and inference details;
- ablations aligned with experiment IDs;
- bilingual/slice results;
- efficiency and failure analysis;
- ethics, licensing, and limitations.

## 6. Experiment Selection Policy

An experiment is eligible to replace the baseline only if:

- all outputs are valid;
- constrained parameters remain `<1B`;
- RTF remains `<=3.0`;
- both languages are evaluated;
- no key slice has an unexplained severe regression;
- reasoning is grounded in the supplied context rather than generic templates;
- speech matches the stated reasoning;
- the artifact is reproducible and offline packageable.

When metrics disagree, prioritize:

1. validity and rule compliance;
2. human-facing contextual appropriateness and speech-reasoning consistency;
3. naturalness and speaker similarity;
4. intelligibility;
5. efficiency margin and robustness.

This ordering reflects the official 50% human + 20% LLM weighting.

## 7. Main Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Registration missed | Register immediately, independently of technical progress |
| “0.6B baseline” exceeds total `<1B` after codec modules | Audit all resident/invoked modules before first submission |
| Official environment file missing | Build and lock a verified server environment on `exp/baseline` |
| Dataset scale overwhelms storage/I/O | Index once, use manifests, staged subsets, cache only derived metadata |
| Data noise and annotation errors | Confidence filtering, slice reports, robust sampling |
| Generic CoT text scores poorly | Context-grounded reasoning templates during training, diversity checks |
| Good reasoning but mismatched speech | Joint evaluation and loss/selection criteria |
| Candidate reranking breaks RTF/params | Start with one candidate; use cheap deterministic checks |
| Offline evaluator misses a dependency | Network-disabled clean-room test |
| Generated output failures | Validation, fallback generation policy, full coverage tests |
| Hidden format mismatch | Build an adapter once official submission schema is published |
| Accidental changes to baseline | Protect `main`; experiments only on `exp/*` |

## 8. Official Links

- Challenge overview:
  https://iscslp2026-cot-tts.github.io/challenge-website/
- Tasks:
  https://iscslp2026-cot-tts.github.io/challenge-website/tasks.html
- Data and models:
  https://iscslp2026-cot-tts.github.io/challenge-website/resources.html
- Rules:
  https://iscslp2026-cot-tts.github.io/challenge-website/rules.html
- Submission:
  https://iscslp2026-cot-tts.github.io/challenge-website/submit.html
- Timeline:
  https://iscslp2026-cot-tts.github.io/challenge-website/timeline.html
- Official baseline:
  https://github.com/iscslp2026-cot-tts/baseline
- Official dataset:
  https://huggingface.co/datasets/HKUSTAudio/ISCSLP2026-CoT-TTS
- Team registration:
  https://docs.google.com/forms/d/e/1FAIpQLSdpfl5xI0nBUQJxdKUBR28gNc4p3qWKz-i7frjGPSULGJuAfw/viewform
