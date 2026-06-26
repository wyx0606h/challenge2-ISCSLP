# Track 1 Evaluation Harness TODO

This repository currently imports the official baseline, but it does not yet
contain a complete local evaluation harness. Build one before judging baseline
quality or comparing improved systems.

The goal is not to reproduce the hidden official evaluator. The goal is to
create a strict, repeatable, Track-1-aware local gate that catches invalid
submissions early and makes baseline-vs-improvement comparisons meaningful.

## 1. Scope

The harness must evaluate systems for **Track 1: Text-Context-Aware CoT-TTS**
in the **Parameter-Constrained** category.

It must verify that each sample uses:

- ordered speaker-labeled text dialogue history;
- target text;
- reference speech;
- one generated reasoning analysis;
- one generated speech waveform.

It must reject or clearly fail runs that violate:

- offline inference requirements;
- `<1B` total inference-time parameter limit;
- RTF `<=3.0`;
- 100% reasoning and waveform output coverage;
- hidden-data integrity and non-identification rules.

## 2. Frozen Manifests

Create deterministic JSONL manifests rather than ad-hoc file lists.

Minimum fields:

- `sample_id`;
- `language` (`zh`, `en`, or explicitly documented mixed labels);
- `history_text_path` or embedded normalized history text;
- `target_text_path` or embedded target text;
- `reference_audio_path`;
- optional `target_audio_path` for local dev data with ground truth;
- optional `cot_text` for supervised diagnostics only;
- duration, context length, speaker-turn count, reference duration, and source
  split metadata;
- slice tags for language, short/long context, expressive intensity, noise,
  duration, reference quality, and speaker/reference consistency.

Required manifests:

- `track1_sample.jsonl`: official sample-case smoke test.
- `track1_fast_dev.jsonl`: 20-50 bilingual examples for every iteration.
- `track1_selection_dev.jsonl`: larger frozen bilingual set for candidate
  selection.
- Optional `track1_stress_dev.jsonl`: long context, noisy reference, high
  emotion/style, and edge-duration cases.

Never tune on hidden evaluation inputs. Do not include source titles, speaker
identities, or reconstructable source-media hints in reports.

## 3. Output Validator

Implement a validator that fails fast on:

- missing reasoning file;
- empty, generic, or repeated-placeholder reasoning;
- missing WAV file;
- unreadable audio;
- NaN/Inf samples;
- all-zero or near-silent output;
- unexpected sample rate unless explicitly converted;
- implausible duration relative to target text;
- missing or malformed metadata;
- incomplete batch coverage.

The validator should emit both per-sample status and aggregate coverage.
Coverage below 100% is a hard failure for baseline reproduction and candidate
selection.

## 4. Rule Compliance Audits

### Parameter count

Count every loaded or invoked inference module:

- CoT-TTS/text backbone;
- LoRA/adapters;
- Spark-TTS/BiCodec tokenizer, decoder, and vocoder components;
- speaker/reference encoders;
- rerankers, denoisers, enhancers, post-processors, or quality models if added.

Report per component:

- total parameters;
- trainable parameters;
- dtype;
- device;
- memory estimate;
- whether the module is invoked during inference.

The run should fail if the deduplicated total is `>=1,000,000,000`.

### RTF

Report both:

- steady-state RTF excluding model load;
- end-to-end RTF including process launch/model load.

Record hardware, GPU driver, CUDA, PyTorch, precision, batch size, generated
duration, wall-clock time, failures/retries, peak VRAM, median RTF, p90 RTF,
and aggregate RTF.

Use GPU synchronization around timed sections.

### Offline execution

Provide a clean-room test mode that disables network access after assets are
staged and confirms all required resources are local.

## 5. Objective Proxy Metrics

Automatic metrics are iteration aids, not official-score claims.

Implement and report:

- Chinese CER proxy;
- English WER proxy;
- language-separated and worst-language summaries;
- duration, speaking rate, pause ratio, F0, energy, and simple prosody
  statistics;
- speech-quality and speaker-similarity proxies if local, declared models are
  available;
- failure rates and latency/VRAM metrics.

When an auxiliary model is used only for offline evaluation, record it in the
experiment resource manifest. If the same model is loaded during inference, it
counts toward the `<1B` parameter limit.

## 6. Reasoning and Context Checks

Track 1 is not generic TTS. The reasoning output is part of the submission and
should be evaluated directly.

Add checks for:

- non-empty reasoning;
- sample specificity;
- grounding in dialogue history;
- target-speaker and target-text awareness;
- emotion/style/prosody intent;
- coherence and internal consistency;
- non-template behavior across a batch;
- absence of forbidden source identification or hidden-media reconstruction.

Add a small human-review sheet or Markdown report for cases where automatic
checks are unreliable.

## 7. Speech-Reasoning Consistency

For each sample, compare the stated reasoning with measurable or reviewable
speech properties:

- speaking rate vs stated urgency/calmness;
- energy/loudness vs stated intensity;
- pitch/prosody movement vs stated emotion;
- pauses vs stated hesitation or narrative pacing;
- reference timbre preservation vs stated speaker identity/timbre objective.

The harness should flag contradictions for manual review. This is especially
important because the official evaluation weights human and LLM judgment
heavily.

## 8. Slice Reports

Every run summary should include:

- overall scorecard;
- Chinese-only and English-only summaries;
- worst-slice table;
- short vs long context;
- low vs high expressive intensity;
- clean vs noisy/reference-challenging cases;
- short vs long generated utterances;
- failures grouped by cause.

Prefer worst-slice and regression analysis over a single averaged score.

## 9. Candidate Comparison

Baseline and improved systems must be compared with:

- same manifest;
- same metric code version;
- same decoding policy unless the experiment is explicitly about decoding;
- same hardware class for RTF comparisons where possible;
- same output validator and failure policy.

The comparison report should include:

- branch and commit;
- checkpoint hash;
- decoding settings;
- total inference parameters;
- RTF;
- output coverage;
- bilingual metrics;
- reasoning metrics;
- speech-reasoning consistency findings;
- keep/reject/investigate conclusion.

## 10. Suggested Implementation Milestones

1. `tools/build_track1_manifest.py`: build deterministic JSONL manifests.
2. `tools/validate_track1_outputs.py`: validate reasoning/WAV coverage and
   audio sanity.
3. `tools/count_inference_params.py`: audit loaded modules and `<1B` gate.
4. `tools/measure_track1_rtf.py`: timed batch runner with GPU sync.
5. `tools/eval_track1.py`: unified runner that calls inference, validation,
   parameter audit, RTF, metrics, and report generation.
6. `tools/compare_track1_runs.py`: baseline-vs-candidate Markdown/CSV report.
7. `experiments/E000_baseline_reproduction/`: first full record using the
   official baseline, sample case, and frozen bilingual fast-dev manifest.

Do not change the official baseline command while implementing these tools.
Wrap it or call it from the evaluator so that the exact baseline command remains
available for comparison.
