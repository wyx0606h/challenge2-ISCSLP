# Track 1 Requirement Matrix

Verified on 2026-06-25.

| Requirement | Project interpretation | Evidence required |
|---|---|---|
| Text dialogue context | Speaker-labeled ordered dialogue history | Input adapter test |
| Target text | Synthesize exactly the requested content | WER/CER and sample audit |
| Reference speech | Preserve reference speaker timbre | Reference-path manifest and similarity proxy |
| Reasoning output | One contextual, coherent, informative analysis per sample | Non-empty/grounding checks and review |
| Speech output | One valid waveform per sample | Audio validator and coverage report |
| Parameter-constrained | `<1B` total inference-time parameters | Per-component audited report |
| Efficiency | Official RTF `<=3.0` | Timed run with hardware/config |
| No cascade | No separate ASR-LLM-TTS chain | Architecture declaration/code review |
| Offline inference | No APIs or remote model calls | Network-disabled clean-room run |
| Complete package | Models, code, runtime config, system description, dependencies | Submission checklist |
| External resources | Everything declared | Resource manifest |
| Hidden-data integrity | No lookup/reconstruction/source tracing | Team policy and code review |
| Validity | No missing files, bad formats, incomplete generation, or severe mismatch | Full-package validator |

## Evaluation Harness TODO

Build local evaluation scripts that are deliberately aligned with Track 1
rather than generic TTS benchmarking. The scripts should support both the
official baseline and later `exp/*` systems without changing the frozen
validation manifests.

Required capabilities:

- load a manifest containing ordered speaker-labeled dialogue history, target
  text, reference speech, expected language, and slice tags;
- run the Track 1 inference entrypoint offline and write one reasoning file and
  one WAV file per sample;
- validate 100% output coverage, non-empty sample-specific reasoning, readable
  finite audio, expected sample rate, plausible duration, and no missing token
  artifacts;
- compute Chinese CER and English WER proxies separately;
- compute UTMOSv2 and DNSMOS P.835 speech-quality proxies where local declared
  evaluators are available;
- compute speaker similarity, F0 correlation, emotion expressiveness, duration
  error, speaking-rate, energy, pause, and style/prosody proxy features where
  local models or deterministic DSP are available;
- score reasoning specificity, context grounding, coherence, non-template
  behavior, and whether the synthesized speech appears consistent with the
  stated reasoning;
- measure steady-state and end-to-end RTF with GPU synchronization, generated
  audio duration, peak VRAM, batch size, precision, and hardware recorded;
- audit inference-time parameters for every loaded or invoked module and fail
  the run if the total reaches `>=1B`;
- report bilingual and worst-slice behavior across language, context length,
  expressive intensity, noise, duration, and reference quality;
- emit machine-readable JSON/CSV plus a compact Markdown summary suitable for
  experiment records.

The local score should mirror the official emphasis: validity and rule
compliance first, then human-facing context/style appropriateness and
speech-reasoning consistency, then naturalness, speaker similarity,
intelligibility, and efficiency. Automatic metrics are iteration aids only; do
not present them as official leaderboard-equivalent scores.

Planned reported metrics:

- UTMOSv2;
- DNSMOS P.835;
- Chinese CER and English WER;
- speaker similarity;
- F0 correlation;
- emotion expressiveness;
- duration error;
- RTF;
- LLM score;
- human score.

## Submission Description Must Cover

- selected track and category;
- model architecture;
- total and trainable parameter counts;
- training data and splits;
- every external dataset/model/resource;
- synthetic data and augmentation;
- preprocessing and post-processing;
- training procedure;
- inference and decoding;
- runtime environment;
- RTF hardware and measurement;
- limitations and ethical use.

## Dataset Usage Constraints

The official dataset is for non-commercial academic research, development,
and evaluation related to context-aware speech generation and CoT-guided TTS.

Do not:

- redistribute, re-host, sell, sublicense, or repackage it;
- reconstruct or redistribute original source media;
- identify speakers, actors, characters, titles, or rights holders;
- use it for impersonation, deception, surveillance, speaker identification,
  biometric profiling, or privacy-invasive analysis.

Expect:

- varied sampling rates, channels, loudness, noise, music, and overlap;
- automatic transcript, diarization, timing, emotion, and CoT annotation
  errors;
- media-domain style bias rather than purely natural daily conversation.
