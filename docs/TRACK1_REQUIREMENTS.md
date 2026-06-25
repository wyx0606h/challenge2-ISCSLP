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
