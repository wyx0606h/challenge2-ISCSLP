# E000 Sample Result

## Output

- Prefix: `/root/autodl-tmp/iscslp2026/artifacts/outputs/E000_sample/text_history_demo`
- Reasoning: non-empty
- WAV: valid
- Sample rate: 16000 Hz
- Duration: 1.96 s
- Semantic tokens: 98
- Global tokens: 32
- Global token source: reference audio

## Runtime

- Subprocess elapsed time: 76.801 s
- Subprocess cold-start RTF: 39.184
- Peak VRAM sampled by `nvidia-smi`: 4595 MiB
- True in-process steady-state RTF: not measured in this run

## Parameter Audit

```text
cot_tts_text_model_lora:          619,708,416
spark_bicodec:                    156,316,291
spark_wav2vec2_feature_extractor: 315,438,720
total:                          1,091,463,427
```

The current total exceeds the `<1B` constrained-category limit.

## Evidence

```text
/root/autodl-tmp/iscslp2026/artifacts/metrics/E000_sample/validation.json
/root/autodl-tmp/iscslp2026/artifacts/metrics/E000_sample/params.json
/root/autodl-tmp/iscslp2026/artifacts/metrics/E000_sample/rtf_vram_summary.json
/root/autodl-tmp/iscslp2026/artifacts/logs/E000_sample/timed_with_vram.log
```
