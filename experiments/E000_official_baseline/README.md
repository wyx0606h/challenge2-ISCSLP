# E000 — Official Track 1 Baseline

## Hypothesis

The official Track 1 text-history baseline can run end-to-end on the rented GPU
server with the released sample case, producing non-empty reasoning and a valid
WAV while recording reproducibility evidence.

## Branch and Commit

- Branch: `exp/baseline`
- Commit: `b05eebdb40865984f2ceb4132d76aa0ccc149ce2`

## Server and Environment

- GPU: NVIDIA GeForce RTX 4090D, 24564 MiB
- Driver/CUDA from `nvidia-smi`: 550.107.02 / CUDA 12.4
- Runtime torch: 2.1.2+cu121
- Python env: `/root/autodl-tmp/iscslp2026/envs/e000`
- Environment locks: `env/requirements-e000.txt`, `env/environment-e000.yml`
- Full external records: `/root/autodl-tmp/iscslp2026/artifacts/env/`

## Inputs

Official sample:

```text
infer/cases/sample_case/history.txt
infer/cases/sample_case/reference.wav
infer/cases/sample_case/target.txt
```

The sample has no real `target.wav` and no gold CoT label, so only output
validity and proxy checks are reported.

## Command

```bash
cd /root/challenge2-ISCSLP
PATH=/root/autodl-tmp/iscslp2026/envs/e000/bin:$PATH \
ISCSLP_ROOT=/root/autodl-tmp/iscslp2026 \
CUDA_VISIBLE_DEVICES=0 \
scripts/run_e000_sample.sh
```

## Output Validation

Validation report:

```text
/root/autodl-tmp/iscslp2026/artifacts/metrics/E000_sample/validation.json
```

Result:

- validation passed;
- CoT files are non-empty;
- WAV is readable, finite, non-zero, and not near-silent;
- sample rate: 16000 Hz;
- duration: 1.96 s;
- semantic tokens: 98;
- global tokens: 32;
- `global_source`: `ref`.

## RTF and VRAM

Report:

```text
/root/autodl-tmp/iscslp2026/artifacts/metrics/E000_sample/rtf_vram_summary.json
```

Current subprocess cold-start result:

- elapsed: 76.801 s;
- generated audio duration: 1.96 s;
- cold-start RTF: 39.184;
- peak VRAM by `nvidia-smi` sampling: 4595 MiB.

True warm-model steady-state RTF is not yet measured.

## Parameter Audit

Report:

```text
/root/autodl-tmp/iscslp2026/artifacts/metrics/E000_sample/params.json
```

Current loaded/invoked total:

```text
1,091,463,427 parameters
```

This exceeds the `<1B` constrained-category limit under the current counting
policy.

## Artifacts

```text
/root/autodl-tmp/iscslp2026/artifacts/logs/E000_sample/
/root/autodl-tmp/iscslp2026/artifacts/outputs/E000_sample/
/root/autodl-tmp/iscslp2026/artifacts/metrics/E000_sample/
/root/autodl-tmp/iscslp2026/artifacts/hashes/
```

## Conclusion

The official sample now runs end-to-end and produces valid reasoning/audio.
E000 is not complete as a constrained-category reproduction because the current
parameter audit exceeds 1B and true steady-state RTF remains to be measured.
