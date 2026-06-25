# Experiments

Experiments live on `exp/*` branches. Do not modify experiment results on
`main`.

Create one directory per experiment:

```text
experiments/
└── E000_official_baseline/
    ├── README.md
    ├── config/
    ├── manifests/
    └── reports/
```

Large outputs, checkpoints, datasets, and generated WAV collections remain
outside Git.

Each experiment README should contain:

```markdown
# E000 — Official Track 1 Baseline

## Hypothesis
## Branch and commit
## Owner and date
## Server and environment
## Data manifest
## Model/resource hashes
## Parameter audit
## Commands
## Metrics and slice results
## RTF and peak VRAM
## Failures
## Conclusion
## Next action
```

Experiment numbering:

- E000: exact official baseline;
- E001-E099: low-risk inference/data/config experiments;
- E100-E199: training and architecture experiments;
- E900+: packaging and submission rehearsals.
