# Branching Strategy

## Branch Relationship

```text
official baseline repository @ 46cd111
                  |
                  v
main  (frozen project root; baseline + governance documents)
  |
  +--> exp/baseline  (first server reproduction and baseline measurements)
          |
          +--> exp/decoding
          +--> exp/context-format
          +--> exp/ref-audio
          +--> exp/data-filter
          +--> exp/lora
          +--> exp/reasoning-audio
          +--> exp/rtf
```

## `main`

`main` is a reference, not a development branch.

Allowed:

- initial official baseline import;
- initial project policy and challenge documents;
- an explicitly approved refresh from a newer official baseline release.

Not allowed:

- experiments;
- server-specific fixes;
- dependency pinning discovered during experiments;
- training configs;
- checkpoint changes;
- metric-driven refactors.

If a future official baseline update is imported, tag the old root first and
document the upstream commit and migration impact.

Recommended protection in GitHub:

- block force pushes and deletions;
- require pull requests;
- require at least one approval;
- require branch to be up to date;
- restrict direct pushes;
- add a CODEOWNERS rule for root governance files if the team grows.

## `exp/baseline`

This branch starts as an exact copy of initialized `main`. It is the first
mutable branch and owns:

- environment reconstruction;
- path portability fixes;
- download/manifest helpers;
- parameter counting;
- RTF measurement;
- input/output validation;
- official sample and local validation reproduction;
- baseline experiment record.

Do not add speculative model changes until the baseline exit gate is met.

## Future `exp/*` Branches

Branch from the reproduced `exp/baseline` commit:

```bash
git switch exp/baseline
git pull --ff-only
git switch -c exp/context-format
```

Rules:

- one primary hypothesis per branch;
- use short, descriptive names;
- include an experiment record;
- keep commits small and interpretable;
- do not merge unrelated experiments together;
- promote by cherry-picking or by creating a dedicated integration branch;
- never merge experiments into `main`.

Suggested later integration branch:

```text
system/track1-constrained
```

That branch should contain only selected, rule-compliant improvements and the
submission package. It is not created during initial setup because selection
must follow baseline reproduction.

## Tags

Suggested tags:

- `baseline/import-2026-06-25`
- `baseline/reproduced-v1`
- `submission/round1`
- `submission/final`

Tags should point to immutable, documented commits. Store archive checksums in
the corresponding experiment/submission record.
