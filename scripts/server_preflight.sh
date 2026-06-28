#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ISCSLP_ROOT="${ISCSLP_ROOT:-/root/autodl-tmp/iscslp2026}"
ISCSLP_MODELS="${ISCSLP_MODELS:-$ISCSLP_ROOT/models}"
ISCSLP_ARTIFACTS="${ISCSLP_ARTIFACTS:-$ISCSLP_ROOT/artifacts}"

mkdir -p "$ISCSLP_ARTIFACTS/env"

{
  echo "===== date ====="
  date -Is
  echo
  echo "===== git ====="
  cd "$REPO_DIR"
  git branch --show-current
  git rev-parse HEAD
  git status --short --branch
  echo
  echo "===== paths ====="
  printf 'REPO_DIR=%s\n' "$REPO_DIR"
  printf 'ISCSLP_ROOT=%s\n' "$ISCSLP_ROOT"
  printf 'ISCSLP_MODELS=%s\n' "$ISCSLP_MODELS"
  printf 'ISCSLP_ARTIFACTS=%s\n' "$ISCSLP_ARTIFACTS"
  echo
  echo "===== disk ====="
  df -h "$REPO_DIR" "$ISCSLP_ROOT" /root || true
  echo
  echo "===== gpu ====="
  nvidia-smi || true
  echo
  echo "===== python ====="
  which python || true
  python --version || true
  python - <<'PY' || true
import importlib.util
mods = [
    "torch", "torchaudio", "transformers", "safetensors", "soundfile",
    "soxr", "omegaconf", "yaml", "flash_attn", "numpy",
]
for name in mods:
    print(f"{name}: {importlib.util.find_spec(name) is not None}")
try:
    import torch
    print("torch_version:", torch.__version__)
    print("torch_cuda:", torch.version.cuda)
    print("cuda_available:", torch.cuda.is_available())
except Exception as exc:
    print("torch_probe_error:", repr(exc))
PY
  echo
  echo "===== model inventory ====="
  find "$ISCSLP_MODELS" -maxdepth 4 -type f -printf '%p\t%s\n' 2>/dev/null | sort || true
} | tee "$ISCSLP_ARTIFACTS/env/preflight.txt"
