#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import torch


def find_latest_checkpoint(run_dir: Path) -> Path:
    ckpt_root = run_dir / "checkpoints"
    if not ckpt_root.exists():
        raise FileNotFoundError(f"checkpoints dir not found: {ckpt_root}")

    cands: list[tuple[int, Path]] = []
    for path in ckpt_root.iterdir():
        if not path.is_dir():
            continue
        match = re.match(r"global_step_(\d+)$", path.name)
        if match and ((path / ".metadata").exists() or (path / "hf_ckpt" / "config.json").exists()):
            cands.append((int(match.group(1)), path))
    if not cands:
        raise FileNotFoundError(f"No usable global_step_* checkpoint under {ckpt_root}")
    cands.sort(key=lambda item: item[0])
    return cands[-1][1]


def resolve_run_dir(args: argparse.Namespace, checkpoint_path: Path) -> Path:
    if getattr(args, "run_dir", ""):
        return Path(args.run_dir)
    if checkpoint_path.parent.name != "checkpoints":
        raise ValueError("--run_dir is required when checkpoint_path is not under <run_dir>/checkpoints/")
    return checkpoint_path.parent.parent


def resolve_lora_hf_dir(args: argparse.Namespace, checkpoint_path: Path) -> Path:
    hf_dir = Path(args.hf_model_dir) if getattr(args, "hf_model_dir", "") else checkpoint_path / "hf_ckpt"
    if (hf_dir / "config.json").exists() and (hf_dir / "model.safetensors").exists():
        return hf_dir

    if not getattr(args, "auto_convert_dcp", True):
        raise FileNotFoundError(
            f"LoRA HF checkpoint not found: {hf_dir}. Provide --hf_model_dir or enable --auto_convert_dcp."
        )

    # Reuse the robust absolute-path loader from the main inference script.
    from cot_tts_inference import load_merge_to_hf_pt

    model_assets_dir = Path(args.run_dir) / "model_assets"
    if not model_assets_dir.exists():
        raise FileNotFoundError(f"model_assets dir not found: {model_assets_dir}")

    hf_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Converting LoRA DCP checkpoint to HF: {checkpoint_path} -> {hf_dir}")
    merge_to_hf_pt = load_merge_to_hf_pt()
    merge_to_hf_pt(load_dir=str(checkpoint_path), save_path=str(hf_dir), model_assets_dir=str(model_assets_dir))
    return hf_dir


def resolve_tokenizer_dir(args: argparse.Namespace, run_dir: Path, hf_dir: Path) -> Path:
    candidates = []
    if getattr(args, "tokenizer_path", ""):
        candidates.append(Path(args.tokenizer_path))
    candidates.extend([hf_dir, run_dir / "model_assets"])
    for candidate in candidates:
        if (candidate / "tokenizer.json").exists() and (candidate / "tokenizer_config.json").exists():
            return candidate
    raise FileNotFoundError("Cannot find tokenizer dir. Checked: " + ", ".join(str(p) for p in candidates))


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        import yaml
    except Exception as exc:
        raise RuntimeError("PyYAML is required to read veomni_cli.yaml for LoRA config.") from exc
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def lora_config_from_run(run_dir: Path) -> dict[str, Any]:
    cfg = load_yaml(run_dir / "veomni_cli.yaml")
    model_cfg = cfg.get("model", {}) if isinstance(cfg.get("model"), dict) else {}
    return {
        "lora_rank": int(model_cfg.get("lora_rank", 8)),
        "lora_alpha": int(model_cfg.get("lora_alpha", 16)),
        "lora_target_modules": str(
            model_cfg.get("lora_target_modules", "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
        ),
        "lora_target_modules_support": str(
            model_cfg.get(
                "lora_target_modules_support",
                "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
            )
        ),
        "init_lora_weights": str(model_cfg.get("init_lora_weights", "kaiming")),
    }


def load_lora_model_and_tokenizer(
    *,
    args: argparse.Namespace,
    checkpoint_path: Path,
    run_dir: Path,
    hf_dir: Path,
    tokenizer_dir: Path,
    device: torch.device,
):
    from cot_tts_inference import install_sklearn_stub

    install_sklearn_stub()

    from safetensors.torch import load_file
    from veomni.models import build_foundation_model, build_tokenizer
    from veomni.utils.lora_utils import add_lora_to_model

    tokenizer = build_tokenizer(str(tokenizer_dir))
    model = build_foundation_model(
        config_path=str(hf_dir),
        weights_path=None,
        torch_dtype=args.torch_dtype,
        attn_implementation=args.attn_implementation,
        init_device="cpu",
    )

    lora_cfg = lora_config_from_run(run_dir)
    lora_model = add_lora_to_model(
        model,
        lora_rank=lora_cfg["lora_rank"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_target_modules=lora_cfg["lora_target_modules"],
        init_lora_weights=lora_cfg["init_lora_weights"],
        pretrained_lora_path=None,
        lora_target_modules_support=lora_cfg["lora_target_modules_support"].split(","),
    )
    if lora_model is not None:
        model = lora_model

    state_path = hf_dir / "model.safetensors"
    if not state_path.exists():
        raise FileNotFoundError(f"LoRA model.safetensors not found: {state_path}")
    state_dict = load_file(str(state_path), device="cpu")
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    real_missing = [
        key
        for key in missing_keys
        if "rotary_emb.inv_freq" not in key and not key.endswith(".base_layer.bias")
    ]
    print(
        f"[INFO] Loaded LoRA checkpoint: {state_path} "
        f"missing={len(missing_keys)} unexpected={len(unexpected_keys)} real_missing={len(real_missing)}"
    )
    if unexpected_keys[:10]:
        print("[WARN] unexpected keys sample:", unexpected_keys[:10])
    if real_missing[:10]:
        print("[WARN] missing keys sample:", real_missing[:10])

    return tokenizer, model.eval().to(device)


def select_global_ids(
    *,
    mode: str,
    generated_global_ids: list[int],
    ref_global_ids: list[int],
    global_prefix_len: int,
) -> tuple[list[int], str]:
    if mode == "generated" and generated_global_ids:
        return generated_global_ids[:global_prefix_len], "generated"
    if not ref_global_ids:
        raise RuntimeError("No reference global tokens available for waveform decoding.")
    return ref_global_ids[:global_prefix_len], "ref"


def save_lora_run_summary(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
