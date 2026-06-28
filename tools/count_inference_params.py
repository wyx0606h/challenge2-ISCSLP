#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
INFER_DIR = REPO_ROOT / "infer"
for path in (REPO_ROOT, INFER_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def module_summary(name: str, module: torch.nn.Module) -> dict[str, Any]:
    seen: set[int] = set()
    total = 0
    trainable = 0
    memory_bytes = 0
    dtypes: set[str] = set()
    devices: set[str] = set()
    for param in module.parameters(recurse=True):
        key = param.untyped_storage().data_ptr()
        if key in seen:
            continue
        seen.add(key)
        n = int(param.numel())
        total += n
        if param.requires_grad:
            trainable += n
        memory_bytes += n * param.element_size()
        dtypes.add(str(param.dtype))
        devices.add(str(param.device))
    return {
        "component": name,
        "total_params": total,
        "trainable_params": trainable,
        "memory_bytes": memory_bytes,
        "dtypes": sorted(dtypes),
        "devices": sorted(devices),
        "invoked": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Count Track 1 inference-time parameters after loading invoked modules.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--spark-model-dir", required=True, type=Path)
    parser.add_argument("--model-architecture", choices=["auto", "full", "lora"], default="lora")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--torch-dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    import cot_tts_text_history_inference as infer
    from sparktts.models.audio_tokenizer import BiCodecTokenizer

    ns = argparse.Namespace(
        checkpoint_path=str(args.checkpoint),
        run_dir=str(args.checkpoint.parent.parent),
        hf_model_dir="",
        tokenizer_path="",
        auto_convert_dcp=False,
        model_architecture=args.model_architecture,
        torch_dtype=args.torch_dtype,
        attn_implementation=args.attn_implementation,
    )
    device = infer.resolve_device(args.device)
    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device)
    hf_dir = infer.ensure_hf_model_dir(ns, args.checkpoint)
    tokenizer_dir = infer.resolve_tokenizer_dir(ns, Path(ns.run_dir), hf_dir)
    _, model, loaded_architecture, detected_lora = infer.load_text_model_and_tokenizer(
        args=ns,
        checkpoint_path=args.checkpoint,
        run_dir=Path(ns.run_dir),
        hf_dir=hf_dir,
        tokenizer_dir=tokenizer_dir,
        device=device,
    )
    audio_tokenizer = BiCodecTokenizer(args.spark_model_dir, device=device)

    components = [
        module_summary(f"cot_tts_text_model_{loaded_architecture}", model),
        module_summary("spark_bicodec", audio_tokenizer.model),
        module_summary("spark_wav2vec2_feature_extractor", audio_tokenizer.feature_extractor),
    ]
    report = {
        "checkpoint": str(args.checkpoint),
        "spark_model_dir": str(args.spark_model_dir),
        "model_architecture": loaded_architecture,
        "detected_lora": detected_lora,
        "components": components,
        "total_inference_params": sum(item["total_params"] for item in components),
        "limit": 1_000_000_000,
        "under_1b": sum(item["total_params"] for item in components) < 1_000_000_000,
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text + "\n", encoding="utf-8")
    if not report["under_1b"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
