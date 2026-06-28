#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import wave
from pathlib import Path
from typing import Any

import numpy as np


def read_text(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path.read_text(encoding="utf-8").strip()


def load_wav(path: Path) -> tuple[np.ndarray, int]:
    try:
        import soundfile as sf

        audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
        return np.asarray(audio, dtype=np.float32), int(sample_rate)
    except Exception:
        with wave.open(str(path), "rb") as handle:
            sample_rate = handle.getframerate()
            channels = handle.getnchannels()
            sampwidth = handle.getsampwidth()
            frames = handle.readframes(handle.getnframes())
        if sampwidth != 2:
            raise RuntimeError(f"Unsupported WAV sample width without soundfile: {sampwidth}")
        audio_i16 = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
        if channels > 1:
            audio_i16 = audio_i16.reshape(-1, channels).mean(axis=1)
        return audio_i16, int(sample_rate)


def load_pt_summary(path: Path) -> dict[str, int]:
    if not path.is_file():
        raise FileNotFoundError(path)
    import torch

    data = torch.load(str(path), map_location="cpu")
    return {
        "global_ids": int(torch.as_tensor(data.get("global_ids", [])).numel()),
        "semantic_ids": int(torch.as_tensor(data.get("semantic_ids", [])).numel()),
    }


def validate_prefix(prefix: Path, expected_sr: int) -> dict[str, Any]:
    cot_path = prefix.with_suffix(".cot.txt")
    cot_thinking_path = prefix.with_suffix(".cot_thinking.txt")
    raw_path = prefix.with_suffix(".raw.txt")
    wav_path = prefix.with_suffix(".wav")
    meta_path = prefix.with_suffix(".meta.json")
    pt_path = prefix.with_suffix(".pt")

    cot = read_text(cot_path)
    cot_thinking = read_text(cot_thinking_path)
    raw = read_text(raw_path)
    meta = json.loads(read_text(meta_path))
    token_counts = load_pt_summary(pt_path)
    audio, sample_rate = load_wav(wav_path)

    audio = np.asarray(audio)
    duration_sec = float(audio.shape[0] / sample_rate) if audio.ndim == 1 else float(audio.shape[0] / sample_rate)
    finite = bool(np.isfinite(audio).all())
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(audio.astype(np.float64))))) if audio.size else 0.0

    checks = {
        "cot_nonempty": bool(cot),
        "cot_thinking_nonempty": bool(cot_thinking),
        "raw_nonempty": bool(raw),
        "wav_exists": wav_path.is_file(),
        "sample_rate_expected": sample_rate == expected_sr,
        "duration_positive": duration_sec > 0.0,
        "finite": finite,
        "not_all_zero": peak > 0.0,
        "not_near_silent": rms > 1e-5 and peak > 1e-4,
        "semantic_tokens_positive": token_counts["semantic_ids"] > 0,
        "meta_semantic_tokens_positive": int(meta.get("generated_semantic_tokens", 0)) > 0,
        "global_source_ref": meta.get("global_source") == "ref",
    }
    ok = all(checks.values())
    return {
        "ok": ok,
        "prefix": str(prefix),
        "files": {
            "cot": str(cot_path),
            "cot_thinking": str(cot_thinking_path),
            "raw": str(raw_path),
            "wav": str(wav_path),
            "meta": str(meta_path),
            "pt": str(pt_path),
        },
        "checks": checks,
        "audio": {
            "sample_rate": sample_rate,
            "duration_sec": duration_sec,
            "samples": int(audio.shape[0]) if audio.ndim else int(audio.size),
            "peak_abs": peak,
            "rms": rms,
            "finite": finite,
        },
        "tokens": token_counts,
        "meta": {
            "generated_semantic_tokens": meta.get("generated_semantic_tokens"),
            "generated_global_tokens": meta.get("generated_global_tokens"),
            "global_source": meta.get("global_source"),
            "parse_source": meta.get("parse_source"),
            "num_candidates": meta.get("num_candidates"),
            "rerank_metric": meta.get("rerank_metric"),
        },
        "cot_preview": cot[:500],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Track 1 reasoning/WAV output for one output prefix.")
    parser.add_argument("--prefix", required=True, type=Path)
    parser.add_argument("--expected-sr", type=int, default=16000)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    report = validate_prefix(args.prefix, args.expected_sr)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text + "\n", encoding="utf-8")
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
