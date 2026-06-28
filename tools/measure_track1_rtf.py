#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

from validate_track1_outputs import load_wav


def run_once(command: list[str], prefix: Path, expected_sr: int) -> dict[str, Any]:
    start = time.perf_counter()
    proc = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    elapsed = time.perf_counter() - start
    output = proc.stdout
    if proc.returncode != 0:
        return {"ok": False, "returncode": proc.returncode, "elapsed_sec": elapsed, "log_tail": output[-4000:]}
    wav, sr = load_wav(prefix.with_suffix(".wav"))
    duration = float(wav.shape[0] / sr)
    return {
        "ok": True,
        "returncode": 0,
        "elapsed_sec": elapsed,
        "audio_duration_sec": duration,
        "sample_rate": sr,
        "rtf": elapsed / duration if duration > 0 else None,
        "sample_rate_expected": sr == expected_sr,
        "log_tail": output[-4000:],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure subprocess-level Track 1 RTF for an inference command.")
    parser.add_argument("--command-file", required=True, type=Path, help="Text file containing argv separated by NUL or newline.")
    parser.add_argument("--prefix", required=True, type=Path)
    parser.add_argument("--expected-sr", type=int, default=16000)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()

    raw = args.command_file.read_text(encoding="utf-8")
    command = [part for part in raw.replace("\0", "\n").splitlines() if part]
    if not command:
        raise SystemExit("empty command file")

    warmups = [run_once(command, args.prefix, args.expected_sr) for _ in range(args.warmup)]
    runs = [run_once(command, args.prefix, args.expected_sr) for _ in range(args.repeat)]
    ok_runs = [run for run in runs if run.get("ok") and run.get("rtf") is not None]
    rtfs = [float(run["rtf"]) for run in ok_runs]
    durations = [float(run["audio_duration_sec"]) for run in ok_runs]
    elapsed = [float(run["elapsed_sec"]) for run in ok_runs]
    report = {
        "mode": "subprocess",
        "note": "This includes process startup and model load. Use it as cold-start RTF unless command itself keeps models warm.",
        "warmup": warmups,
        "runs": runs,
        "summary": {
            "ok_runs": len(ok_runs),
            "total_runs": len(runs),
            "aggregate_rtf": sum(elapsed) / sum(durations) if durations and sum(durations) > 0 else None,
            "median_rtf": statistics.median(rtfs) if rtfs else None,
            "p90_rtf": statistics.quantiles(rtfs, n=10)[8] if len(rtfs) >= 10 else (max(rtfs) if rtfs else None),
            "total_elapsed_sec": sum(elapsed) if elapsed else None,
            "total_audio_duration_sec": sum(durations) if durations else None,
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    if len(ok_runs) != len(runs):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
