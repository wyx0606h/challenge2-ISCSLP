#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.machinery
import importlib.util
import json
import os
import re
import sys
import tempfile
import types
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch

VEOMNI_ROOT = Path(__file__).resolve().parent.parent
if str(VEOMNI_ROOT) not in sys.path:
    sys.path.insert(0, str(VEOMNI_ROOT))

DEFAULT_RUN_DIR = "model/cot_tts_audio_history_baseline"
DEFAULT_CHECKPOINT_PATH = f"{DEFAULT_RUN_DIR}/checkpoints/global_step_37000"
DEFAULT_SPARK_MODEL_DIR = "model/Spark-TTS"
DEFAULT_CASES_DIR = "infer/cases"
DEFAULT_OUTPUT_DIR = "infer/results"
DEFAULT_CASE_NAMES = ["sample_case"]
AUDIO_SUFFIXES = (".wav", ".mp3", ".flac", ".m4a", ".ogg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch COT-TTS inference for named sample cases with Spark BiCodec.")
    parser.add_argument("--run_dir", type=str, default=DEFAULT_RUN_DIR, help="Run dir with checkpoints/ and veomni_cli.yaml.")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=DEFAULT_CHECKPOINT_PATH,
        help="Checkpoint dir. Empty means latest under run_dir/checkpoints.",
    )
    parser.add_argument("--hf_model_dir", type=str, default="", help="HF dir. Empty means <checkpoint_path>/hf_ckpt.")
    parser.add_argument("--tokenizer_path", type=str, default="", help="Tokenizer dir. Empty means auto resolve.")
    parser.add_argument("--auto_convert_dcp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--model_architecture",
        choices=["auto", "full", "lora"],
        default="auto",
        help="auto detects LoRA-style HF checkpoints with lora_A/lora_B weights.",
    )

    parser.add_argument("--cases_dir", type=str, default=DEFAULT_CASES_DIR)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--case_names", nargs="*", default=DEFAULT_CASE_NAMES)
    parser.add_argument(
        "--history_glob",
        type=str,
        default="",
        help=(
            "Optional history glob template relative to each case dir, e.g. 'history/*.wav' or 'history_*.wav'. "
            "Default auto-detects history.wav, history_*.wav, and history/* audio files."
        ),
    )
    parser.add_argument(
        "--ref_template",
        type=str,
        default="reference.wav",
        help="Reference audio template relative to each case dir.",
    )
    parser.add_argument(
        "--text_template",
        type=str,
        default="target.txt",
        help="Target text template relative to each case dir.",
    )
    parser.add_argument(
        "--history_mode",
        choices=["full", "semantic", "full_first_then_semantic"],
        default="full",
        help="How to put history audio into <audio_his_start>. full means global+semantic for every history file.",
    )

    parser.add_argument("--spark_model_dir", type=str, default=DEFAULT_SPARK_MODEL_DIR)
    parser.add_argument("--global_prefix", type=str, default="bicodec_global")
    parser.add_argument("--semantic_prefix", type=str, default="bicodec_semantic")
    parser.add_argument("--global_prefix_len", type=int, default=32)

    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--torch_dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample_rate", type=int, default=16000)

    parser.add_argument("--max_new_tokens", type=int, default=2000)
    parser.add_argument("--min_new_tokens", type=int, default=256)
    parser.add_argument("--do_sample", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.75)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=0)
    parser.add_argument(
        "--num_candidates",
        type=int,
        default=1,
        help="Generate N candidates per case and select one. 1 keeps the fast path.",
    )
    parser.add_argument(
        "--rerank_metric",
        choices=["none", "mel_similarity"],
        default="none",
        help="mel_similarity chooses the candidate closest to ref audio by lightweight log-mel stats.",
    )

    parser.add_argument("--max_semantic_tokens", type=int, default=0)
    parser.add_argument("--use_ref_global_tokens", action="store_true")
    return parser.parse_args()


def resolve_device(device_str: str) -> torch.device:
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_str)


def resolve_run_dir(args: argparse.Namespace, checkpoint_path: Path) -> Path:
    if args.checkpoint_path and not args.run_dir:
        if checkpoint_path.parent.name != "checkpoints":
            raise ValueError("--run_dir is required when checkpoint_path is not under <run_dir>/checkpoints/")
        return checkpoint_path.parent.parent
    if args.checkpoint_path and args.run_dir == DEFAULT_RUN_DIR:
        if checkpoint_path.parent.name == "checkpoints":
            return checkpoint_path.parent.parent
    return Path(args.run_dir)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def ensure_hf_model_dir(args: argparse.Namespace, checkpoint_path: Path) -> Path:
    hf_dir = Path(args.hf_model_dir) if args.hf_model_dir else checkpoint_path / "hf_ckpt"
    if (hf_dir / "config.json").exists():
        return hf_dir

    if not args.auto_convert_dcp:
        raise FileNotFoundError(
            f"HF model dir not found: {hf_dir}. Provide --hf_model_dir or enable --auto_convert_dcp."
        )

    model_assets_dir = Path(args.run_dir) / "model_assets"
    if not model_assets_dir.exists():
        raise FileNotFoundError(f"model_assets dir not found: {model_assets_dir}")

    merge_to_hf_pt = load_merge_to_hf_pt()

    os.makedirs(hf_dir, exist_ok=True)
    print(f"[INFO] Converting DCP checkpoint to HF: {checkpoint_path} -> {hf_dir}")
    merge_to_hf_pt(load_dir=str(checkpoint_path), save_path=str(hf_dir), model_assets_dir=str(model_assets_dir))
    return hf_dir


def install_sklearn_stub() -> None:
    if "sklearn" in sys.modules:
        return

    sklearn_stub = types.ModuleType("sklearn")
    sklearn_stub.__spec__ = importlib.machinery.ModuleSpec("sklearn", loader=None)
    metrics_stub = types.ModuleType("sklearn.metrics")
    metrics_stub.__spec__ = importlib.machinery.ModuleSpec("sklearn.metrics", loader=None)

    def roc_curve(*_args, **_kwargs):
        raise RuntimeError("The lightweight sklearn stub does not implement roc_curve.")

    metrics_stub.roc_curve = roc_curve
    sklearn_stub.metrics = metrics_stub
    sys.modules["sklearn"] = sklearn_stub
    sys.modules["sklearn.metrics"] = metrics_stub


def load_merge_to_hf_pt():
    install_sklearn_stub()
    merge_path = VEOMNI_ROOT / "scripts" / "merge_dcp_to_hf.py"
    if not merge_path.is_file():
        raise FileNotFoundError(f"merge_dcp_to_hf.py not found: {merge_path}")

    spec = importlib.util.spec_from_file_location("veomni_merge_dcp_to_hf", merge_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load merge module from {merge_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.merge_to_hf_pt


def resolve_tokenizer_dir(args: argparse.Namespace, run_dir: Path, hf_dir: Path) -> Path:
    candidates = []
    if args.tokenizer_path:
        candidates.append(Path(args.tokenizer_path))
    candidates.extend([hf_dir, run_dir / "model_assets"])

    for candidate in candidates:
        if (candidate / "tokenizer.json").exists() and (candidate / "tokenizer_config.json").exists():
            return candidate
    raise FileNotFoundError("Cannot find tokenizer dir. Checked: " + ", ".join(str(p) for p in candidates))


def read_text(path: Path) -> str:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Empty target text: {path}")
    return text


def flatten_long_tensor(x: torch.Tensor) -> list[int]:
    return [int(v) for v in torch.as_tensor(x).long().reshape(-1).tolist()]


def to_tokens(ids: list[int], prefix: str) -> str:
    return " ".join(f"<|{prefix}_{idx}|>" for idx in ids)


def remove_silence(audio: np.ndarray, threshold: float) -> np.ndarray:
    if audio.ndim == 1:
        energy = np.abs(audio)
    else:
        energy = np.max(np.abs(audio), axis=1)
    non_silent = np.where(energy > threshold)[0]
    if len(non_silent) == 0:
        return audio
    return audio[non_silent]


def maybe_trim_ref_audio(audio_path: Path) -> tuple[str, str | None]:
    audio, sample_rate = sf.read(str(audio_path))
    audio = np.asarray(audio)
    if audio.size == 0:
        return str(audio_path), None

    peak = float(np.max(np.abs(audio)))
    if peak <= 0.0:
        return str(audio_path), None

    trimmed = remove_silence(audio, threshold=max(peak * 0.01, 1e-4))
    if trimmed.shape == audio.shape:
        return str(audio_path), None

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_path = tmp.name
    tmp.close()
    sf.write(tmp_path, trimmed, sample_rate)
    return tmp_path, tmp_path


def encode_audio_global(audio_tokenizer: BiCodecTokenizer, audio_path: Path, global_prefix_len: int) -> list[int]:
    audio_for_global, tmp_path = maybe_trim_ref_audio(audio_path)
    try:
        global_ids, _ = audio_tokenizer.tokenize(audio_for_global)
        return flatten_long_tensor(global_ids)[:global_prefix_len]
    finally:
        if tmp_path is not None:
            Path(tmp_path).unlink(missing_ok=True)


def encode_audio_semantic(audio_tokenizer: BiCodecTokenizer, audio_path: Path) -> list[int]:
    _, semantic_ids = audio_tokenizer.tokenize(str(audio_path))
    return flatten_long_tensor(semantic_ids)


def encode_audio_full(
    audio_tokenizer: BiCodecTokenizer,
    audio_path: Path,
    global_prefix_len: int,
) -> tuple[list[int], list[int]]:
    global_ids, semantic_ids = audio_tokenizer.tokenize(str(audio_path))
    return flatten_long_tensor(global_ids)[:global_prefix_len], flatten_long_tensor(semantic_ids)


def format_template(template: str, case_name: str) -> str:
    return template.format(case=case_name, case_name=case_name, name=case_name)


def natural_key(path: Path) -> list[Any]:
    parts = re.split(r"(\d+)", path.name)
    return [int(part) if part.isdigit() else part for part in parts]


def audio_files_under(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        return []
    files = [item for item in path.iterdir() if item.is_file() and item.suffix.lower() in AUDIO_SUFFIXES]
    return sorted(files, key=natural_key)


def resolve_case_dir(args: argparse.Namespace, case_name: str) -> Path:
    case_dir = Path(args.cases_dir) / case_name
    if not case_dir.is_dir():
        raise FileNotFoundError(f"case dir not found: {case_dir}")
    return case_dir


def resolve_history_paths(args: argparse.Namespace, case_name: str) -> list[Path]:
    case_dir = resolve_case_dir(args, case_name)
    if args.history_glob:
        paths = [p for p in case_dir.glob(format_template(args.history_glob, case_name)) if p.is_file()]
        return sorted(paths, key=natural_key)

    direct_matches = []
    for suffix in AUDIO_SUFFIXES:
        path = case_dir / f"history{suffix}"
        if path.exists():
            direct_matches.append(path)
    if direct_matches:
        return sorted(direct_matches, key=natural_key)

    glob_matches = [p for p in case_dir.glob("history_*") if p.is_file() and p.suffix.lower() in AUDIO_SUFFIXES]
    if glob_matches:
        return sorted(glob_matches, key=natural_key)

    return audio_files_under(case_dir / "history")


def resolve_ref_path(args: argparse.Namespace, case_name: str) -> Path:
    path = resolve_case_dir(args, case_name) / format_template(args.ref_template, case_name)
    if path.exists():
        return path
    raise FileNotFoundError(path)


def resolve_text_path(args: argparse.Namespace, case_name: str) -> Path:
    path = resolve_case_dir(args, case_name) / format_template(args.text_template, case_name)
    if path.exists():
        return path
    raise FileNotFoundError(path)


def build_history_tokens(
    args: argparse.Namespace,
    audio_tokenizer: BiCodecTokenizer,
    history_paths: list[Path],
) -> tuple[str, int, int]:
    if not history_paths:
        return "", 0, 0

    global_count = 0
    semantic_count = 0
    parts: list[str] = []
    for idx, history_path in enumerate(history_paths):
        if args.history_mode == "full" or (args.history_mode == "full_first_then_semantic" and idx == 0):
            global_ids, semantic_ids = encode_audio_full(audio_tokenizer, history_path, args.global_prefix_len)
            if global_ids:
                parts.append(to_tokens(global_ids, args.global_prefix))
                global_count += len(global_ids)
        else:
            semantic_ids = encode_audio_semantic(audio_tokenizer, history_path)
        if semantic_ids:
            parts.append(to_tokens(semantic_ids, args.semantic_prefix))
            semantic_count += len(semantic_ids)
    return " ".join(parts), global_count, semantic_count


def build_prompt(text: str, history_tokens: str, ref_tokens: str) -> str:
    return (
        "<bos>"
        "<task_start>COT-TTS<task_end>"
        "<history_start>"
        "<spk_his_start><spk_his_end>"
        f"<audio_his_start>{history_tokens}<audio_his_end>"
        "<history_end>"
        "<target_start>"
        f"<text_start>{text}<text_end>"
        "<spk_tar_start><spk_tar_end>"
        f"<audio_ref_start>{ref_tokens}<audio_ref_end>"
        "<target_end>"
        "<output_start>"
        "<cot_start>"
    )


def encode_chat_prompt(tokenizer, prompt: str, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        return_tensors="pt",
    )
    if hasattr(encoded, "to"):
        encoded = encoded.to(device)

    if isinstance(encoded, torch.Tensor):
        return encoded, torch.ones_like(encoded)

    if hasattr(encoded, "get"):
        input_ids = encoded.get("input_ids")
        if input_ids is None:
            raise TypeError("tokenizer.apply_chat_template returned a mapping without input_ids.")
        attention_mask = encoded.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        return input_ids.to(device), attention_mask.to(device)

    raise TypeError(f"Unsupported apply_chat_template return type: {type(encoded)!r}")


def extract_between(text: str, start: str, end: str) -> str:
    start_idx = text.find(start)
    if start_idx < 0:
        return ""
    start_idx += len(start)
    end_idx = text.find(end, start_idx)
    if end_idx < 0:
        return text[start_idx:]
    return text[start_idx:end_idx]


def parse_audio_ids(text: str, global_prefix: str, semantic_prefix: str) -> tuple[list[int], list[int]]:
    global_ids = [int(x) for x in re.findall(rf"{re.escape(global_prefix)}_(\d+)", text)]
    semantic_ids = [int(x) for x in re.findall(rf"{re.escape(semantic_prefix)}_(\d+)", text)]
    return global_ids, semantic_ids


def parse_audio_ids_from_token_ids(
    tokenizer,
    token_ids: list[int],
    global_prefix: str,
    semantic_prefix: str,
) -> tuple[list[int], list[int]]:
    if not token_ids:
        return [], []
    tokens = tokenizer.convert_ids_to_tokens(token_ids, skip_special_tokens=False)
    if isinstance(tokens, str):
        tokens = [tokens]
    return parse_audio_ids(" ".join(str(token) for token in tokens), global_prefix, semantic_prefix)


def extract_audio_ids(
    gen_text: str,
    args: argparse.Namespace,
    tokenizer=None,
    generated_token_ids: list[int] | None = None,
) -> tuple[list[int], list[int], str]:
    audio_block = extract_between(gen_text, "<audio_tar_start>", "<audio_tar_end>")
    if audio_block.strip():
        global_ids, semantic_ids = parse_audio_ids(audio_block, args.global_prefix, args.semantic_prefix)
        if semantic_ids:
            return global_ids, semantic_ids, "audio_tar_block"

    global_ids, semantic_ids = parse_audio_ids(gen_text, args.global_prefix, args.semantic_prefix)
    if semantic_ids:
        return global_ids, semantic_ids, "full_generation_text"

    if tokenizer is not None and generated_token_ids is not None:
        global_ids, semantic_ids = parse_audio_ids_from_token_ids(
            tokenizer,
            generated_token_ids,
            args.global_prefix,
            args.semantic_prefix,
        )
        if semantic_ids:
            return global_ids, semantic_ids, "generated_token_ids"

    return global_ids, semantic_ids, "no_audio_tokens"


def resolve_stop_token_ids(tokenizer, extra_tokens: list[str]) -> list[int]:
    stop_ids: list[int] = []
    seen: set[int] = set()

    def _add(token_id) -> None:
        if token_id is None:
            return
        token_id = int(token_id)
        if token_id < 0 or token_id in seen:
            return
        seen.add(token_id)
        stop_ids.append(token_id)

    eos_token_id = tokenizer.eos_token_id
    if isinstance(eos_token_id, (list, tuple)):
        for token_id in eos_token_id:
            _add(token_id)
    else:
        _add(eos_token_id)

    for token in extra_tokens:
        token_ids = tokenizer.encode(token, add_special_tokens=False)
        if len(token_ids) == 1:
            _add(token_ids[0])
    return stop_ids


def generation_kwargs(args: argparse.Namespace, tokenizer) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "max_new_tokens": args.max_new_tokens,
        "min_new_tokens": args.min_new_tokens,
        "do_sample": args.do_sample,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "eos_token_id": resolve_stop_token_ids(tokenizer, ["<audio_tar_end>", "<output_end>", "<eos>"]),
        "pad_token_id": tokenizer.eos_token_id,
    }
    if args.repetition_penalty > 1.0:
        kwargs["repetition_penalty"] = args.repetition_penalty
    if args.no_repeat_ngram_size > 0:
        kwargs["no_repeat_ngram_size"] = args.no_repeat_ngram_size
    return kwargs


def candidate_prefix(case_prefix: Path, candidate_idx: int) -> Path:
    return case_prefix.with_name(f"{case_prefix.name}.cand{candidate_idx:02d}")


def load_wav_mono(path: Path, sample_rate: int) -> torch.Tensor:
    wav, sr = sf.read(str(path), dtype="float32", always_2d=True)
    wav_tensor = torch.from_numpy(wav).mean(dim=1).unsqueeze(0)
    if sr != sample_rate:
        try:
            import torchaudio.functional as AF
        except Exception as exc:
            raise RuntimeError("torchaudio is required to resample audio for mel_similarity") from exc
        wav_tensor = AF.resample(wav_tensor, orig_freq=sr, new_freq=sample_rate)
    return wav_tensor


def log_mel_stats(wav: torch.Tensor, sample_rate: int) -> torch.Tensor:
    try:
        import torchaudio.transforms as AT
    except Exception as exc:
        raise RuntimeError("torchaudio is required for --rerank_metric mel_similarity") from exc

    mel = AT.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=1024,
        hop_length=256,
        n_mels=80,
        f_min=40,
        f_max=min(sample_rate // 2, 7600),
        power=2.0,
    )(wav)
    log_mel = torch.log(mel.clamp_min(1e-5)).squeeze(0)
    feat = torch.cat([log_mel.mean(dim=1), log_mel.std(dim=1)])
    return torch.nn.functional.normalize(feat.float(), dim=0)


def mel_similarity(ref_path: Path, wav: torch.Tensor, sample_rate: int) -> float:
    ref_feat = log_mel_stats(load_wav_mono(ref_path, sample_rate), sample_rate)
    cand_feat = log_mel_stats(wav.float().view(1, -1).cpu(), sample_rate)
    return float(torch.nn.functional.cosine_similarity(ref_feat, cand_feat, dim=0).item())


def select_candidate(candidates: list[dict[str, Any]], metric: str) -> int:
    if metric == "none":
        return 0
    scored = [(idx, item.get("score")) for idx, item in enumerate(candidates) if item.get("score") is not None]
    if not scored:
        return 0
    return max(scored, key=lambda item: float(item[1]))[0]


def write_candidate_files(prefix: Path, candidate: dict[str, Any], sample_rate: int) -> None:
    torch.save(
        {
            "global_ids": torch.tensor(candidate["global_ids"], dtype=torch.int32),
            "semantic_ids": torch.tensor(candidate["semantic_ids"], dtype=torch.int32),
        },
        prefix.with_suffix(".pt"),
    )
    prefix.with_suffix(".raw.txt").write_text(candidate["gen_text"], encoding="utf-8")
    prefix.with_suffix(".under.txt").write_text(candidate["history_transcript"], encoding="utf-8")
    prefix.with_suffix(".cot.txt").write_text(candidate["cot_thinking"], encoding="utf-8")
    sf.write(str(prefix.with_suffix(".wav")), candidate["wav"].numpy(), samplerate=sample_rate)


def write_failed_generation_files(prefix: Path, prompt: str, gen_text: str, reason: str) -> None:
    prefix.with_suffix(".failed.prompt.txt").write_text(prompt, encoding="utf-8")
    prefix.with_suffix(".failed.raw.txt").write_text(gen_text, encoding="utf-8")
    prefix.with_suffix(".failed.meta.json").write_text(
        json.dumps({"reason": reason}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def infer_case(
    args: argparse.Namespace,
    case_name: str,
    tokenizer,
    model,
    audio_tokenizer: BiCodecTokenizer,
    device: torch.device,
    out_dir: Path,
) -> None:
    if args.num_candidates <= 0:
        raise ValueError("--num_candidates must be > 0")
    if args.num_candidates > 1 and args.rerank_metric == "none":
        print(f"[case {case_name}] --num_candidates > 1 but rerank_metric=none; selecting candidate 1.")

    history_paths = resolve_history_paths(args, case_name)
    ref_path = resolve_ref_path(args, case_name)
    text_path = resolve_text_path(args, case_name)
    if not history_paths:
        raise FileNotFoundError(f"No history audio found for case {case_name} under {args.cases_dir}")

    text = read_text(text_path)
    history_tokens, history_global_count, history_semantic_count = build_history_tokens(args, audio_tokenizer, history_paths)
    ref_global_ids = encode_audio_global(audio_tokenizer, ref_path, args.global_prefix_len)
    if not ref_global_ids:
        raise RuntimeError(f"Reference audio produced no global tokens: {ref_path}")

    prompt = build_prompt(
        text=text,
        history_tokens=history_tokens,
        ref_tokens=to_tokens(ref_global_ids, args.global_prefix),
    )
    input_ids, attention_mask = encode_chat_prompt(tokenizer, prompt, device)

    case_prefix = out_dir / case_name
    candidates: list[dict[str, Any]] = []
    gen_kwargs = generation_kwargs(args, tokenizer)

    for candidate_idx in range(1, args.num_candidates + 1):
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **gen_kwargs,
            )

        generated_token_ids = output_ids[0, input_ids.shape[1] :].detach().cpu().tolist()
        gen_text = tokenizer.decode(generated_token_ids, skip_special_tokens=False)
        generated_global_ids, semantic_ids, parse_source = extract_audio_ids(
            gen_text,
            args,
            tokenizer=tokenizer,
            generated_token_ids=generated_token_ids,
        )
        if args.max_semantic_tokens > 0 and len(semantic_ids) > args.max_semantic_tokens:
            semantic_ids = semantic_ids[: args.max_semantic_tokens]
        if not semantic_ids:
            failed_prefix = candidate_prefix(case_prefix, candidate_idx) if args.num_candidates > 1 else case_prefix
            write_failed_generation_files(failed_prefix, prompt, gen_text, parse_source)
            print(
                f"[case {case_name} cand {candidate_idx:02d}] "
                f"no bicodec semantic tokens found; parse={parse_source}; skipped"
            )
            continue

        if args.use_ref_global_tokens or not generated_global_ids:
            global_ids = ref_global_ids
            global_source = "ref"
        else:
            global_ids = generated_global_ids
            global_source = "generated"

        history_transcript = extract_between(gen_text, "<under_start>", "<under_end>")
        cot_thinking = extract_between(gen_text, "<cot_start>", "<cot_end>")
        wav = audio_tokenizer.detokenize(
            torch.tensor(global_ids, dtype=torch.long, device=device).unsqueeze(0),
            torch.tensor(semantic_ids, dtype=torch.long, device=device).unsqueeze(0),
        )
        wav = torch.as_tensor(wav).reshape(-1).detach().cpu()
        score = None
        if args.rerank_metric == "mel_similarity":
            score = mel_similarity(ref_path, wav, args.sample_rate)

        candidate = {
            "candidate_idx": candidate_idx,
            "gen_text": gen_text,
            "global_ids": global_ids,
            "semantic_ids": semantic_ids,
            "global_source": global_source,
            "parse_source": parse_source,
            "history_transcript": history_transcript,
            "cot_thinking": cot_thinking,
            "wav": wav,
            "score": score,
        }
        candidates.append(candidate)

        if args.num_candidates > 1:
            write_candidate_files(candidate_prefix(case_prefix, candidate_idx), candidate, args.sample_rate)
        score_text = "" if score is None else f" score={score:.4f}"
        print(
            f"[case {case_name} cand {candidate_idx:02d}] "
            f"global={len(global_ids)}({global_source}) semantic={len(semantic_ids)} "
            f"parse={parse_source}{score_text}"
        )

    if not candidates:
        raise RuntimeError(f"Case {case_name}: no valid BiCodec audio candidates found in generation.")

    selected_idx = select_candidate(candidates, args.rerank_metric)
    selected = candidates[selected_idx]
    write_candidate_files(case_prefix, selected, args.sample_rate)
    case_prefix.with_suffix(".history_transcript.txt").write_text(selected["history_transcript"], encoding="utf-8")
    case_prefix.with_suffix(".cot_thinking.txt").write_text(selected["cot_thinking"], encoding="utf-8")
    case_prefix.with_suffix(".target_text.txt").write_text(text, encoding="utf-8")
    case_prefix.with_suffix(".meta.json").write_text(
        json.dumps(
            {
                "case_name": case_name,
                "target_text": text,
                "history_audio_paths": [str(path) for path in history_paths],
                "ref_audio_path": str(ref_path),
                "history_mode": args.history_mode,
                "history_global_tokens": history_global_count,
                "history_semantic_tokens": history_semantic_count,
                "ref_global_tokens": len(ref_global_ids),
                "num_candidates": args.num_candidates,
                "rerank_metric": args.rerank_metric,
                "selected_candidate_idx": selected["candidate_idx"],
                "selected_candidate_rank": selected_idx + 1,
                "selected_score": selected["score"],
                "generated_global_tokens": len(selected["global_ids"]),
                "generated_semantic_tokens": len(selected["semantic_ids"]),
                "global_source": selected["global_source"],
                "parse_source": selected["parse_source"],
                "candidates": [
                    {
                        "candidate_idx": item["candidate_idx"],
                        "global_tokens": len(item["global_ids"]),
                        "semantic_tokens": len(item["semantic_ids"]),
                        "global_source": item["global_source"],
                        "parse_source": item["parse_source"],
                        "score": item["score"],
                        "wav_path": str(candidate_prefix(case_prefix, item["candidate_idx"]).with_suffix(".wav"))
                        if args.num_candidates > 1
                        else str(case_prefix.with_suffix(".wav")),
                    }
                    for item in candidates
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"[case {case_name}] text_len={len(text)} "
        f"history_files={len(history_paths)} history_semantic={history_semantic_count} "
        f"ref_global={len(ref_global_ids)} selected={selected['candidate_idx']}/{args.num_candidates} "
        f"semantic={len(selected['semantic_ids'])} parse={selected['parse_source']} "
        f"score={selected['score']} wav={case_prefix.with_suffix('.wav')}"
    )


def hf_checkpoint_has_lora(hf_dir: Path) -> bool:
    state_path = hf_dir / "model.safetensors"
    if not state_path.exists():
        return False
    try:
        from safetensors import safe_open
    except Exception as exc:
        print(f"[WARN] Cannot import safetensors to auto-detect LoRA checkpoint: {exc}")
        return False

    with safe_open(str(state_path), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            if ".lora_A." in key or ".lora_B." in key or ".base_layer." in key:
                return True
    return False


def load_text_model_and_tokenizer(
    args: argparse.Namespace,
    checkpoint_path: Path,
    run_dir: Path,
    hf_dir: Path,
    tokenizer_dir: Path,
    device: torch.device,
):
    has_lora_weights = hf_checkpoint_has_lora(hf_dir)
    use_lora = args.model_architecture == "lora" or (
        args.model_architecture == "auto" and has_lora_weights
    )

    install_sklearn_stub()

    if use_lora:
        from lora_infer_utils import load_lora_model_and_tokenizer

        tokenizer, model = load_lora_model_and_tokenizer(
            args=args,
            checkpoint_path=checkpoint_path,
            run_dir=run_dir,
            hf_dir=hf_dir,
            tokenizer_dir=tokenizer_dir,
            device=device,
        )
        return tokenizer, model, "lora", has_lora_weights

    if has_lora_weights and args.model_architecture == "full":
        print("[WARN] model_architecture=full but LoRA keys were detected in model.safetensors.")

    from veomni.models import build_foundation_model, build_tokenizer

    tokenizer = build_tokenizer(str(tokenizer_dir))
    model = build_foundation_model(
        config_path=str(hf_dir),
        weights_path=str(hf_dir),
        torch_dtype=args.torch_dtype,
        attn_implementation=args.attn_implementation,
    ).eval().to(device)
    return tokenizer, model, "full", has_lora_weights


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    checkpoint_path = Path(args.checkpoint_path) if args.checkpoint_path else Path()
    run_dir = resolve_run_dir(args, checkpoint_path) if args.checkpoint_path else Path(args.run_dir)
    args.run_dir = str(run_dir)
    if not args.checkpoint_path:
        checkpoint_path = find_latest_checkpoint(run_dir)
    device = resolve_device(args.device)
    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device)

    hf_dir = ensure_hf_model_dir(args, checkpoint_path)
    tokenizer_dir = resolve_tokenizer_dir(args, run_dir, hf_dir)

    from sparktts.models.audio_tokenizer import BiCodecTokenizer

    tokenizer, model, loaded_architecture, detected_lora = load_text_model_and_tokenizer(
        args=args,
        checkpoint_path=checkpoint_path,
        run_dir=run_dir,
        hf_dir=hf_dir,
        tokenizer_dir=tokenizer_dir,
        device=device,
    )
    audio_tokenizer = BiCodecTokenizer(args.spark_model_dir, device=device)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("===== COT-TTS Case Inference (Spark BiCodec) =====")
    print(f"checkpoint_path : {checkpoint_path}")
    print(f"run_dir         : {run_dir}")
    print(f"hf_model_dir    : {hf_dir}")
    print(f"tokenizer_dir   : {tokenizer_dir}")
    print(f"model_arch      : {loaded_architecture} (detected_lora={detected_lora})")
    print(f"spark_model_dir : {args.spark_model_dir}")
    print(f"cases_dir       : {args.cases_dir}")
    print(f"output_dir      : {out_dir}")
    print(f"case_names      : {args.case_names}")
    print(f"history_mode    : {args.history_mode}")
    print(f"num_candidates  : {args.num_candidates}")
    print(f"rerank_metric   : {args.rerank_metric}")
    print("----------------------------------------------------------")

    for case_name in args.case_names:
        infer_case(args, case_name, tokenizer, model, audio_tokenizer, device, out_dir)


if __name__ == "__main__":
    main()
