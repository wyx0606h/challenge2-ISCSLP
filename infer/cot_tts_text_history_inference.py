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

DEFAULT_RUN_DIR = "model/cot_tts_text_history_baseline"
DEFAULT_CHECKPOINT_PATH = f"{DEFAULT_RUN_DIR}/checkpoints/global_step_24500"
DEFAULT_SPARK_MODEL_DIR = "model/Spark-TTS"
DEFAULT_REF_AUDIO_PATH = "infer/cases/sample_case/reference.wav"
DEFAULT_TARGET_TEXT_PATH = "infer/cases/sample_case/target.txt"
DEFAULT_HISTORY_TEXT_PATH = "infer/cases/sample_case/history.txt"
DEFAULT_TARGET_TEXT = ""
DEFAULT_OUTPUT_PREFIX = "infer/results/text_history_demo"
DEFAULT_HISTORY_TEXT = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="COT-TTS inference with text dialogue history and Spark BiCodec reference audio."
    )
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

    parser.add_argument(
        "--history_text",
        type=str,
        default=DEFAULT_HISTORY_TEXT,
        help="Inline dialogue history text placed inside <under_start>...<under_end>.",
    )
    parser.add_argument(
        "--history_text_path",
        type=str,
        default=DEFAULT_HISTORY_TEXT_PATH,
        help="Dialogue history text file. Used when --history_text is empty.",
    )
    parser.add_argument(
        "--target_text",
        type=str,
        default=DEFAULT_TARGET_TEXT,
        help="Inline target text. Overrides --target_text_path when set.",
    )
    parser.add_argument(
        "--target_text_path",
        type=str,
        default=DEFAULT_TARGET_TEXT_PATH,
        help="Path to target text file.",
    )
    parser.add_argument(
        "--ref_audio_path",
        type=str,
        default=DEFAULT_REF_AUDIO_PATH,
        help="Reference audio path used to produce <audio_ref_start> global tokens.",
    )
    parser.add_argument(
        "--prompt_override",
        type=str,
        default="",
        help="Optional raw user prompt. If set, history/ref/target inputs are ignored.",
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        default=DEFAULT_OUTPUT_PREFIX,
        help="Output prefix without extension, e.g. infer/results/text_history_demo.",
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
        help="Generate N candidates and select one. 1 keeps the fast path.",
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
        raise ValueError(f"Empty text file: {path}")
    return text


def normalize_multiline_text(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        raise ValueError("Text content is empty after stripping.")
    return normalized


def resolve_history_text(args: argparse.Namespace) -> str:
    if args.history_text:
        return normalize_multiline_text(args.history_text)
    if args.history_text_path:
        return normalize_multiline_text(read_text(Path(args.history_text_path)))
    raise ValueError("Provide --history_text or --history_text_path.")


def resolve_target_text(args: argparse.Namespace) -> str:
    if args.target_text:
        return normalize_multiline_text(args.target_text)
    if args.target_text_path:
        return normalize_multiline_text(read_text(Path(args.target_text_path)))
    raise ValueError("Provide --target_text or --target_text_path.")


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


def build_prompt(history_text: str, target_text: str, ref_tokens: str) -> str:
    # This matches the cot-tts user prompt shape found in the new parquet data:
    # user ends at <output_start>, assistant generation begins with <cot_start>.
    return (
        "<bos>"
        "<task_start>COT-TTS<task_end>"
        "<history_start>"
        f"<under_start>{history_text}<under_end>"
        "<history_end>"
        "<target_start>"
        f"<text_start>{target_text}<text_end>"
        f"<audio_ref_start>{ref_tokens}<audio_ref_end>"
        "<target_end>"
        "<output_start>"
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


def candidate_prefix(output_prefix: Path, candidate_idx: int) -> Path:
    return output_prefix.with_name(f"{output_prefix.name}.cand{candidate_idx:02d}")


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
    prefix.with_suffix(".cot.txt").write_text(candidate["cot_thinking"], encoding="utf-8")
    sf.write(str(prefix.with_suffix(".wav")), candidate["wav"].numpy(), samplerate=sample_rate)


def write_failed_generation_files(prefix: Path, prompt: str, gen_text: str, reason: str) -> None:
    prefix.with_suffix(".failed.prompt.txt").write_text(prompt, encoding="utf-8")
    prefix.with_suffix(".failed.raw.txt").write_text(gen_text, encoding="utf-8")
    prefix.with_suffix(".failed.meta.json").write_text(
        json.dumps({"reason": reason}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def infer_text_history(
    args: argparse.Namespace,
    tokenizer,
    model,
    audio_tokenizer: BiCodecTokenizer,
    device: torch.device,
    output_prefix: Path,
    prompt: str,
    history_text: str,
    target_text: str,
    ref_path: Path | None,
    ref_global_ids: list[int],
) -> None:
    if args.num_candidates <= 0:
        raise ValueError("--num_candidates must be > 0")
    if args.num_candidates > 1 and args.rerank_metric == "none":
        print("[demo] --num_candidates > 1 but rerank_metric=none; selecting candidate 1.")

    input_ids, attention_mask = encode_chat_prompt(tokenizer, prompt, device)
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
            failed_prefix = candidate_prefix(output_prefix, candidate_idx) if args.num_candidates > 1 else output_prefix
            write_failed_generation_files(failed_prefix, prompt, gen_text, parse_source)
            print(f"[demo cand {candidate_idx:02d}] no semantic tokens found; parse={parse_source}; skipped")
            continue

        if args.use_ref_global_tokens:
            if not ref_global_ids:
                raise RuntimeError("--use_ref_global_tokens is enabled, but prompt has no reference global tokens.")
            global_ids = ref_global_ids
            global_source = "ref"
        elif generated_global_ids:
            global_ids = generated_global_ids
            global_source = "generated"
        elif ref_global_ids:
            global_ids = ref_global_ids
            global_source = "ref"
        else:
            raise RuntimeError("No generated or reference global tokens found for waveform decoding.")

        cot_thinking = extract_between(gen_text, "<cot_start>", "<cot_end>")
        wav = audio_tokenizer.detokenize(
            torch.tensor(global_ids, dtype=torch.long, device=device).unsqueeze(0),
            torch.tensor(semantic_ids, dtype=torch.long, device=device).unsqueeze(0),
        )
        wav = torch.as_tensor(wav).reshape(-1).detach().cpu()
        score = None
        if args.rerank_metric == "mel_similarity":
            if ref_path is None:
                raise RuntimeError("--rerank_metric mel_similarity requires a reference audio path.")
            score = mel_similarity(ref_path, wav, args.sample_rate)

        candidate = {
            "candidate_idx": candidate_idx,
            "gen_text": gen_text,
            "global_ids": global_ids,
            "semantic_ids": semantic_ids,
            "global_source": global_source,
            "parse_source": parse_source,
            "cot_thinking": cot_thinking,
            "wav": wav,
            "score": score,
        }
        candidates.append(candidate)

        if args.num_candidates > 1:
            write_candidate_files(candidate_prefix(output_prefix, candidate_idx), candidate, args.sample_rate)
        score_text = "" if score is None else f" score={score:.4f}"
        print(
            f"[demo cand {candidate_idx:02d}] "
            f"global={len(global_ids)}({global_source}) semantic={len(semantic_ids)} "
            f"parse={parse_source}{score_text}"
        )

    if not candidates:
        raise RuntimeError("No valid BiCodec audio candidates found in generation.")

    selected_idx = select_candidate(candidates, args.rerank_metric)
    selected = candidates[selected_idx]
    write_candidate_files(output_prefix, selected, args.sample_rate)
    output_prefix.with_suffix(".history.txt").write_text(history_text, encoding="utf-8")
    output_prefix.with_suffix(".target_text.txt").write_text(target_text, encoding="utf-8")
    output_prefix.with_suffix(".prompt.txt").write_text(prompt, encoding="utf-8")
    output_prefix.with_suffix(".cot_thinking.txt").write_text(selected["cot_thinking"], encoding="utf-8")
    output_prefix.with_suffix(".meta.json").write_text(
        json.dumps(
            {
                "task": "cot-tts",
                "prompt_format": "history_text_under_plus_ref_global",
                "history_text": history_text,
                "target_text": target_text,
                "ref_audio_path": str(ref_path) if ref_path is not None else "",
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
                        "wav_path": str(candidate_prefix(output_prefix, item["candidate_idx"]).with_suffix(".wav"))
                        if args.num_candidates > 1
                        else str(output_prefix.with_suffix(".wav")),
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
        f"[demo] history_chars={len(history_text)} target_len={len(target_text)} "
        f"ref_global={len(ref_global_ids)} selected={selected['candidate_idx']}/{args.num_candidates} "
        f"semantic={len(selected['semantic_ids'])} parse={selected['parse_source']} "
        f"score={selected['score']} wav={output_prefix.with_suffix('.wav')}"
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

    ref_path: Path | None = None
    if args.prompt_override:
        prompt = args.prompt_override
        history_text = extract_between(prompt, "<under_start>", "<under_end>")
        target_text = extract_between(prompt, "<text_start>", "<text_end>")
        ref_global_ids, _ = parse_audio_ids(
            extract_between(prompt, "<audio_ref_start>", "<audio_ref_end>"),
            args.global_prefix,
            args.semantic_prefix,
        )
        ref_global_ids = ref_global_ids[: args.global_prefix_len]
    else:
        history_text = resolve_history_text(args)
        target_text = resolve_target_text(args)
        ref_path = Path(args.ref_audio_path)
        if not ref_path.is_file():
            raise FileNotFoundError(ref_path)
        ref_global_ids = encode_audio_global(audio_tokenizer, ref_path, args.global_prefix_len)
        if not ref_global_ids:
            raise RuntimeError(f"Reference audio produced no global tokens: {ref_path}")
        prompt = build_prompt(
            history_text=history_text,
            target_text=target_text,
            ref_tokens=to_tokens(ref_global_ids, args.global_prefix),
        )

    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    print("===== COT-TTS Text-History Inference (Spark BiCodec) =====")
    print(f"checkpoint_path : {checkpoint_path}")
    print(f"run_dir         : {run_dir}")
    print(f"hf_model_dir    : {hf_dir}")
    print(f"tokenizer_dir   : {tokenizer_dir}")
    print(f"model_arch      : {loaded_architecture} (detected_lora={detected_lora})")
    print(f"spark_model_dir : {args.spark_model_dir}")
    print(f"ref_audio_path  : {ref_path if ref_path is not None else '<prompt_override>'}")
    print(
        f"target_text_src : "
        f"{'<prompt_override>' if args.prompt_override else (args.target_text_path if not args.target_text else '<inline>')}"
    )
    print(
        f"history_text_src: "
        f"{'<prompt_override>' if args.prompt_override else ('<inline>' if args.history_text else args.history_text_path)}"
    )
    print(f"output_prefix   : {output_prefix}")
    print(f"num_candidates  : {args.num_candidates}")
    print(f"rerank_metric   : {args.rerank_metric}")
    print("----------------------------------------------------------")

    infer_text_history(
        args=args,
        tokenizer=tokenizer,
        model=model,
        audio_tokenizer=audio_tokenizer,
        device=device,
        output_prefix=output_prefix,
        prompt=prompt,
        history_text=history_text,
        target_text=target_text,
        ref_path=ref_path,
        ref_global_ids=ref_global_ids,
    )


if __name__ == "__main__":
    main()
