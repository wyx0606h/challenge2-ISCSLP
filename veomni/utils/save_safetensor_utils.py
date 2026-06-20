# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gc
import re
import time
from typing import Dict, Optional, Sequence

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp

from veomni.checkpoint import ckpt_to_state_dict
from veomni.models import save_model_assets, save_model_weights
from veomni.utils import helper
from veomni.utils.device import synchronize
from veomni.utils.import_utils import is_torch_version_greater_than


logger = helper.create_logger(__name__)


def _split_csv_patterns(value: Optional[str]):
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _layer_index_from_name(name: str) -> Optional[int]:
    match = re.search(r"(?:^|\.)layers\.(\d+)\.", name)
    if match is None:
        return None
    return int(match.group(1))


def _filter_lora_state_dict(
    state_dict: Dict[str, torch.Tensor],
    extra_param_patterns: Optional[str],
    extra_last_n_layers: int,
) -> Dict[str, torch.Tensor]:
    pattern_texts = _split_csv_patterns(extra_param_patterns)
    compiled_patterns = [re.compile(pattern) for pattern in pattern_texts]
    layer_indices = [
        layer_idx for name in state_dict.keys() if (layer_idx := _layer_index_from_name(name)) is not None
    ]
    layer_cutoff = None
    if extra_last_n_layers > 0:
        if not layer_indices:
            raise ValueError("partial_train_last_n_layers is set, but no checkpoint keys matched '.layers.<idx>.'")
        total_layers = max(layer_indices) + 1
        layer_cutoff = max(0, total_layers - extra_last_n_layers)

    filtered_state = {}
    for name, tensor in state_dict.items():
        if "lora" in name:
            filtered_state[name] = tensor
            continue

        matched_by_pattern = any(pattern.search(name) for pattern in compiled_patterns)
        layer_idx = _layer_index_from_name(name)
        matched_by_layer = layer_cutoff is not None and layer_idx is not None and layer_idx >= layer_cutoff
        if matched_by_pattern or matched_by_layer:
            filtered_state[name] = tensor

    logger.info_rank0(
        f"Filtered LoRA checkpoint to {len(filtered_state)} tensors "
        f"(extra_param_patterns={pattern_texts}, extra_last_n_layers={extra_last_n_layers})."
    )
    return filtered_state


@torch.no_grad()
def get_model_save_state(
    model: torch.nn.Module,
    fqn_to_index_mapping: Optional[Dict[str, int]],
) -> Dict[str, torch.Tensor]:
    """Build a flat state dict suitable for HuggingFace safetensors saving.

    1. Extracts a flat state dict via ``ModelState`` (FQNs match HF weight_map keys).
    2. Casts float32 tensors to bfloat16 on copies (original model dtypes are preserved).
    3. Filters out tied weights not present in ``fqn_to_index_mapping``.
    """
    from veomni.checkpoint.dcp_checkpointer import ModelState

    # Use flat state dict so DCP FQNs match the original HF weight_map keys
    # (e.g. "model.embed_tokens.weight" instead of "model.model.embed_tokens.weight")
    save_state = ModelState(model).state_dict()

    # Convert float32 tensors to bfloat16 on a copy of the state dict,
    # so the original model parameters remain unchanged.
    converted_state = {}
    for k, v in save_state.items():
        if v.dtype == torch.float32:
            logger.info_rank0(f"Converting {k} from {v.dtype} to torch.bfloat16")
            converted_state[k] = v.to(torch.bfloat16)
        else:
            converted_state[k] = v
    save_state = converted_state

    # Remove tied weights not present in the HF weight_map
    # (e.g. lm_head.weight is tied to model.embed_tokens.weight via tie_word_embeddings)
    if fqn_to_index_mapping is not None:
        filtered_state = {}
        for k, v in save_state.items():
            if k in fqn_to_index_mapping:
                filtered_state[k] = v
            else:
                logger.info_rank0(f"Skipping weight not in HF weight_map: {k}")
        save_state = filtered_state
    else:
        logger.warning_rank0(
            "fqn_to_index_mapping is None, HuggingFaceStorageWriter will save "
            "all model weights into a single safetensors file."
        )

    return save_state


def _save_hf_safetensor_distributed(
    model: torch.nn.Module,
    save_path: str,
    fqn_to_index_mapping: Optional[Dict[str, int]],
    model_assets: Optional[Sequence],
):
    """Distributed HuggingFace safetensors save using HuggingFaceStorageWriter (PyTorch >= 2.9).

    All ranks must call this function.
    """
    from torch.distributed.checkpoint import HuggingFaceStorageWriter

    storage_writer = HuggingFaceStorageWriter(
        path=save_path,
        save_distributed=True,
        fqn_to_index_mapping=fqn_to_index_mapping,
        enable_consolidation=True,
        thread_count_consolidation=5,
    )

    save_state = get_model_save_state(model, fqn_to_index_mapping)

    logger.info_rank0("Starting distributed HuggingFace safetensors save...")
    if dist.is_initialized():
        dist.barrier()
    start_time = time.time()
    dcp.save(
        state_dict=save_state,
        storage_writer=storage_writer,
    )
    del save_state  # Free copied tensors (e.g. fp32->bf16) to reduce peak memory
    if dist.is_initialized():
        dist.barrier()
    gc.collect()
    helper.empty_cache()
    elapsed_time = time.time() - start_time
    logger.info_rank0(f"Distributed HuggingFace safetensors save took {elapsed_time:.2f}s")

    # Save model assets (config, tokenizer, etc.) on rank 0
    if model_assets and (not dist.is_initialized() or dist.get_rank() == 0):
        save_model_assets(save_path, model_assets)

    logger.info_rank0(f"HuggingFace checkpoint saved at {save_path} successfully!")


def _save_hf_safetensor_legacy(
    save_checkpoint_path: str,
    save_hf_safetensor_path: str,
    model_assets: Optional[Sequence],
    ckpt_manager: str,
    train_architecture: Optional[str],
    output_dir: Optional[str],
    lora_extra_param_patterns: Optional[str],
    lora_extra_last_n_layers: int,
):
    """Legacy HuggingFace safetensors save via checkpoint conversion (rank-0 only)."""
    model_state_dict = ckpt_to_state_dict(
        save_checkpoint_path=save_checkpoint_path,
        ckpt_manager=ckpt_manager,
        output_dir=output_dir,
    )
    if train_architecture == "lora":
        model_state_dict = _filter_lora_state_dict(
            model_state_dict,
            extra_param_patterns=lora_extra_param_patterns,
            extra_last_n_layers=lora_extra_last_n_layers,
        )
    save_model_weights(save_hf_safetensor_path, model_state_dict, model_assets=model_assets)
    logger.info_rank0(f"HuggingFace checkpoint saved at {save_hf_safetensor_path} successfully!")


def save_hf_safetensor(
    save_hf_safetensor_path: Optional[str] = None,
    ckpt_manager: Optional[str] = None,
    model_assets: Optional[Sequence] = None,
    train_architecture: Optional[str] = None,
    # Legacy only
    save_checkpoint_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    is_rank_0: bool = False,
    # Distributed only
    model: Optional[torch.nn.Module] = None,
    fqn_to_index_mapping: Optional[Dict[str, int]] = None,
    lora_extra_param_patterns: Optional[str] = None,
    lora_extra_last_n_layers: int = 0,
):
    """Save model weights in HuggingFace safetensors format.

    This function is self-contained w.r.t. synchronization: it calls ``synchronize()`` at
    entry to flush pending GPU operations before reading tensor data, and calls
    ``dist.barrier()`` before returning to ensure all ranks complete the save. Callers
    do not need to add external synchronization around this function.

    Supports two modes:
    - Distributed mode (PyTorch >= 2.9, ckpt_manager="dcp", non-LoRA): Uses HuggingFaceStorageWriter
      for efficient distributed save directly from the live FSDP model. Must be called on all ranks.
    - Legacy mode: Loads from checkpoint and converts to safetensors on rank 0.

    Args:
        save_hf_safetensor_path: Output path for saved HuggingFace safetensors.
        ckpt_manager: Checkpoint manager type. Used for routing (distributed when "dcp")
            and passed to legacy ``ckpt_to_state_dict``.
        model_assets: Model assets (e.g., config, tokenizer) to save alongside weights.
        train_architecture: Training architecture type. Used for routing (legacy when "lora")
            and to filter LoRA weights in legacy mode.
        lora_extra_param_patterns: Extra non-LoRA checkpoint key patterns to include when saving
            LoRA checkpoints, for hybrid LoRA plus partial full-parameter tuning.
        lora_extra_last_n_layers: Extra final transformer layers to include when saving LoRA
            checkpoints, for hybrid LoRA plus partial full-parameter tuning.
        save_checkpoint_path: [Legacy only] Path to the distributed checkpoint for conversion.
        output_dir: [Legacy only] Output directory passed to ``ckpt_to_state_dict``.
        is_rank_0: [Legacy only] Whether the current process is global rank 0.
            Legacy save is rank-0 only; non-rank-0 processes return immediately.
            Required by non-dcp checkpoint managers (e.g., omnistore).
        model: [Distributed only] Live FSDP model for distributed save.
        fqn_to_index_mapping: [Distributed only] Maps FQNs to safetensors file indices
            for multi-file output.
    """
    from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer

    use_distributed = is_torch_version_greater_than("2.9") and train_architecture != "lora" and ckpt_manager == "dcp"

    # Ensure all GPU operations are complete before reading tensor data for saving
    synchronize()

    # Wait for any pending async DCP save
    if ckpt_manager == "dcp" and DistributedCheckpointer.dcp_save_future is not None:
        logger.info_rank0("Waiting for pending async DCP save to complete before HF safetensor save...")
        DistributedCheckpointer.dcp_save_future.result()
        DistributedCheckpointer.dcp_save_future = None
        if dist.is_initialized():
            dist.barrier()

    if use_distributed:
        _save_hf_safetensor_distributed(model, save_hf_safetensor_path, fqn_to_index_mapping, model_assets)
    else:
        # Legacy path is rank-0 only; non-rank-0 waits at the barrier below
        if is_rank_0:
            _save_hf_safetensor_legacy(
                save_checkpoint_path,
                save_hf_safetensor_path,
                model_assets,
                ckpt_manager,
                train_architecture,
                output_dir,
                lora_extra_param_patterns,
                lora_extra_last_n_layers,
            )

    # Ensure all ranks finish saving before anyone proceeds
    if dist.is_initialized():
        dist.barrier()
