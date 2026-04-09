"""
NVFP4 Training Configuration and DeepSpeed Integration.

Generates the DeepSpeed ZeRO-2 configuration and Transformer Engine
NVFP4 settings required for native 4-bit training on Blackwell GPUs.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

from sentinel.config import Precision, TrainingConfig

logger = logging.getLogger(__name__)


def build_deepspeed_config(config: TrainingConfig) -> dict[str, Any]:
    """
    Generate DeepSpeed ZeRO-2 configuration for SENTINEL training.

    ZeRO Stage 2 shards optimizer states and gradients across GPUs
    while replicating model weights. For the core 8K-context phase:
    - TP=8 would waste NVLink bandwidth on tiny 2GB shards
    - Pure DDP can't fit optimizer states (372 GB) on single GPU (192 GB)
    - ZeRO-2 gives ~79 GB fixed overhead, ~113 GB for activations per GPU

    NOTE: ZeRO-2 alone is NOT sufficient for long-context phases.
    For 32K+ sequences, Context Parallelism (CP) or Sequence Parallelism
    (SP) is required to distribute activation memory across GPUs.
    Phase B/C configs add CP on top of ZeRO-2.
    See: NVIDIA Megatron-LM Context Parallelism documentation.
    """
    phase_config = config.context_phases.get("core", {})
    micro_batch = int(phase_config.get("batch_size", 8))

    ds_config: dict[str, Any] = {
        "train_batch_size": config.effective_batch_size,
        "train_micro_batch_size_per_gpu": micro_batch,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,

        # ZeRO Stage 2: shard optimizer states + gradients
        "zero_optimization": {
            "stage": config.zero_stage,
            "allgather_partitions": True,
            "allgather_bucket_size": 5e8,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e8,
            "contiguous_gradients": True,
            # Round-robin partition for balanced memory
            "round_robin_gradients": True,
        },

        # Communication
        "communication_data_type": "bf16",

        # Gradient clipping
        "gradient_clipping": 1.0,

        # Logging
        "steps_per_print": 50,
        "wall_clock_breakdown": True,

        # Checkpointing
        "activation_checkpointing": {
            "partition_activations": True,
            "cpu_checkpointing": False,
            "contiguous_memory_optimization": True,
            "number_checkpoints": None,
            "synchronize_checkpoint_boundary": False,
            "profile": False,
        },
    }

    # Precision-specific settings
    if config.precision == Precision.BF16:
        ds_config["bf16"] = {"enabled": True}
    elif config.precision == Precision.FP8:
        ds_config["bf16"] = {"enabled": True}  # FP8 uses BF16 as base
        ds_config["fp8"] = {
            "enabled": True,
            "recipe": "tensorwise",
        }
    elif config.precision == Precision.NVFP4:
        # NVFP4 uses BF16 as the base precision for non-quantized operations
        ds_config["bf16"] = {"enabled": True}
        # NVFP4-specific settings are handled by Transformer Engine, not DeepSpeed
        # DeepSpeed only manages the parallelism and optimizer sharding

    # Optimizer
    if config.optimizer.value == "adam":
        ds_config["optimizer"] = {
            "type": "AdamW",
            "params": {
                "lr": config.sft_lr,
                "betas": [config.adam_beta1, config.adam_beta2],
                "eps": 1e-8,
                "weight_decay": config.weight_decay,
            },
        }
    # Muon optimizer would be configured differently (custom)

    # Scheduler
    ds_config["scheduler"] = {
        "type": "WarmupCosineLR",
        "params": {
            "warmup_min_lr": 0,
            "warmup_max_lr": config.sft_lr,
            "warmup_num_steps": config.warmup_steps,
            "total_num_steps": 100000,  # Will be overridden per phase
        },
    }

    return ds_config


def build_nvfp4_te_config(config: TrainingConfig) -> dict[str, Any]:
    """
    Generate NVIDIA Transformer Engine configuration for NVFP4 training.

    Implements the four-component NVFP4 methodology from Definition 1
    of the doctoral research proposal:
    1. Random Hadamard Transforms (16×16 on Wgrad)
    2. Two-dimensional block scaling (16×16 for weights)
    3. Stochastic rounding (gradients only)
    4. Selective high-precision layers (15%)
    """
    te_config: dict[str, Any] = {
        "fp4_training": {
            "enabled": config.precision == Precision.NVFP4,

            # Component 1: Random Hadamard Transforms
            "rht": {
                "enabled": True,
                "matrix_size": config.nvfp4_rht_size,  # 16×16
                "apply_to": ["wgrad"],  # Only weight gradient GEMMs
                "random_sign_vector": "single_fixed_seed",
            },

            # Component 2: Two-dimensional block scaling
            "scaling": {
                "weight_block_size": [
                    config.nvfp4_block_size,
                    config.nvfp4_block_size,
                ],  # 16×16 2D blocks
                "activation_block_size": [1, config.nvfp4_block_size],  # 1×16 1D
                "gradient_block_size": [1, config.nvfp4_block_size],   # 1×16 1D
                "scale_format": "e4m3",          # Block-level FP8 scales
                "tensor_scale_format": "fp32",   # Tensor-level FP32 scales
            },

            # Component 3: Stochastic rounding
            "rounding": {
                "weights": "round_to_nearest_even",
                "activations": "round_to_nearest_even",
                "gradients": "stochastic",  # Critical for convergence
            },

            # Component 4: Selective high-precision layers
            "high_precision_layers": {
                "enabled": True,
                "ratio": config.nvfp4_high_precision_ratio,  # 15%
                "strategy": "first_and_last",
                "first_n_blocks": 2,
                "last_n_blocks": 8,
                "high_precision_format": "bf16",
            },
        },

        # Attention always in higher precision
        "attention": {
            "softmax_precision": "fp32",
            "qk_precision": "bf16",
            "av_precision": "bf16",
        },

        # Embeddings in original precision
        "embeddings": {
            "precision": "bf16",
        },

        # Normalization in original precision
        "normalization": {
            "precision": "fp32",
        },
    }

    return te_config


def build_phase_config(
    config: TrainingConfig,
    phase_name: str,
) -> dict[str, Any]:
    """
    Generate training configuration for a specific context-extension phase.

    Implements phased context extension from Section 3.1 of the spec:
    - Phase A (core): 8,192 tokens, batch 8-16, 14 days
    - Phase B (extended): 32,768 tokens, batch 2-4, 5 days
    - Phase C (long): 131,072 tokens, batch 1, 2 days
    """
    phase = config.context_phases.get(phase_name)
    if phase is None:
        raise ValueError(f"Unknown phase: {phase_name}. Valid: {list(config.context_phases.keys())}")

    seq_length = int(phase["seq_length"])
    batch_size = int(phase["batch_size"])
    epochs = int(phase.get("epochs", 1))

    return {
        "phase_name": phase_name,
        "max_seq_length": seq_length,
        "per_device_batch_size": batch_size,
        "num_epochs": epochs,
        "estimated_days": int(phase.get("days", 0)),

        # YaRN context extension for phases B and C
        "rope_scaling": _build_yarn_config(seq_length) if seq_length > 8192 else None,

        # Gradient checkpointing is essential for long contexts
        "gradient_checkpointing": seq_length >= 32768,
        "gradient_checkpointing_kwargs": {
            "use_reentrant": False,
        } if seq_length >= 32768 else {},

        # Context Parallelism (CP) for long sequences.
        # ZeRO-2 only shards optimizer/gradients, NOT activations.
        # For 32K+ contexts, activation memory exceeds per-GPU budget
        # without CP distributing the sequence dimension across GPUs.
        "context_parallelism": _build_context_parallel_config(seq_length, config.num_gpus),

        # Effective batch size for this phase
        "effective_batch_size": (
            config.num_gpus * batch_size * config.gradient_accumulation_steps
        ),
    }


def _build_yarn_config(target_seq_length: int) -> dict[str, Any]:
    """
    Build YaRN (Yet another RoPE extensioN) configuration for context extension.

    Uses NTK-aware interpolation as specified in the training protocol.
    """
    base_seq_length = 8192
    scale_factor = target_seq_length / base_seq_length

    return {
        "type": "yarn",
        "factor": scale_factor,
        "original_max_position_embeddings": base_seq_length,
        "attention_factor": 1.0,
        "beta_fast": 32,
        "beta_slow": 1,
        # NTK-aware: blend between linear interpolation and NTK scaling
        "ntk_alpha": scale_factor,
    }


def _build_context_parallel_config(
    seq_length: int,
    num_gpus: int,
) -> Optional[dict[str, Any]]:
    """
    Build Context Parallelism (CP) configuration for long sequences.

    Context Parallelism distributes activation memory across GPUs
    by splitting the sequence dimension. This is essential for 32K+
    contexts because ZeRO-2 only shards optimizer state and gradients,
    NOT activations — which grow quadratically with sequence length
    in standard attention.

    For 8K contexts: CP disabled (activations fit in per-GPU budget).
    For 32K contexts: CP degree 2 (split across GPU pairs).
    For 128K contexts: CP degree 4-8 (full sequence distribution).

    Reference: NVIDIA Megatron-LM Context Parallelism, 2024.
    """
    if seq_length <= 8192:
        return None

    # CP degree scales with sequence length relative to base
    if seq_length <= 32768:
        cp_degree = 2
    elif seq_length <= 65536:
        cp_degree = 4
    else:
        cp_degree = min(8, num_gpus)

    # Ensure CP degree divides num_gpus evenly
    while num_gpus % cp_degree != 0 and cp_degree > 1:
        cp_degree -= 1

    return {
        "enabled": True,
        "cp_degree": cp_degree,
        # Data-parallel degree is reduced proportionally
        "dp_degree": num_gpus // cp_degree,
        "attention_type": "ring_attention",
        # Ring attention distributes KV across the CP group
        # with overlapping communication and computation
        "overlap_comm": True,
        # Sequence dimension is split evenly across CP ranks
        "sequence_split_strategy": "equal",
    }


def save_configs(
    config: TrainingConfig,
    output_dir: Path,
) -> dict[str, Path]:
    """
    Save all training configurations to disk.

    Returns dict mapping config name to file path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: dict[str, Path] = {}

    # DeepSpeed config
    ds_path = output_dir / "ds_config.json"
    with open(ds_path, "w") as f:
        json.dump(build_deepspeed_config(config), f, indent=2)
    saved["deepspeed"] = ds_path

    # Transformer Engine config
    te_path = output_dir / "te_config.json"
    with open(te_path, "w") as f:
        json.dump(build_nvfp4_te_config(config), f, indent=2)
    saved["transformer_engine"] = te_path

    # Phase configs
    for phase_name in config.context_phases:
        phase_path = output_dir / f"phase_{phase_name}.json"
        with open(phase_path, "w") as f:
            json.dump(build_phase_config(config, phase_name), f, indent=2)
        saved[f"phase_{phase_name}"] = phase_path

    logger.info("Saved %d config files to %s", len(saved), output_dir)
    return saved


def estimate_vram_budget(config: TrainingConfig) -> dict[str, float]:
    """
    Estimate per-GPU VRAM budget under ZeRO-2.

    Returns breakdown in GB.
    """
    param_count = 31e9

    # Model weights in NVFP4
    if config.precision == Precision.NVFP4:
        weight_bytes_per_param = 0.5 + 0.0625  # FP4 + scale overhead
    elif config.precision == Precision.FP8:
        weight_bytes_per_param = 1.0
    else:
        weight_bytes_per_param = 2.0  # BF16

    model_gb = (param_count * weight_bytes_per_param) / 1e9

    # Optimizer states (sharded under ZeRO-2)
    optimizer_total_gb = (param_count * 12) / 1e9  # master + m + v in FP32
    optimizer_per_gpu = optimizer_total_gb / config.num_gpus

    # Gradients (sharded under ZeRO-2)
    gradient_total_gb = (param_count * 4) / 1e9  # FP32
    gradient_per_gpu = gradient_total_gb / config.num_gpus

    fixed_per_gpu = model_gb + optimizer_per_gpu + gradient_per_gpu
    activation_budget = config.gpu_memory_gb - fixed_per_gpu

    return {
        "model_weights_gb": round(model_gb, 1),
        "optimizer_shard_gb": round(optimizer_per_gpu, 1),
        "gradient_shard_gb": round(gradient_per_gpu, 1),
        "fixed_total_gb": round(fixed_per_gpu, 1),
        "activation_budget_gb": round(activation_budget, 1),
        "gpu_memory_gb": config.gpu_memory_gb,
        "utilization_pct": round((fixed_per_gpu / config.gpu_memory_gb) * 100, 1),
    }
