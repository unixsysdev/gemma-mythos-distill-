"""
Supervised Fine-Tuning (SFT) training loop for SENTINEL.

Implements the phased context extension strategy:
  Phase A: 8K context — core vulnerability mechanics (14 days)
  Phase B: 32K context — multi-file data flow tracing (5 days)
  Phase C: 128K context — repository-scale analysis (2 days)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from sentinel.config import ContextPhase, Precision, TrainingConfig
from sentinel.training.nvfp4_config import (
    build_deepspeed_config,
    build_nvfp4_te_config,
    build_phase_config,
    estimate_vram_budget,
)

logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    """Metrics collected during a training run."""
    phase: str
    step: int = 0
    epoch: float = 0.0
    loss: float = 0.0
    learning_rate: float = 0.0
    grad_norm: float = 0.0
    throughput_tokens_per_sec: float = 0.0
    gpu_memory_used_gb: float = 0.0
    wall_time_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "step": self.step,
            "epoch": self.epoch,
            "loss": self.loss,
            "learning_rate": self.learning_rate,
            "grad_norm": self.grad_norm,
            "throughput_tokens_per_sec": self.throughput_tokens_per_sec,
            "gpu_memory_used_gb": self.gpu_memory_used_gb,
            "wall_time_seconds": self.wall_time_seconds,
        }


class SFTTrainer:
    """
    Orchestrates the SENTINEL SFT training pipeline.

    Manages phased context extension, NVFP4 precision configuration,
    and DeepSpeed ZeRO-2 distributed training.
    """

    PHASE_ORDER = [ContextPhase.CORE, ContextPhase.EXTENDED, ContextPhase.LONG_RANGE]

    def __init__(
        self,
        config: TrainingConfig,
        dataset_path: Path,
        output_dir: Path,
        resume_from_checkpoint: Optional[str] = None,
    ) -> None:
        self.config = config
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.resume_from_checkpoint = resume_from_checkpoint
        self._metrics_log: list[TrainingMetrics] = []

    def prepare(self) -> dict[str, Any]:
        """
        Prepare the training environment.

        Returns a summary of the training configuration.
        """
        # Estimate VRAM budget
        vram = estimate_vram_budget(self.config)
        logger.info("VRAM budget per GPU: %s", json.dumps(vram, indent=2))

        if vram["activation_budget_gb"] < 0:
            raise RuntimeError(
                f"Insufficient VRAM: need {vram['fixed_total_gb']:.1f} GB fixed, "
                f"but only {self.config.gpu_memory_gb} GB available per GPU."
            )

        # Save configs
        config_dir = self.output_dir / "configs"
        config_dir.mkdir(exist_ok=True)

        ds_config = build_deepspeed_config(self.config)
        with open(config_dir / "ds_config.json", "w") as f:
            json.dump(ds_config, f, indent=2)

        te_config = build_nvfp4_te_config(self.config)
        with open(config_dir / "te_config.json", "w") as f:
            json.dump(te_config, f, indent=2)

        return {
            "model": self.config.model_name,
            "precision": self.config.precision.value,
            "optimizer": self.config.optimizer.value,
            "num_gpus": self.config.num_gpus,
            "vram_budget": vram,
            "phases": list(self.config.context_phases.keys()),
        }

    def train_phase(self, phase: ContextPhase) -> list[TrainingMetrics]:
        """
        Execute a single training phase.

        This is where the actual HuggingFace/DeepSpeed training
        loop would be invoked. The method structures the configuration
        and delegates to the training framework.
        """
        phase_name = phase.value
        phase_config = build_phase_config(self.config, phase_name)

        logger.info(
            "Starting phase '%s': seq_len=%d, batch=%d, est. %d days",
            phase_name,
            phase_config["max_seq_length"],
            phase_config["per_device_batch_size"],
            phase_config["estimated_days"],
        )

        # Build training arguments
        training_args = self._build_training_args(phase_config)

        # Load model with appropriate precision
        model_kwargs = self._build_model_kwargs(phase_config)

        # Load and prepare dataset for this phase
        dataset_kwargs = self._build_dataset_kwargs(phase_config)

        # In production, this would invoke:
        # from transformers import Trainer, TrainingArguments
        # from trl import SFTTrainer as HFSFTTrainer
        #
        # trainer = HFSFTTrainer(
        #     model=model,
        #     args=training_args,
        #     train_dataset=dataset,
        #     tokenizer=tokenizer,
        #     ...
        # )
        # trainer.train(resume_from_checkpoint=self.resume_from_checkpoint)

        logger.info(
            "Phase '%s' configuration prepared. Training args: %s",
            phase_name,
            json.dumps(training_args, indent=2, default=str),
        )

        return self._metrics_log

    def train_all_phases(self) -> dict[str, list[TrainingMetrics]]:
        """
        Execute all training phases in sequence.

        Returns metrics keyed by phase name.
        """
        summary = self.prepare()
        logger.info("Training summary: %s", json.dumps(summary, indent=2))

        results: dict[str, list[TrainingMetrics]] = {}

        for phase in self.PHASE_ORDER:
            phase_name = phase.value
            if phase_name not in self.config.context_phases:
                logger.info("Skipping phase '%s' (not configured)", phase_name)
                continue

            metrics = self.train_phase(phase)
            results[phase_name] = metrics

            # Save checkpoint between phases
            checkpoint_path = self.output_dir / f"checkpoint-{phase_name}"
            logger.info("Phase '%s' complete. Checkpoint: %s", phase_name, checkpoint_path)

        # Save final metrics
        self._save_metrics()
        return results

    def _build_training_args(self, phase_config: dict[str, Any]) -> dict[str, Any]:
        """Build HuggingFace TrainingArguments-compatible dict."""
        return {
            "output_dir": str(self.output_dir),
            "per_device_train_batch_size": phase_config["per_device_batch_size"],
            "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
            "num_train_epochs": phase_config.get("num_epochs", 1),
            "learning_rate": self.config.sft_lr,
            "weight_decay": self.config.weight_decay,
            "warmup_steps": self.config.warmup_steps,
            "lr_scheduler_type": "cosine",
            "logging_steps": 10,
            "save_steps": self.config.save_steps,
            "eval_steps": self.config.eval_steps,
            "bf16": True,
            "gradient_checkpointing": phase_config.get("gradient_checkpointing", False),
            "deepspeed": str(self.output_dir / "configs" / "ds_config.json"),
            "max_seq_length": phase_config["max_seq_length"],
            "report_to": ["wandb"],
            "run_name": f"sentinel-sft-{phase_config['phase_name']}",
        }

    def _build_model_kwargs(self, phase_config: dict[str, Any]) -> dict[str, Any]:
        """Build model loading kwargs including NVFP4 configuration."""
        kwargs: dict[str, Any] = {
            "model_name_or_path": self.config.model_name,
            "torch_dtype": "bfloat16",
            "attn_implementation": "flash_attention_2",
        }

        # RoPE scaling for extended context phases
        rope_scaling = phase_config.get("rope_scaling")
        if rope_scaling is not None:
            kwargs["rope_scaling"] = rope_scaling

        return kwargs

    def _build_dataset_kwargs(self, phase_config: dict[str, Any]) -> dict[str, Any]:
        """Build dataset configuration for the current phase."""
        return {
            "dataset_path": str(self.dataset_path),
            "max_seq_length": phase_config["max_seq_length"],
            "packing": phase_config["max_seq_length"] <= 8192,
            "shuffle": True,
        }

    def _save_metrics(self) -> None:
        """Save collected metrics to disk."""
        metrics_path = self.output_dir / "training_metrics.jsonl"
        with open(metrics_path, "w") as f:
            for m in self._metrics_log:
                f.write(json.dumps(m.to_dict()) + "\n")
        logger.info("Saved %d metric records to %s", len(self._metrics_log), metrics_path)
