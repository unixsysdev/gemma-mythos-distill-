"""Central configuration for all SENTINEL subsystems."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


class Precision(Enum):
    """Training precision formats."""
    NVFP4 = "nvfp4"
    FP8 = "fp8"
    BF16 = "bf16"
    FP32 = "fp32"


class Optimizer(Enum):
    """Supported optimizers."""
    ADAM = "adam"
    MUON = "muon"


class ContextPhase(Enum):
    """Phased context extension stages."""
    CORE = "core"           # 8,192 tokens
    EXTENDED = "extended"   # 32,768 tokens
    LONG_RANGE = "long"     # 131,072 tokens


# ---------------------------------------------------------------------------
# Data curation
# ---------------------------------------------------------------------------
@dataclass
class DeltaExtractionConfig:
    """Configuration for commit-delta extraction pipeline."""
    max_files_per_commit: int = 20
    max_diff_tokens: int = 4096
    context_functions: int = 3
    require_cve_link: bool = True
    output_dir: Path = Path("data/deltas")
    sources: list[str] = field(default_factory=lambda: [
        "cvefixes",
        "osv",
        "ghsa",
        "nvd",
    ])

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)


@dataclass
class DataIntegrityConfig:
    """Configuration for dataset supply-chain integrity."""
    hash_algorithm: str = "sha256"
    enable_merkle_tree: bool = True
    provenance_tracking: bool = True
    contamination_scan: bool = True
    audit_sample_ratio: float = 0.05


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
@dataclass
class TrainingConfig:
    """Configuration for SFT and DPO training phases."""
    model_name: str = "google/gemma-4-31b"
    precision: Precision = Precision.NVFP4
    optimizer: Optimizer = Optimizer.ADAM

    # Adam defaults (validated with NVFP4)
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    weight_decay: float = 0.1

    # Learning rates
    sft_lr: float = 2e-5
    dpo_lr: float = 5e-7
    dpo_beta: float = 0.1
    lr_schedule: str = "cosine_with_warmup"
    warmup_steps: int = 500

    # Parallelism (ZeRO-2)
    zero_stage: int = 2
    num_gpus: int = 8
    gpu_memory_gb: float = 192.0

    # NVFP4-specific
    nvfp4_rht_size: int = 16
    nvfp4_high_precision_ratio: float = 0.15
    nvfp4_stochastic_rounding_gradients: bool = True
    nvfp4_2d_weight_scaling: bool = True
    nvfp4_block_size: int = 16

    # Phased context extension
    context_phases: dict[str, dict[str, int | float]] = field(default_factory=lambda: {
        "core": {"seq_length": 8192, "batch_size": 8, "epochs": 3, "days": 14},
        "extended": {"seq_length": 32768, "batch_size": 2, "epochs": 1, "days": 5},
        "long": {"seq_length": 131072, "batch_size": 1, "epochs": 1, "days": 2},
    })

    # Gradient accumulation
    gradient_accumulation_steps: int = 4

    # Checkpointing
    checkpoint_dir: Path = Path("checkpoints")
    save_steps: int = 500
    eval_steps: int = 250

    @property
    def effective_batch_size(self) -> int:
        """Compute effective batch size across all GPUs."""
        phase = self.context_phases.get("core", {})
        per_device = int(phase.get("batch_size", 8))
        return self.num_gpus * per_device * self.gradient_accumulation_steps

    @property
    def per_gpu_fixed_memory_gb(self) -> float:
        """Estimate fixed VRAM overhead per GPU under ZeRO-2."""
        param_count = 31e9
        model_weights_gb = (param_count * 0.5) / 1e9  # NVFP4
        # ZeRO-2 shards optimizer states + gradients
        optimizer_shard_gb = (param_count * 12) / (1e9 * self.num_gpus)  # FP32 master + m + v
        gradient_shard_gb = (param_count * 4) / (1e9 * self.num_gpus)   # FP32 gradients
        return model_weights_gb + optimizer_shard_gb + gradient_shard_gb

    @property
    def per_gpu_activation_budget_gb(self) -> float:
        """Available VRAM for activations per GPU."""
        return self.gpu_memory_gb - self.per_gpu_fixed_memory_gb


# ---------------------------------------------------------------------------
# DPO alignment
# ---------------------------------------------------------------------------
@dataclass
class DPOSynthesisConfig:
    """Configuration for automated DPO preference-pair synthesis."""
    sandbox_image: str = "sentinel-sandbox:latest"
    sandbox_timeout_seconds: int = 120
    fuzz_duration_seconds: int = 60
    fuzz_tool: str = "afl++"
    target_pairs: int = 50_000
    differential_fuzzing: bool = True
    max_retries_per_prompt: int = 5
    output_dir: Path = Path("data/dpo_pairs")

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------
@dataclass
class AgentConfig:
    """Configuration for the SENTINEL agentic harness."""
    architect_model: str = "google/gemma-4-31b"
    executor_model: str = "deepseek-ai/deepseek-coder-v2"
    max_tool_calls: int = 200
    session_timeout_minutes: int = 30
    working_memory_tokens: int = 65536
    retrieval_chunk_tokens: int = 4096
    max_executor_retries: int = 3

    # Tool configuration
    enable_lsp: bool = True
    enable_terminal: bool = True
    enable_semantic_search: bool = True
    enable_compiler: bool = True
    enable_fuzzer: bool = True
    enable_sast: bool = True

    # LSP servers
    lsp_servers: dict[str, str] = field(default_factory=lambda: {
        "python": "pyright",
        "c": "clangd",
        "go": "gopls",
        "rust": "rust-analyzer",
        "javascript": "typescript-language-server",
    })


# ---------------------------------------------------------------------------
# HOTL
# ---------------------------------------------------------------------------
class AutonomyTier(Enum):
    """Human-on-the-loop autonomy levels."""
    FULL_AUTONOMY = 1       # Read-only scanning, report generation
    AUTO_WITH_NOTIFY = 2    # Endpoint isolation, WAF injection
    HUMAN_REQUIRED = 3      # Production code deployment


@dataclass
class HOTLConfig:
    """Configuration for Human-On-The-Loop integration."""
    notification_channel: str = "slack"
    notification_timeout_seconds: int = 30
    kill_switch_enabled: bool = True
    audit_log_dir: Path = Path("logs/audit")
    telemetry_enabled: bool = True

    # Action → tier mapping
    tier_mapping: dict[str, AutonomyTier] = field(default_factory=lambda: {
        "scan_repository": AutonomyTier.FULL_AUTONOMY,
        "generate_report": AutonomyTier.FULL_AUTONOMY,
        "run_fuzzer": AutonomyTier.FULL_AUTONOMY,
        "isolate_endpoint": AutonomyTier.AUTO_WITH_NOTIFY,
        "inject_waf_rule": AutonomyTier.AUTO_WITH_NOTIFY,
        "rotate_certificate": AutonomyTier.AUTO_WITH_NOTIFY,
        "deploy_patch": AutonomyTier.HUMAN_REQUIRED,
        "modify_infrastructure": AutonomyTier.HUMAN_REQUIRED,
        "change_access_control": AutonomyTier.HUMAN_REQUIRED,
    })

    def __post_init__(self) -> None:
        self.audit_log_dir = Path(self.audit_log_dir)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@dataclass
class EvaluationConfig:
    """Configuration for benchmark evaluation."""
    benchmarks: list[str] = field(default_factory=lambda: [
        "cyberseceval3",
        "cvefixes",
        "swebench",
        "purplellama",
        "redteam",
        "mitre_attack",
    ])
    num_runs: int = 3
    confidence_level: float = 0.95
    statistical_test: str = "wilcoxon"
    correction: str = "bonferroni"
    results_dir: Path = Path("results/evaluation")

    def __post_init__(self) -> None:
        self.results_dir = Path(self.results_dir)


# ---------------------------------------------------------------------------
# Top-level project config
# ---------------------------------------------------------------------------
@dataclass
class SentinelConfig:
    """Master configuration for the SENTINEL system."""
    project_name: str = "sentinel"
    project_dir: Path = Path(".")
    data: DeltaExtractionConfig = field(default_factory=DeltaExtractionConfig)
    integrity: DataIntegrityConfig = field(default_factory=DataIntegrityConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    dpo: DPOSynthesisConfig = field(default_factory=DPOSynthesisConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    hotl: HOTLConfig = field(default_factory=HOTLConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    @classmethod
    def from_env(cls) -> SentinelConfig:
        """Build config with overrides from environment variables."""
        config = cls()
        if env_model := os.environ.get("SENTINEL_MODEL"):
            config.training.model_name = env_model
        if env_precision := os.environ.get("SENTINEL_PRECISION"):
            config.training.precision = Precision(env_precision)
        if env_gpus := os.environ.get("SENTINEL_NUM_GPUS"):
            config.training.num_gpus = int(env_gpus)
        return config
