"""CLI: Run SFT training with NVFP4 precision."""

from __future__ import annotations

import json
import logging
from pathlib import Path


def main() -> None:
    """Main entry point for sentinel-train-sft."""
    import argparse

    parser = argparse.ArgumentParser(
        description="SENTINEL: Supervised Fine-Tuning with NVFP4",
    )
    parser.add_argument(
        "--model", type=str, default="google/gemma-4-31b",
        help="Base model name or path",
    )
    parser.add_argument(
        "--dataset", type=Path, required=True,
        help="Path to training dataset (JSONL)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("checkpoints/sft"),
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--precision", choices=["nvfp4", "fp8", "bf16"],
        default="nvfp4", help="Training precision format",
    )
    parser.add_argument(
        "--phase", choices=["core", "extended", "long", "all"],
        default="all", help="Context extension phase to run",
    )
    parser.add_argument(
        "--num-gpus", type=int, default=8,
        help="Number of GPUs",
    )
    parser.add_argument(
        "--lr", type=float, default=2e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Generate configs without training",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    from sentinel.config import Precision, TrainingConfig
    from sentinel.training.nvfp4_config import (
        estimate_vram_budget,
        save_configs,
    )
    from sentinel.training.sft import SFTTrainer

    config = TrainingConfig(
        model_name=args.model,
        precision=Precision(args.precision),
        num_gpus=args.num_gpus,
        sft_lr=args.lr,
    )

    # Print VRAM budget
    vram = estimate_vram_budget(config)
    print(f"\n{'='*60}")
    print(f"SENTINEL SFT Training Configuration")
    print(f"{'='*60}")
    print(f"  Model:           {config.model_name}")
    print(f"  Precision:       {config.precision.value}")
    print(f"  GPUs:            {config.num_gpus}")
    print(f"  Effective batch: {config.effective_batch_size}")
    print(f"  VRAM/GPU fixed:  {vram['fixed_total_gb']:.1f} GB")
    print(f"  VRAM/GPU free:   {vram['activation_budget_gb']:.1f} GB")
    print(f"  GPU utilization: {vram['utilization_pct']:.1f}%")
    print(f"{'='*60}")

    if args.dry_run:
        saved = save_configs(config, args.output_dir / "configs")
        print(f"\nDry run: saved {len(saved)} config files")
        for name, path in saved.items():
            print(f"  {name}: {path}")
        return

    trainer = SFTTrainer(
        config=config,
        dataset_path=args.dataset,
        output_dir=args.output_dir,
    )
    summary = trainer.prepare()
    print(f"\nTraining prepared: {json.dumps(summary, indent=2, default=str)}")

    if args.phase == "all":
        trainer.train_all_phases()
    else:
        from sentinel.config import ContextPhase
        phase_map = {
            "core": ContextPhase.CORE,
            "extended": ContextPhase.EXTENDED,
            "long": ContextPhase.LONG_RANGE,
        }
        trainer.train_phase(phase_map[args.phase])


if __name__ == "__main__":
    main()
