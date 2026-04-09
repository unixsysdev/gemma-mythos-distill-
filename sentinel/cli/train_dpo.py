"""CLI: Run DPO alignment training."""

from __future__ import annotations

import logging
from pathlib import Path


def main() -> None:
    """Main entry point for sentinel-train-dpo."""
    import argparse

    parser = argparse.ArgumentParser(
        description="SENTINEL: DPO Alignment Training",
    )
    parser.add_argument(
        "--model", type=Path, required=True,
        help="Path to SFT checkpoint",
    )
    parser.add_argument(
        "--dataset", type=Path, default=None,
        help="Path to preference pairs (JSONL). If not provided, runs synthesis.",
    )
    parser.add_argument(
        "--synthesize", action="store_true",
        help="Run DPO synthesis pipeline to generate preference pairs",
    )
    parser.add_argument(
        "--vulnerable-snippets", type=Path, default=None,
        help="Path to vulnerable code snippets for synthesis",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("checkpoints/dpo"),
        help="Output directory",
    )
    parser.add_argument(
        "--target-pairs", type=int, default=50000,
        help="Target number of preference pairs to synthesize",
    )
    parser.add_argument(
        "--beta", type=float, default=0.1,
        help="DPO temperature parameter",
    )
    parser.add_argument(
        "--lr", type=float, default=5e-7,
        help="DPO learning rate",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    from sentinel.config import DPOSynthesisConfig

    print(f"\n{'='*60}")
    print(f"SENTINEL DPO Alignment Pipeline")
    print(f"{'='*60}")
    print(f"  SFT checkpoint: {args.model}")
    print(f"  β (temperature): {args.beta}")
    print(f"  Learning rate:   {args.lr}")

    if args.synthesize:
        print(f"  Mode:            Synthesis + Training")
        print(f"  Target pairs:    {args.target_pairs}")

        config = DPOSynthesisConfig(
            target_pairs=args.target_pairs,
            output_dir=args.output_dir / "synthesis",
        )

        from sentinel.alignment.dpo_synthesis import (
            DPOSynthesisPipeline,
            ModelInterface,
        )

        model = ModelInterface(model_path=str(args.model))
        pipeline = DPOSynthesisPipeline(config=config, model=model)
        print(f"  Pipeline ready. Would process snippets from: {args.vulnerable_snippets}")

    else:
        print(f"  Mode:            Training only")
        print(f"  Dataset:         {args.dataset}")

    print(f"  Output:          {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
