"""CLI: Run evaluation benchmarks."""

from __future__ import annotations

import json
import logging
from pathlib import Path


def main() -> None:
    """Main entry point for sentinel-eval."""
    import argparse

    parser = argparse.ArgumentParser(
        description="SENTINEL: Evaluation & Benchmarking",
    )
    parser.add_argument(
        "--model", type=Path, required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--benchmarks", nargs="+",
        default=["cyberseceval3", "cvefixes", "swebench", "purplellama", "redteam"],
        help="Benchmarks to run",
    )
    parser.add_argument(
        "--num-runs", type=int, default=3,
        help="Number of runs per benchmark (for statistical significance)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/evaluation"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--check-readiness", action="store_true",
        help="Check deployment readiness criteria",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    from sentinel.config import EvaluationConfig
    from sentinel.evaluation import (
        EvaluationRunner,
        check_deployment_readiness,
    )

    config = EvaluationConfig(
        benchmarks=args.benchmarks,
        num_runs=args.num_runs,
        results_dir=args.output_dir,
    )

    runner = EvaluationRunner(config)

    print(f"\n{'='*60}")
    print(f"SENTINEL Evaluation Framework")
    print(f"{'='*60}")
    print(f"  Model:      {args.model}")
    print(f"  Benchmarks: {', '.join(args.benchmarks)}")
    print(f"  Runs each:  {args.num_runs}")
    print(f"{'='*60}")

    results = runner.run_all_benchmarks(str(args.model))

    for benchmark, result in results.items():
        status = "✓ PASS" if result["passed"] else "✗ FAIL"
        print(f"\n  {benchmark}: {result['mean_score']:.4f} "
              f"(±{result['std_dev']:.4f}) [{status}]"
              f"  threshold={result['threshold']}")

    runner.save_results()

    if args.check_readiness:
        readiness = check_deployment_readiness(results)
        print(f"\n{'='*60}")
        print(f"Deployment Readiness: {'✓ READY' if readiness['deployment_ready'] else '✗ NOT READY'}")
        print(f"  Criteria met: {readiness['passed_count']}/{readiness['total_criteria']}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
