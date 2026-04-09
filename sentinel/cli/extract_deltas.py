"""CLI: Extract commit deltas from vulnerability databases."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Optional


def main() -> None:
    """Main entry point for sentinel-extract."""
    import argparse

    parser = argparse.ArgumentParser(
        description="SENTINEL: Commit Delta Extraction Pipeline",
    )
    parser.add_argument(
        "--source", choices=["cvefixes", "osv", "ghsa", "all"],
        default="all", help="Data source to extract from",
    )
    parser.add_argument(
        "--cvefixes-path", type=Path, default=None,
        help="Path to CVEFixes dataset directory",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/deltas"),
        help="Output directory for extracted deltas",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Maximum advisories per source (0 = unlimited)",
    )
    parser.add_argument(
        "--max-files", type=int, default=20,
        help="Reject commits touching more than N files",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=4096,
        help="Reject diffs longer than N tokens",
    )
    parser.add_argument(
        "--verify-integrity", action="store_true",
        help="Verify dataset integrity after extraction",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    from sentinel.config import DeltaExtractionConfig
    from sentinel.data.commit_delta import (
        CVEFixesAdapter,
        CommitDeltaPipeline,
        GHSAAdapter,
        OSVAdapter,
    )
    from sentinel.data.integrity import DatasetIntegrityManager, ProvenanceRecord

    # Build config
    config = DeltaExtractionConfig(
        max_files_per_commit=args.max_files,
        max_diff_tokens=args.max_tokens,
        output_dir=args.output_dir,
    )

    # Build pipeline
    pipeline = CommitDeltaPipeline(config)

    if args.source in ("cvefixes", "all") and args.cvefixes_path:
        pipeline.register_adapter("cvefixes", CVEFixesAdapter(args.cvefixes_path))
    if args.source in ("osv", "all"):
        pipeline.register_adapter("osv", OSVAdapter())
    if args.source in ("ghsa", "all"):
        pipeline.register_adapter("ghsa", GHSAAdapter())

    # Run extraction
    integrity = DatasetIntegrityManager(output_dir=args.output_dir)
    deltas = []

    for delta in pipeline.run(limit_per_source=args.limit):
        record = ProvenanceRecord(
            sample_id=delta.delta_id,
            content_hash=delta.content_hash,
            source_commit_sha=delta.patched_commit.commit_sha if delta.patched_commit else "",
            cve_id=delta.cve_id,
            source_database=delta.source.value,
            cwe_ids=delta.cwe_ids,
        )
        if integrity.register_sample(record):
            deltas.append(delta)

    # Save dataset
    output_path = args.output_dir / "commit_deltas.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for delta in deltas:
            f.write(json.dumps(delta.to_training_sample()) + "\n")

    # Finalize integrity
    root_hash = integrity.finalize()

    # Summary
    stats = pipeline.stats
    print(f"\n{'='*60}")
    print(f"SENTINEL Commit Delta Extraction Complete")
    print(f"{'='*60}")
    print(f"  Total advisories scanned: {stats['total_advisories']}")
    print(f"  Commit pairs resolved:    {stats['resolved_pairs']}")
    print(f"  Passed quality filters:   {stats['passed_filters']}")
    print(f"  Output:                   {output_path}")
    print(f"  Dataset root hash:        {root_hash[:32]}...")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
