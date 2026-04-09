"""
Dataset integrity verification using SHA-256 Merkle trees.

Implements the supply chain integrity requirements from Section 2.3
of the SENTINEL specification.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class MerkleNode:
    """A node in the Merkle tree."""
    hash_value: str
    left: Optional[MerkleNode] = None
    right: Optional[MerkleNode] = None
    data_id: str = ""
    is_leaf: bool = False


class MerkleTree:
    """
    Tamper-evident Merkle tree for dataset provenance.

    Each training sample's SHA-256 hash forms a leaf node.
    Any modification to any sample changes the root hash,
    enabling efficient integrity verification.
    """

    def __init__(self, hash_algorithm: str = "sha256") -> None:
        self.hash_algorithm = hash_algorithm
        self._leaves: list[MerkleNode] = []
        self._root: Optional[MerkleNode] = None

    def _hash(self, data: str) -> str:
        """Compute hash of a string."""
        h = hashlib.new(self.hash_algorithm)
        h.update(data.encode("utf-8"))
        return h.hexdigest()

    def _hash_pair(self, left_hash: str, right_hash: str) -> str:
        """Compute hash of two concatenated hashes."""
        return self._hash(left_hash + right_hash)

    def add_leaf(self, data_id: str, content_hash: str) -> None:
        """Add a leaf node for a training sample."""
        node = MerkleNode(
            hash_value=content_hash,
            data_id=data_id,
            is_leaf=True,
        )
        self._leaves.append(node)
        self._root = None  # Invalidate cached root

    def build(self) -> str:
        """
        Build the Merkle tree and return the root hash.

        Returns:
            The root hash of the tree.
        """
        if not self._leaves:
            return self._hash("")

        # Ensure even number of leaves by duplicating last if odd
        nodes = list(self._leaves)
        if len(nodes) % 2 != 0:
            nodes.append(nodes[-1])

        while len(nodes) > 1:
            next_level: list[MerkleNode] = []
            for i in range(0, len(nodes), 2):
                left = nodes[i]
                right = nodes[i + 1] if i + 1 < len(nodes) else left
                parent = MerkleNode(
                    hash_value=self._hash_pair(left.hash_value, right.hash_value),
                    left=left,
                    right=right,
                )
                next_level.append(parent)
            nodes = next_level

        self._root = nodes[0]
        return self._root.hash_value

    @property
    def root_hash(self) -> str:
        """Get the root hash, building the tree if necessary."""
        if self._root is None:
            return self.build()
        return self._root.hash_value

    def verify_leaf(self, data_id: str, content_hash: str) -> bool:
        """Verify that a specific leaf exists with the given hash."""
        for leaf in self._leaves:
            if leaf.data_id == data_id and leaf.hash_value == content_hash:
                return True
        return False

    @property
    def leaf_count(self) -> int:
        return len(self._leaves)


@dataclass
class ProvenanceRecord:
    """
    Provenance tracking for a single training sample.

    Links each sample to its source commit SHA, CVE ID, and advisory URL
    as required by the specification.
    """
    sample_id: str
    content_hash: str
    source_commit_sha: str = ""
    cve_id: str = ""
    advisory_url: str = ""
    source_database: str = ""
    extraction_timestamp: str = ""
    cwe_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "content_hash": self.content_hash,
            "source_commit_sha": self.source_commit_sha,
            "cve_id": self.cve_id,
            "advisory_url": self.advisory_url,
            "source_database": self.source_database,
            "extraction_timestamp": self.extraction_timestamp,
            "cwe_ids": self.cwe_ids,
        }


class ContaminationScanner:
    """
    Scans training data against known benchmark samples to prevent
    data leakage into evaluation sets.
    """

    def __init__(self) -> None:
        self._benchmark_hashes: set[str] = set()
        self._benchmark_fingerprints: set[str] = set()

    def load_benchmark_hashes(self, benchmark_name: str, hashes: list[str]) -> None:
        """Register hashes of known benchmark samples."""
        self._benchmark_hashes.update(hashes)
        logger.info(
            "Loaded %d hashes for benchmark %s (total: %d)",
            len(hashes),
            benchmark_name,
            len(self._benchmark_hashes),
        )

    def load_benchmark_fingerprints(
        self, benchmark_name: str, fingerprints: list[str]
    ) -> None:
        """Register content fingerprints (normalized code strings) for fuzzy matching."""
        self._benchmark_fingerprints.update(fingerprints)

    def is_contaminated(self, content_hash: str, content: str = "") -> bool:
        """
        Check if a training sample appears in any registered benchmark.

        Args:
            content_hash: SHA-256 hash of the sample
            content: Raw content for fuzzy matching (optional)

        Returns:
            True if the sample is found in a benchmark (contaminated)
        """
        # Exact hash match
        if content_hash in self._benchmark_hashes:
            return True

        # Fuzzy content match (normalized)
        if content and self._benchmark_fingerprints:
            normalized = self._normalize_code(content)
            fingerprint = hashlib.sha256(normalized.encode()).hexdigest()
            if fingerprint in self._benchmark_fingerprints:
                return True

        return False

    @staticmethod
    def _normalize_code(code: str) -> str:
        """Normalize code for fuzzy matching (strip whitespace, comments)."""
        lines = []
        for line in code.splitlines():
            stripped = line.strip()
            # Skip empty lines and common comment patterns
            if not stripped or stripped.startswith("#") or stripped.startswith("//"):
                continue
            lines.append(stripped)
        return "\n".join(lines)


class DatasetIntegrityManager:
    """
    Orchestrates all integrity verification for the training dataset.

    Combines Merkle tree hashing, provenance tracking, and
    contamination scanning into a single manager.
    """

    def __init__(
        self,
        output_dir: Path,
        hash_algorithm: str = "sha256",
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.merkle_tree = MerkleTree(hash_algorithm=hash_algorithm)
        self.contamination_scanner = ContaminationScanner()
        self._provenance_records: list[ProvenanceRecord] = []
        self._seen_hashes: set[str] = set()

    def register_sample(self, record: ProvenanceRecord) -> bool:
        """
        Register a training sample for integrity tracking.

        Returns:
            False if the sample is contaminated or a duplicate (should be excluded)
        """
        # Check contamination
        if self.contamination_scanner.is_contaminated(record.content_hash):
            logger.warning(
                "Sample %s is contaminated (appears in benchmark). Excluding.",
                record.sample_id,
            )
            return False

        # Check duplicates
        if record.content_hash in self._seen_hashes:
            logger.debug(
                "Sample %s is a duplicate (hash %s). Excluding.",
                record.sample_id,
                record.content_hash[:16],
            )
            return False
        self._seen_hashes.add(record.content_hash)

        # Add to Merkle tree
        self.merkle_tree.add_leaf(record.sample_id, record.content_hash)

        # Store provenance
        self._provenance_records.append(record)
        return True

    def finalize(self) -> str:
        """
        Build the Merkle tree and save all integrity artifacts.

        Returns:
            The root hash of the complete dataset.
        """
        root_hash = self.merkle_tree.build()

        # Save provenance records
        provenance_path = self.output_dir / "provenance.jsonl"
        with open(provenance_path, "w") as f:
            for record in self._provenance_records:
                f.write(json.dumps(record.to_dict()) + "\n")

        # Save root hash
        manifest_path = self.output_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump({
                "root_hash": root_hash,
                "hash_algorithm": self.merkle_tree.hash_algorithm,
                "total_samples": self.merkle_tree.leaf_count,
                "provenance_file": str(provenance_path),
            }, f, indent=2)

        logger.info(
            "Dataset finalized: %d samples, root hash: %s",
            self.merkle_tree.leaf_count,
            root_hash[:16] + "...",
        )
        return root_hash

    def verify(self) -> bool:
        """Verify the integrity of the dataset against stored manifest."""
        manifest_path = self.output_dir / "manifest.json"
        if not manifest_path.exists():
            logger.error("Manifest not found at %s", manifest_path)
            return False

        with open(manifest_path) as f:
            manifest = json.load(f)

        current_root = self.merkle_tree.root_hash
        stored_root = manifest.get("root_hash", "")

        if current_root != stored_root:
            logger.error(
                "Integrity verification FAILED. Current: %s, Stored: %s",
                current_root[:16],
                stored_root[:16],
            )
            return False

        logger.info("Integrity verification PASSED.")
        return True
