"""
Commit Delta Extraction Pipeline.

Extracts the semantic transformation between vulnerable and patched code
states from version control history. This is the core data curation strategy
described in the SENTINEL specification — training exclusively on the
mathematical delta between vulnerable commits and their secure counterparts.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Generator, Optional, Protocol

from sentinel.config import DeltaExtractionConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------
class VulnerabilitySource(Enum):
    """Provenance source for vulnerability data."""
    CVEFIXES = "cvefixes"
    OSV = "osv"
    GHSA = "ghsa"
    NVD = "nvd"


@dataclass(frozen=True)
class CommitRef:
    """Reference to a specific commit in a repository."""
    repo_url: str
    commit_sha: str
    branch: str = "main"


@dataclass
class DiffHunk:
    """A single hunk from a unified diff."""
    file_path: str
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    old_content: str
    new_content: str
    context_before: str = ""
    context_after: str = ""


@dataclass
class CommitDelta:
    """
    A single training sample representing the transformation
    from vulnerable to patched code.

    This is the core data unit defined in Definition 1 of the
    doctoral research proposal:

        Δ(c_v, c_p) = (diff(c_v, c_p), context_±3(c_v, c_p), CWE(c_v), CVSS(c_v))
    """
    # Identification
    delta_id: str = ""
    cve_id: str = ""
    advisory_url: str = ""
    source: VulnerabilitySource = VulnerabilitySource.CVEFIXES

    # Commit references
    vulnerable_commit: Optional[CommitRef] = None
    patched_commit: Optional[CommitRef] = None

    # Code content
    vulnerable_code: str = ""
    patched_code: str = ""
    unified_diff: str = ""
    diff_hunks: list[DiffHunk] = field(default_factory=list)
    function_context: str = ""

    # Taxonomy
    cwe_ids: list[str] = field(default_factory=list)
    cvss_score: float = 0.0
    cvss_vector: str = ""
    severity: str = "UNKNOWN"

    # Explanation (for training)
    explanation: str = ""
    secure_alternative_rationale: str = ""

    # Metadata
    language: str = ""
    files_changed: int = 0
    diff_token_count: int = 0

    # Integrity
    content_hash: str = ""

    def compute_hash(self) -> str:
        """Compute SHA-256 hash of the delta content for integrity verification."""
        content = json.dumps({
            "cve_id": self.cve_id,
            "vulnerable_code": self.vulnerable_code,
            "patched_code": self.patched_code,
            "unified_diff": self.unified_diff,
        }, sort_keys=True)
        self.content_hash = hashlib.sha256(content.encode()).hexdigest()
        return self.content_hash

    def to_training_sample(self) -> dict[str, Any]:
        """Convert to the structured format used for SFT training."""
        return {
            "delta_id": self.delta_id,
            "cve_id": self.cve_id,
            "vulnerable_code": self.vulnerable_code,
            "patched_code": self.patched_code,
            "diff": self.unified_diff,
            "cwe": self.cwe_ids,
            "severity": self.severity,
            "cvss_vector": self.cvss_vector,
            "explanation": self.explanation,
            "language": self.language,
            "content_hash": self.content_hash,
        }


# ---------------------------------------------------------------------------
# Source adapters
# ---------------------------------------------------------------------------
class VulnerabilitySourceAdapter(Protocol):
    """Protocol for vulnerability data source adapters."""

    def fetch_advisories(self, limit: int = 0) -> Generator[dict[str, Any], None, None]:
        """Yield raw advisory records from the source."""
        ...

    def resolve_commit_pair(
        self, advisory: dict[str, Any]
    ) -> Optional[tuple[CommitRef, CommitRef]]:
        """Resolve advisory to (vulnerable_commit, patched_commit) pair."""
        ...


class CVEFixesAdapter:
    """
    Adapter for the CVEFixes dataset (Bhandari et al., 2021).

    5,495 vulnerability-fixing commits across 1,754 open-source projects.
    """

    def __init__(self, dataset_path: Path) -> None:
        self.dataset_path = dataset_path
        self._validate_path()

    def _validate_path(self) -> None:
        if not self.dataset_path.exists():
            logger.warning(
                "CVEFixes dataset path does not exist: %s. "
                "Download from https://zenodo.org/record/7029359",
                self.dataset_path,
            )

    def fetch_advisories(self, limit: int = 0) -> Generator[dict[str, Any], None, None]:
        """Yield advisory records from the CVEFixes database."""
        # In production: query the SQLite database
        # Here we define the interface contract
        db_path = self.dataset_path / "CVEfixes.db"
        if not db_path.exists():
            logger.error("CVEFixes database not found at %s", db_path)
            return

        import sqlite3
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        query = """
            SELECT
                c.cve_id,
                c.repo_url,
                fc.hash AS fix_hash,
                fc.message AS commit_message,
                m.cwe_id
            FROM cve AS c
            JOIN fixes AS f ON c.cve_id = f.cve_id
            JOIN fix_commits AS fc ON f.repo_url = fc.repo_url AND f.hash = fc.hash
            LEFT JOIN cwe_classification AS m ON c.cve_id = m.cve_id
            ORDER BY c.cve_id
        """
        if limit > 0:
            query += f" LIMIT {limit}"

        cursor.execute(query)
        for row in cursor:
            yield dict(row)

        conn.close()

    def resolve_commit_pair(
        self, advisory: dict[str, Any]
    ) -> Optional[tuple[CommitRef, CommitRef]]:
        """Resolve CVEFixes entry to commit pair."""
        repo_url = advisory.get("repo_url", "")
        fix_hash = advisory.get("fix_hash", "")
        if not repo_url or not fix_hash:
            return None

        patched = CommitRef(repo_url=repo_url, commit_sha=fix_hash)
        # Parent commit is the vulnerable state
        vulnerable = CommitRef(repo_url=repo_url, commit_sha=f"{fix_hash}~1")
        return vulnerable, patched


class OSVAdapter:
    """Adapter for the OSV.dev vulnerability database."""

    API_BASE = "https://api.osv.dev/v1"

    def __init__(self, ecosystems: Optional[list[str]] = None) -> None:
        self.ecosystems = ecosystems or ["PyPI", "npm", "Go", "crates.io"]

    def fetch_advisories(self, limit: int = 0) -> Generator[dict[str, Any], None, None]:
        """Query OSV API for advisories with fix commits.

        Uses the /v1/query endpoint with the documented package-level
        query schema. OSV requires at least a package name + ecosystem
        or a commit hash — a bare {"ecosystem": ...} is not a valid query.

        For ecosystem-level enumeration we use /v1/vulns/{ecosystem}
        listing then fetch each advisory via /v1/vulns/{id}.
        """
        import httpx

        for ecosystem in self.ecosystems:
            page_token: Optional[str] = None
            count = 0

            while True:
                if limit > 0 and count >= limit:
                    return

                try:
                    # Use the querybatch endpoint for ecosystem-level listing
                    # or paginate through ecosystem vulns via the list API
                    params: dict[str, Any] = {"ecosystem": ecosystem}
                    if page_token:
                        params["page_token"] = page_token

                    response = httpx.get(
                        f"{self.API_BASE}/vulns",
                        params=params,
                        timeout=30.0,
                    )
                    response.raise_for_status()
                    data = response.json()
                    vulns = data.get("vulns", [])

                    if not vulns:
                        break

                    for vuln_summary in vulns:
                        if limit > 0 and count >= limit:
                            return

                        vuln_id = vuln_summary.get("id", "")
                        if not vuln_id:
                            continue

                        # Fetch full advisory details
                        try:
                            detail_resp = httpx.get(
                                f"{self.API_BASE}/vulns/{vuln_id}",
                                timeout=15.0,
                            )
                            detail_resp.raise_for_status()
                            vuln = detail_resp.json()
                        except Exception as detail_err:
                            logger.debug("Failed to fetch OSV vuln %s: %s", vuln_id, detail_err)
                            continue

                        # Only yield vulns with fix commits in their ranges
                        affected = vuln.get("affected", [])
                        has_fix = False
                        for aff in affected:
                            for r in aff.get("ranges", []):
                                if r.get("type") != "GIT":
                                    continue
                                events = r.get("events", [])
                                if any(e.get("fixed") for e in events):
                                    has_fix = True
                                    break
                            if has_fix:
                                break

                        if has_fix:
                            yield vuln
                            count += 1

                    # Pagination
                    page_token = data.get("next_page_token")
                    if not page_token:
                        break

                except Exception as e:
                    logger.error("Failed to query OSV for ecosystem %s: %s", ecosystem, e)
                    break

    def resolve_commit_pair(
        self, advisory: dict[str, Any]
    ) -> Optional[tuple[CommitRef, CommitRef]]:
        """Extract commit pair from OSV advisory."""
        for ref in advisory.get("references", []):
            if ref.get("type") == "FIX":
                url = ref.get("url", "")
                # Parse GitHub commit URL
                if "github.com" in url and "/commit/" in url:
                    parts = url.split("/commit/")
                    repo_url = parts[0]
                    fix_sha = parts[1].split("#")[0]
                    return (
                        CommitRef(repo_url=repo_url, commit_sha=f"{fix_sha}~1"),
                        CommitRef(repo_url=repo_url, commit_sha=fix_sha),
                    )
        return None


class GHSAAdapter:
    """Adapter for GitHub Security Advisories."""

    def __init__(self, github_token: Optional[str] = None) -> None:
        import os
        self.token = github_token or os.environ.get("GITHUB_TOKEN", "")

    def fetch_advisories(self, limit: int = 0) -> Generator[dict[str, Any], None, None]:
        """Fetch advisories via GitHub GraphQL API."""
        import httpx

        query = """
        query($cursor: String) {
            securityAdvisories(first: 100, after: $cursor, orderBy: {field: PUBLISHED_AT, direction: DESC}) {
                nodes {
                    ghsaId
                    summary
                    severity
                    cwes(first: 5) { nodes { cweId } }
                    vulnerabilities(first: 10) {
                        nodes {
                            package { ecosystem name }
                            firstPatchedVersion { identifier }
                        }
                    }
                    references { url }
                }
                pageInfo { hasNextPage endCursor }
            }
        }
        """
        cursor: Optional[str] = None
        count = 0

        while True:
            if limit > 0 and count >= limit:
                return

            try:
                response = httpx.post(
                    "https://api.github.com/graphql",
                    json={"query": query, "variables": {"cursor": cursor}},
                    headers={"Authorization": f"Bearer {self.token}"},
                    timeout=30.0,
                )
                response.raise_for_status()
                data = response.json()
                advisories_data = data.get("data", {}).get("securityAdvisories", {})
                nodes = advisories_data.get("nodes", [])

                for node in nodes:
                    if limit > 0 and count >= limit:
                        return
                    yield node
                    count += 1

                page_info = advisories_data.get("pageInfo", {})
                if not page_info.get("hasNextPage"):
                    break
                cursor = page_info.get("endCursor")

            except Exception as e:
                logger.error("GHSA fetch failed: %s", e)
                break

    def resolve_commit_pair(
        self, advisory: dict[str, Any]
    ) -> Optional[tuple[CommitRef, CommitRef]]:
        """Extract commit pair from GHSA references."""
        for ref in advisory.get("references", []):
            url = ref.get("url", "") if isinstance(ref, dict) else ref
            if "github.com" in url and "/commit/" in url:
                parts = url.split("/commit/")
                repo_url = parts[0]
                fix_sha = parts[1].split("#")[0].split("?")[0]
                return (
                    CommitRef(repo_url=repo_url, commit_sha=f"{fix_sha}~1"),
                    CommitRef(repo_url=repo_url, commit_sha=fix_sha),
                )
        return None


# ---------------------------------------------------------------------------
# Diff extraction
# ---------------------------------------------------------------------------
class DiffExtractor:
    """
    Extracts unified diffs and function-level context from commit pairs.

    Uses pygit2 for git operations and tree-sitter for function boundary
    detection across multiple languages.
    """

    # Supported language extensions → tree-sitter language names
    LANGUAGE_MAP: dict[str, str] = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".c": "c",
        ".cpp": "cpp",
        ".h": "c",
        ".go": "go",
        ".rs": "rust",
        ".java": "java",
        ".rb": "ruby",
        ".php": "php",
    }

    def __init__(self, workspace_dir: Path) -> None:
        self.workspace_dir = workspace_dir
        self.workspace_dir.mkdir(parents=True, exist_ok=True)

    def extract_diff(
        self,
        vulnerable: CommitRef,
        patched: CommitRef,
        context_functions: int = 3,
    ) -> Optional[list[DiffHunk]]:
        """
        Clone the repository and extract diff hunks between commits.

        Args:
            vulnerable: The vulnerable commit reference
            patched: The patched commit reference
            context_functions: Number of enclosing functions for context

        Returns:
            List of DiffHunk objects, or None if extraction fails
        """
        try:
            import pygit2

            repo_name = vulnerable.repo_url.rstrip("/").split("/")[-1]
            repo_path = self.workspace_dir / repo_name

            # Clone if not exists
            if not repo_path.exists():
                logger.info("Cloning %s to %s", vulnerable.repo_url, repo_path)
                pygit2.clone_repository(
                    vulnerable.repo_url + ".git",
                    str(repo_path),
                )

            repo = pygit2.Repository(str(repo_path))

            # Resolve commits
            try:
                patch_oid = repo.revparse_single(patched.commit_sha).id
                vuln_oid = repo.revparse_single(vulnerable.commit_sha).id
            except KeyError:
                logger.error(
                    "Could not resolve commits: %s or %s",
                    vulnerable.commit_sha,
                    patched.commit_sha,
                )
                return None

            # Compute diff
            diff = repo.diff(
                repo.get(vuln_oid),
                repo.get(patch_oid),
                context_lines=context_functions * 10,
            )

            hunks: list[DiffHunk] = []
            for patch in diff:
                file_path = patch.delta.new_file.path
                for hunk in patch.hunks:
                    old_lines = []
                    new_lines = []
                    for line in hunk.lines:
                        if line.origin in ("+",):
                            new_lines.append(line.content)
                        elif line.origin in ("-",):
                            old_lines.append(line.content)

                    hunks.append(DiffHunk(
                        file_path=file_path,
                        old_start=hunk.old_start,
                        old_count=hunk.old_lines,
                        new_start=hunk.new_start,
                        new_count=hunk.new_lines,
                        old_content="".join(old_lines),
                        new_content="".join(new_lines),
                    ))

            return hunks if hunks else None

        except ImportError:
            logger.error("pygit2 not installed. Run: pip install pygit2")
            return None
        except Exception as e:
            logger.error("Diff extraction failed: %s", e)
            return None

    def detect_language(self, file_path: str) -> str:
        """Detect programming language from file extension."""
        suffix = Path(file_path).suffix.lower()
        return self.LANGUAGE_MAP.get(suffix, "unknown")


# ---------------------------------------------------------------------------
# Main extraction pipeline
# ---------------------------------------------------------------------------
class CommitDeltaPipeline:
    """
    End-to-end pipeline for extracting commit deltas across all sources.

    This implements the extraction protocol from Section 4.2.1 of the
    doctoral research proposal.
    """

    def __init__(self, config: DeltaExtractionConfig) -> None:
        self.config = config
        self.diff_extractor = DiffExtractor(
            workspace_dir=config.output_dir / "repos"
        )
        self._adapters: dict[str, VulnerabilitySourceAdapter] = {}
        self._stats: dict[str, int] = {
            "total_advisories": 0,
            "resolved_pairs": 0,
            "passed_filters": 0,
            "failed_too_many_files": 0,
            "failed_too_long": 0,
            "failed_no_cve": 0,
            "failed_extraction": 0,
        }

    def register_adapter(
        self, name: str, adapter: VulnerabilitySourceAdapter
    ) -> None:
        """Register a vulnerability source adapter."""
        self._adapters[name] = adapter
        logger.info("Registered adapter: %s", name)

    def run(self, limit_per_source: int = 0) -> Generator[CommitDelta, None, None]:
        """
        Execute the full extraction pipeline.

        Yields CommitDelta objects that have passed all quality filters.
        """
        for source_name, adapter in self._adapters.items():
            logger.info("Processing source: %s", source_name)

            for advisory in adapter.fetch_advisories(limit=limit_per_source):
                self._stats["total_advisories"] += 1

                # Resolve commit pair
                pair = adapter.resolve_commit_pair(advisory)
                if pair is None:
                    continue
                self._stats["resolved_pairs"] += 1

                vulnerable, patched = pair

                # Extract diff
                hunks = self.diff_extractor.extract_diff(
                    vulnerable,
                    patched,
                    context_functions=self.config.context_functions,
                )
                if hunks is None:
                    self._stats["failed_extraction"] += 1
                    continue

                # Build delta
                delta = self._build_delta(advisory, vulnerable, patched, hunks, source_name)
                if delta is None:
                    continue

                # Apply quality filters
                if not self._passes_quality_filters(delta):
                    continue

                self._stats["passed_filters"] += 1
                delta.compute_hash()
                yield delta

        logger.info("Extraction complete. Stats: %s", self._stats)

    def _build_delta(
        self,
        advisory: dict[str, Any],
        vulnerable: CommitRef,
        patched: CommitRef,
        hunks: list[DiffHunk],
        source_name: str,
    ) -> Optional[CommitDelta]:
        """Construct a CommitDelta from raw components."""
        try:
            cve_id = (
                advisory.get("cve_id", "")
                or advisory.get("id", "")
                or advisory.get("ghsaId", "")
            )

            cwe_ids = []
            # CVEFixes format
            if "cwe_id" in advisory:
                cwe_ids = [advisory["cwe_id"]]
            # GHSA format
            elif "cwes" in advisory:
                cwe_nodes = advisory.get("cwes", {}).get("nodes", [])
                cwe_ids = [n.get("cweId", "") for n in cwe_nodes if n.get("cweId")]

            vulnerable_code = "\n".join(h.old_content for h in hunks)
            patched_code = "\n".join(h.new_content for h in hunks)
            unified_diff = "\n".join(
                f"--- {h.file_path}\n+++ {h.file_path}\n"
                f"@@ -{h.old_start},{h.old_count} +{h.new_start},{h.new_count} @@\n"
                f"-{h.old_content}\n+{h.new_content}"
                for h in hunks
            )

            language = self.diff_extractor.detect_language(hunks[0].file_path) if hunks else ""

            return CommitDelta(
                delta_id=f"{source_name}:{cve_id}:{patched.commit_sha[:8]}",
                cve_id=cve_id,
                source=self._resolve_source(source_name),
                vulnerable_commit=vulnerable,
                patched_commit=patched,
                vulnerable_code=vulnerable_code,
                patched_code=patched_code,
                unified_diff=unified_diff,
                diff_hunks=hunks,
                cwe_ids=cwe_ids,
                language=language,
                files_changed=len({h.file_path for h in hunks}),
                diff_token_count=len(unified_diff.split()),
            )
        except Exception as e:
            logger.error("Failed to build delta: %s", e)
            return None

    def _passes_quality_filters(self, delta: CommitDelta) -> bool:
        """
        Apply quality filters as specified in Section 2.1 of the spec:
        - Reject commits touching >20 files
        - Reject diffs >4,096 tokens
        - Require linked CVE or security advisory
        """
        if delta.files_changed > self.config.max_files_per_commit:
            self._stats["failed_too_many_files"] += 1
            logger.debug("Rejected %s: too many files (%d)", delta.delta_id, delta.files_changed)
            return False

        if delta.diff_token_count > self.config.max_diff_tokens:
            self._stats["failed_too_long"] += 1
            logger.debug("Rejected %s: diff too long (%d tokens)", delta.delta_id, delta.diff_token_count)
            return False

        if self.config.require_cve_link and not delta.cve_id:
            self._stats["failed_no_cve"] += 1
            logger.debug("Rejected %s: no CVE link", delta.delta_id)
            return False

        return True

    @staticmethod
    def _resolve_source(source_name: str) -> VulnerabilitySource:
        """Resolve source name string to enum value.

        Uses the enum value constructor (not __members__) because
        adapter names are lowercase ('osv', 'ghsa') which match
        enum *values*, not enum *names* (OSV, GHSA).
        """
        try:
            return VulnerabilitySource(source_name)
        except ValueError:
            logger.warning("Unknown source '%s', defaulting to CVEFIXES", source_name)
            return VulnerabilitySource.CVEFIXES


    @property
    def stats(self) -> dict[str, int]:
        """Return extraction statistics."""
        return dict(self._stats)
