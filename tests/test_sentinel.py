"""
Tests for the commit delta extraction pipeline.

Validates the enum mapping fix, adapter contracts, quality filters,
and pipeline orchestration without requiring external dependencies.
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any, Generator, Optional

import pytest

from sentinel.data.commit_delta import (
    CommitDelta,
    CommitDeltaPipeline,
    CommitRef,
    DiffHunk,
    VulnerabilitySource,
    VulnerabilitySourceAdapter,
)
from sentinel.config import DeltaExtractionConfig


# ---------------------------------------------------------------------------
# Enum mapping tests (the bug that was found and fixed)
# ---------------------------------------------------------------------------
class TestVulnerabilitySourceEnum:
    """Verify enum value resolution works with lowercase adapter names."""

    def test_cvefixes_by_value(self) -> None:
        assert VulnerabilitySource("cvefixes") == VulnerabilitySource.CVEFIXES

    def test_osv_by_value(self) -> None:
        assert VulnerabilitySource("osv") == VulnerabilitySource.OSV

    def test_ghsa_by_value(self) -> None:
        assert VulnerabilitySource("ghsa") == VulnerabilitySource.GHSA

    def test_nvd_by_value(self) -> None:
        assert VulnerabilitySource("nvd") == VulnerabilitySource.NVD

    def test_members_does_not_match_values(self) -> None:
        """
        Regression test for the original bug: __members__ checks NAMES
        (CVEFIXES, OSV, etc.), not values ('cvefixes', 'osv', etc.).
        This means `'osv' in VulnerabilitySource.__members__` is False.
        """
        assert "osv" not in VulnerabilitySource.__members__
        assert "OSV" in VulnerabilitySource.__members__

    def test_resolve_source_on_pipeline(self) -> None:
        """Test the _resolve_source static method on the pipeline."""
        config = DeltaExtractionConfig(output_dir=Path("/tmp/test"))
        pipeline = CommitDeltaPipeline(config)

        assert pipeline._resolve_source("osv") == VulnerabilitySource.OSV
        assert pipeline._resolve_source("ghsa") == VulnerabilitySource.GHSA
        assert pipeline._resolve_source("cvefixes") == VulnerabilitySource.CVEFIXES
        # Unknown source should fallback
        assert pipeline._resolve_source("unknown_db") == VulnerabilitySource.CVEFIXES


# ---------------------------------------------------------------------------
# CommitDelta data model tests
# ---------------------------------------------------------------------------
class TestCommitDelta:
    """Test the CommitDelta data model."""

    def test_compute_hash_deterministic(self) -> None:
        delta = CommitDelta(
            cve_id="CVE-2024-1234",
            vulnerable_code="int x = gets(buf);",
            patched_code="int x = fgets(buf, sizeof(buf), stdin);",
            unified_diff="--- old\n+++ new\n-gets(buf)\n+fgets(buf, sizeof(buf), stdin)",
        )
        h1 = delta.compute_hash()
        h2 = delta.compute_hash()
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex

    def test_compute_hash_differs_on_change(self) -> None:
        delta1 = CommitDelta(cve_id="CVE-2024-1234", vulnerable_code="a")
        delta2 = CommitDelta(cve_id="CVE-2024-1234", vulnerable_code="b")
        assert delta1.compute_hash() != delta2.compute_hash()

    def test_to_training_sample_format(self) -> None:
        delta = CommitDelta(
            delta_id="test:CVE-2024-1234:abc12345",
            cve_id="CVE-2024-1234",
            vulnerable_code="bad code",
            patched_code="good code",
            unified_diff="diff",
            cwe_ids=["CWE-787"],
            severity="CRITICAL",
            language="c",
        )
        delta.compute_hash()
        sample = delta.to_training_sample()

        assert sample["delta_id"] == "test:CVE-2024-1234:abc12345"
        assert sample["cve_id"] == "CVE-2024-1234"
        assert sample["vulnerable_code"] == "bad code"
        assert sample["patched_code"] == "good code"
        assert sample["cwe"] == ["CWE-787"]
        assert sample["severity"] == "CRITICAL"
        assert sample["language"] == "c"
        assert sample["content_hash"]  # non-empty


# ---------------------------------------------------------------------------
# Mock adapter for pipeline tests
# ---------------------------------------------------------------------------
class MockAdapter:
    """A mock vulnerability source adapter for testing pipeline logic."""

    def __init__(self, advisories: list[dict[str, Any]]) -> None:
        self._advisories = advisories

    def fetch_advisories(self, limit: int = 0) -> Generator[dict[str, Any], None, None]:
        for i, adv in enumerate(self._advisories):
            if limit > 0 and i >= limit:
                return
            yield adv

    def resolve_commit_pair(
        self, advisory: dict[str, Any]
    ) -> Optional[tuple[CommitRef, CommitRef]]:
        repo = advisory.get("repo_url", "")
        fix = advisory.get("fix_hash", "")
        if not repo or not fix:
            return None
        return (
            CommitRef(repo_url=repo, commit_sha=f"{fix}~1"),
            CommitRef(repo_url=repo, commit_sha=fix),
        )


# ---------------------------------------------------------------------------
# Quality filter tests
# ---------------------------------------------------------------------------
class TestQualityFilters:
    """Test the quality filter logic on CommitDeltaPipeline."""

    def _make_pipeline(self, **kwargs: Any) -> CommitDeltaPipeline:
        config = DeltaExtractionConfig(
            output_dir=Path("/tmp/test"),
            **kwargs,
        )
        return CommitDeltaPipeline(config)

    def test_rejects_too_many_files(self) -> None:
        pipeline = self._make_pipeline(max_files_per_commit=5)
        delta = CommitDelta(
            delta_id="test",
            files_changed=10,
            diff_token_count=100,
            cve_id="CVE-2024-1234",
        )
        assert not pipeline._passes_quality_filters(delta)
        assert pipeline.stats["failed_too_many_files"] == 1

    def test_rejects_too_long_diff(self) -> None:
        pipeline = self._make_pipeline(max_diff_tokens=1000)
        delta = CommitDelta(
            delta_id="test",
            files_changed=1,
            diff_token_count=5000,
            cve_id="CVE-2024-1234",
        )
        assert not pipeline._passes_quality_filters(delta)
        assert pipeline.stats["failed_too_long"] == 1

    def test_rejects_missing_cve_when_required(self) -> None:
        pipeline = self._make_pipeline(require_cve_link=True)
        delta = CommitDelta(
            delta_id="test",
            files_changed=1,
            diff_token_count=100,
            cve_id="",
        )
        assert not pipeline._passes_quality_filters(delta)
        assert pipeline.stats["failed_no_cve"] == 1

    def test_passes_valid_delta(self) -> None:
        pipeline = self._make_pipeline()
        delta = CommitDelta(
            delta_id="test",
            files_changed=3,
            diff_token_count=500,
            cve_id="CVE-2024-1234",
        )
        assert pipeline._passes_quality_filters(delta)

    def test_allows_missing_cve_when_not_required(self) -> None:
        pipeline = self._make_pipeline(require_cve_link=False)
        delta = CommitDelta(
            delta_id="test",
            files_changed=1,
            diff_token_count=100,
            cve_id="",
        )
        assert pipeline._passes_quality_filters(delta)


# ---------------------------------------------------------------------------
# State machine tests
# ---------------------------------------------------------------------------
class TestStateMachine:
    """Test the deterministic state machine orchestrator."""

    def test_valid_transitions(self) -> None:
        from sentinel.agent.state_machine import (
            Orchestrator,
            State,
            Transition,
        )

        orchestrator = Orchestrator()
        session = orchestrator.create_session(
            target_repo="https://github.com/test/repo",
        )

        assert session.state == State.INTAKE

        # INTAKE → ANALYSIS
        orchestrator.step(session, Transition.TARGET_RECEIVED)
        assert session.state == State.ANALYSIS

        # ANALYSIS → ARCHITECT_REASONING
        orchestrator.step(session, Transition.METADATA_LOADED)
        assert session.state == State.ARCHITECT_REASONING

        # ARCHITECT_REASONING → TOOL_EXPLORATION
        orchestrator.step(session, Transition.TOOL_CALL_ISSUED)
        assert session.state == State.TOOL_EXPLORATION

        # TOOL_EXPLORATION → ARCHITECT_REASONING (loop)
        orchestrator.step(session, Transition.TOOL_RESULT_RETURNED)
        assert session.state == State.ARCHITECT_REASONING

    def test_invalid_transition_raises(self) -> None:
        from sentinel.agent.state_machine import (
            Orchestrator,
            State,
            Transition,
        )

        orchestrator = Orchestrator()
        session = orchestrator.create_session(target_repo="test")

        # Cannot go from INTAKE directly to TOOL_EXPLORATION
        with pytest.raises(ValueError, match="Invalid transition"):
            orchestrator.step(session, Transition.TOOL_CALL_ISSUED)

    def test_terminal_states(self) -> None:
        from sentinel.agent.state_machine import State, StateMachine

        sm = StateMachine()
        assert sm.is_terminal(State.COMPLETED)
        assert sm.is_terminal(State.FAILED)
        assert not sm.is_terminal(State.INTAKE)
        assert not sm.is_terminal(State.ARCHITECT_REASONING)

    def test_audit_log_generated(self) -> None:
        from sentinel.agent.state_machine import Orchestrator, Transition

        orchestrator = Orchestrator()
        session = orchestrator.create_session(target_repo="test/repo")
        orchestrator.step(session, Transition.TARGET_RECEIVED)

        log = orchestrator.get_audit_log(session.session_id)
        assert log is not None
        assert log["target_repo"] == "test/repo"
        assert log["final_state"] == "ANALYSIS"
        assert log["tool_calls_made"] == 0


# ---------------------------------------------------------------------------
# Training config tests
# ---------------------------------------------------------------------------
class TestTrainingConfig:
    """Test NVFP4 training configuration generation."""

    def test_deepspeed_config_format(self) -> None:
        from sentinel.config import TrainingConfig
        from sentinel.training.nvfp4_config import build_deepspeed_config

        config = TrainingConfig()
        ds = build_deepspeed_config(config)

        assert ds["zero_optimization"]["stage"] == 2
        assert ds["bf16"]["enabled"] is True
        assert ds["gradient_clipping"] == 1.0
        assert "train_batch_size" in ds

    def test_vram_budget_positive(self) -> None:
        from sentinel.config import TrainingConfig
        from sentinel.training.nvfp4_config import estimate_vram_budget

        config = TrainingConfig()
        budget = estimate_vram_budget(config)

        assert budget["activation_budget_gb"] > 0
        assert budget["model_weights_gb"] > 0
        assert budget["gpu_memory_gb"] == 192

    def test_context_parallel_disabled_for_8k(self) -> None:
        from sentinel.training.nvfp4_config import _build_context_parallel_config

        cp = _build_context_parallel_config(8192, 8)
        assert cp is None

    def test_context_parallel_enabled_for_32k(self) -> None:
        from sentinel.training.nvfp4_config import _build_context_parallel_config

        cp = _build_context_parallel_config(32768, 8)
        assert cp is not None
        assert cp["enabled"] is True
        assert cp["cp_degree"] == 2
        assert cp["dp_degree"] == 4

    def test_context_parallel_enabled_for_128k(self) -> None:
        from sentinel.training.nvfp4_config import _build_context_parallel_config

        cp = _build_context_parallel_config(131072, 8)
        assert cp is not None
        assert cp["cp_degree"] in (4, 8)


# ---------------------------------------------------------------------------
# Evaluation tests
# ---------------------------------------------------------------------------
class TestEvaluation:
    """Test statistical analysis utilities."""

    def test_mean(self) -> None:
        from sentinel.evaluation import StatisticalAnalysis

        assert StatisticalAnalysis.mean([1.0, 2.0, 3.0]) == 2.0
        assert StatisticalAnalysis.mean([]) == 0.0

    def test_std_dev(self) -> None:
        from sentinel.evaluation import StatisticalAnalysis

        std = StatisticalAnalysis.std_dev([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
        assert 1.5 < std < 2.5  # Known: ~2.0

    def test_confidence_interval(self) -> None:
        from sentinel.evaluation import StatisticalAnalysis

        low, high = StatisticalAnalysis.confidence_interval_95([70.0, 72.0, 74.0, 76.0, 78.0])
        assert low < 74.0 < high
        assert low > 60.0  # Sanity
        assert high < 90.0  # Sanity

    def test_bonferroni_correction(self) -> None:
        from sentinel.evaluation import StatisticalAnalysis

        corrected = StatisticalAnalysis.bonferroni_correction([0.01, 0.03, 0.05])
        assert corrected[0] == pytest.approx(0.03)   # 0.01 * 3
        assert corrected[1] == pytest.approx(0.09)   # 0.03 * 3
        assert corrected[2] == pytest.approx(0.15)   # 0.05 * 3

    def test_deployment_readiness_check(self) -> None:
        from sentinel.evaluation import check_deployment_readiness

        # All passing
        results = {
            "cyberseceval3": {"passed": True},
            "cvefixes": {"passed": True},
            "swebench": {"passed": True},
            "purplellama": {"passed": True},
            "redteam": {"passed": True},
        }
        readiness = check_deployment_readiness(results)
        assert readiness["deployment_ready"] is True

        # One failing
        results["swebench"]["passed"] = False
        readiness = check_deployment_readiness(results)
        assert readiness["deployment_ready"] is False


# ---------------------------------------------------------------------------
# CWE mapper tests
# ---------------------------------------------------------------------------
class TestCWEMapper:
    """Test CWE taxonomy and inference."""

    def test_known_cwe_lookup(self) -> None:
        from sentinel.data.cwe_mapper import CWEMapper

        mapper = CWEMapper()
        entry = mapper.lookup("CWE-787")
        assert entry is not None
        assert "Out-of-bounds Write" in entry.name

    def test_unknown_cwe_returns_none(self) -> None:
        from sentinel.data.cwe_mapper import CWEMapper

        mapper = CWEMapper()
        assert mapper.lookup("CWE-99999") is None

    def test_infer_from_diff_detects_sql_injection(self) -> None:
        from sentinel.data.cwe_mapper import CWEMapper

        mapper = CWEMapper()
        diff = '-cursor.execute(f"SELECT * FROM users WHERE id={uid}")\n+cursor.execute("SELECT * FROM users WHERE id=%s", (uid,))'
        inferred = mapper.infer_from_code(diff)
        assert any("CWE-89" in cwe for cwe in inferred)


# ---------------------------------------------------------------------------
# Integrity tests
# ---------------------------------------------------------------------------
class TestIntegrity:
    """Test dataset integrity management."""

    def test_merkle_root_deterministic(self) -> None:
        from sentinel.data.integrity import DatasetIntegrityManager, ProvenanceRecord

        mgr = DatasetIntegrityManager(output_dir=Path("/tmp/test_integrity"))

        record1 = ProvenanceRecord(
            sample_id="test-1",
            content_hash="abc123",
            source_commit_sha="deadbeef",
            cve_id="CVE-2024-001",
            source_database="cvefixes",
        )
        record2 = ProvenanceRecord(
            sample_id="test-2",
            content_hash="def456",
            source_commit_sha="cafebabe",
            cve_id="CVE-2024-002",
            source_database="osv",
        )

        mgr.register_sample(record1)
        mgr.register_sample(record2)
        root1 = mgr.finalize()

        # Same data → same root
        mgr2 = DatasetIntegrityManager(output_dir=Path("/tmp/test_integrity2"))
        mgr2.register_sample(record1)
        mgr2.register_sample(record2)
        root2 = mgr2.finalize()

        assert root1 == root2

    def test_duplicate_rejection(self) -> None:
        from sentinel.data.integrity import DatasetIntegrityManager, ProvenanceRecord

        mgr = DatasetIntegrityManager(output_dir=Path("/tmp/test_dup"))
        record = ProvenanceRecord(
            sample_id="test-1",
            content_hash="abc123",
            source_commit_sha="deadbeef",
            cve_id="CVE-2024-001",
            source_database="cvefixes",
        )

        assert mgr.register_sample(record) is True
        assert mgr.register_sample(record) is False  # Duplicate
