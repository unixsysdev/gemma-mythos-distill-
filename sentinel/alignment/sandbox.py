"""
Docker Sandbox for Compilation and Execution.

Provides isolated environments for compiling and testing
model-generated patches during DPO synthesis. All code
execution is containerized to prevent host system compromise.
"""

from __future__ import annotations

import hashlib
import json
import logging
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class SandboxResult(Enum):
    """Result of a sandbox execution."""
    SUCCESS = "success"
    COMPILE_FAIL = "compile_fail"
    TEST_FAIL = "test_fail"
    FUZZ_CRASH = "fuzz_crash"
    TIMEOUT = "timeout"
    ERROR = "error"


@dataclass
class ExecutionResult:
    """Complete result from a sandbox execution."""
    result: SandboxResult
    exit_code: int = -1
    stdout: str = ""
    stderr: str = ""
    duration_seconds: float = 0.0
    memory_peak_mb: float = 0.0
    fuzz_crashes: int = 0
    fuzz_inputs_tested: int = 0

    @property
    def passed(self) -> bool:
        return self.result == SandboxResult.SUCCESS


class DockerSandbox:
    """
    Manages Docker containers for secure code compilation and testing.

    Used by the DPO synthesis pipeline (Algorithm 1 in the proposal)
    to validate model-generated patches.
    """

    # Language → compilation commands
    COMPILE_COMMANDS: dict[str, list[str]] = {
        "c": ["gcc", "-Wall", "-Werror", "-o", "/workspace/output", "/workspace/code.c"],
        "cpp": ["g++", "-Wall", "-Werror", "-o", "/workspace/output", "/workspace/code.cpp"],
        "python": ["python3", "-m", "py_compile", "/workspace/code.py"],
        "rust": ["rustc", "-o", "/workspace/output", "/workspace/code.rs"],
        "go": ["go", "build", "-o", "/workspace/output", "/workspace/code.go"],
        "java": ["javac", "/workspace/Code.java"],
    }

    # Language → file extension
    FILE_EXTENSIONS: dict[str, str] = {
        "c": ".c",
        "cpp": ".cpp",
        "python": ".py",
        "rust": ".rs",
        "go": ".go",
        "java": ".java",
    }

    # Language → test execution
    TEST_COMMANDS: dict[str, list[str]] = {
        "c": ["/workspace/output"],
        "cpp": ["/workspace/output"],
        "python": ["python3", "/workspace/code.py"],
        "rust": ["/workspace/output"],
        "go": ["/workspace/output"],
        "java": ["java", "-cp", "/workspace", "Code"],
    }

    def __init__(
        self,
        image: str = "sentinel-sandbox:latest",
        timeout_seconds: int = 120,
        memory_limit: str = "512m",
        network_disabled: bool = True,
    ) -> None:
        self.image = image
        self.timeout_seconds = timeout_seconds
        self.memory_limit = memory_limit
        self.network_disabled = network_disabled
        self._client: Any = None

    def _get_client(self) -> Any:
        """Lazily initialize Docker client."""
        if self._client is None:
            import docker
            self._client = docker.from_env()
        return self._client

    def compile_code(
        self,
        code: str,
        language: str,
        test_code: Optional[str] = None,
    ) -> ExecutionResult:
        """
        Compile code in an isolated Docker container.

        Args:
            code: Source code to compile
            language: Programming language
            test_code: Optional test code to run after compilation

        Returns:
            ExecutionResult with compilation status
        """
        if language not in self.COMPILE_COMMANDS:
            return ExecutionResult(
                result=SandboxResult.ERROR,
                stderr=f"Unsupported language: {language}",
            )

        ext = self.FILE_EXTENSIONS[language]
        compile_cmd = self.COMPILE_COMMANDS[language]

        try:
            client = self._get_client()

            # Create a temporary directory with the code
            with tempfile.TemporaryDirectory() as tmpdir:
                code_path = Path(tmpdir) / f"code{ext}"
                code_path.write_text(code)

                if test_code:
                    test_path = Path(tmpdir) / f"test{ext}"
                    test_path.write_text(test_code)

                # Run compilation in container
                container = client.containers.run(
                    self.image,
                    command=compile_cmd,
                    volumes={tmpdir: {"bind": "/workspace", "mode": "rw"}},
                    mem_limit=self.memory_limit,
                    network_disabled=self.network_disabled,
                    remove=True,
                    detach=False,
                    stdout=True,
                    stderr=True,
                    timeout=self.timeout_seconds,
                )

                stdout = container.decode("utf-8") if isinstance(container, bytes) else ""

                return ExecutionResult(
                    result=SandboxResult.SUCCESS,
                    exit_code=0,
                    stdout=stdout,
                )

        except Exception as e:
            error_msg = str(e)
            if "non-zero exit" in error_msg.lower():
                return ExecutionResult(
                    result=SandboxResult.COMPILE_FAIL,
                    exit_code=1,
                    stderr=error_msg,
                )
            elif "timeout" in error_msg.lower():
                return ExecutionResult(
                    result=SandboxResult.TIMEOUT,
                    stderr=error_msg,
                )
            else:
                logger.error("Sandbox execution error: %s", e)
                return ExecutionResult(
                    result=SandboxResult.ERROR,
                    stderr=error_msg,
                )

    def run_tests(
        self,
        code: str,
        test_code: str,
        language: str,
    ) -> ExecutionResult:
        """
        Run tests against compiled code in sandbox.

        Args:
            code: Source code
            test_code: Test code to execute
            language: Programming language

        Returns:
            ExecutionResult with test status
        """
        # First compile
        compile_result = self.compile_code(code, language)
        if not compile_result.passed:
            return compile_result

        if language not in self.TEST_COMMANDS:
            return ExecutionResult(
                result=SandboxResult.ERROR,
                stderr=f"No test runner for language: {language}",
            )

        try:
            client = self._get_client()
            ext = self.FILE_EXTENSIONS[language]

            with tempfile.TemporaryDirectory() as tmpdir:
                code_path = Path(tmpdir) / f"code{ext}"
                code_path.write_text(code)

                test_path = Path(tmpdir) / f"test{ext}"
                test_path.write_text(test_code)

                # Run test
                test_cmd = self.TEST_COMMANDS[language]
                container = client.containers.run(
                    self.image,
                    command=test_cmd,
                    volumes={tmpdir: {"bind": "/workspace", "mode": "rw"}},
                    mem_limit=self.memory_limit,
                    network_disabled=self.network_disabled,
                    remove=True,
                    detach=False,
                    stdout=True,
                    stderr=True,
                    timeout=self.timeout_seconds,
                )

                stdout = container.decode("utf-8") if isinstance(container, bytes) else ""
                return ExecutionResult(
                    result=SandboxResult.SUCCESS,
                    exit_code=0,
                    stdout=stdout,
                )

        except Exception as e:
            return ExecutionResult(
                result=SandboxResult.TEST_FAIL,
                exit_code=1,
                stderr=str(e),
            )


class FuzzerHarness:
    """
    Differential fuzzing harness for DPO preference-pair validation.

    Implements lines 9-13 of Algorithm 1 from the proposal:
    - Fuzz the original vulnerable code → must crash
    - Fuzz the patched code → must NOT crash
    """

    FUZZER_COMMANDS: dict[str, list[str]] = {
        "afl++": ["afl-fuzz", "-i", "/workspace/seeds", "-o", "/workspace/findings",
                   "-t", "1000", "--", "/workspace/target"],
        "libfuzzer": ["/workspace/target", "-max_total_time={duration}",
                      "-artifact_prefix=/workspace/findings/"],
    }

    def __init__(
        self,
        sandbox: DockerSandbox,
        fuzzer: str = "afl++",
        duration_seconds: int = 60,
    ) -> None:
        self.sandbox = sandbox
        self.fuzzer = fuzzer
        self.duration_seconds = duration_seconds

    def fuzz(
        self,
        code: str,
        language: str,
        seed_inputs: Optional[list[bytes]] = None,
    ) -> ExecutionResult:
        """
        Fuzz compiled code for the configured duration.

        Args:
            code: Source code to fuzz
            language: Programming language
            seed_inputs: Initial fuzzer seed corpus

        Returns:
            ExecutionResult with fuzzing results (FUZZ_CRASH if crashes found)
        """
        # First compile the target
        compile_result = self.sandbox.compile_code(code, language)
        if not compile_result.passed:
            return ExecutionResult(
                result=SandboxResult.ERROR,
                stderr=f"Cannot fuzz: compilation failed. {compile_result.stderr}",
            )

        try:
            client = self.sandbox._get_client()

            with tempfile.TemporaryDirectory() as tmpdir:
                # Write code and seeds
                ext = self.sandbox.FILE_EXTENSIONS.get(language, ".c")
                code_path = Path(tmpdir) / f"code{ext}"
                code_path.write_text(code)

                seeds_dir = Path(tmpdir) / "seeds"
                seeds_dir.mkdir()
                if seed_inputs:
                    for i, seed in enumerate(seed_inputs):
                        (seeds_dir / f"seed_{i}").write_bytes(seed)
                else:
                    (seeds_dir / "default").write_bytes(b"AAAA")

                findings_dir = Path(tmpdir) / "findings"
                findings_dir.mkdir()

                # Run fuzzer
                fuzz_cmd = self.FUZZER_COMMANDS.get(self.fuzzer, [])
                if not fuzz_cmd:
                    return ExecutionResult(
                        result=SandboxResult.ERROR,
                        stderr=f"Unknown fuzzer: {self.fuzzer}",
                    )

                container = client.containers.run(
                    self.sandbox.image,
                    command=fuzz_cmd,
                    volumes={tmpdir: {"bind": "/workspace", "mode": "rw"}},
                    mem_limit="1g",
                    network_disabled=True,
                    remove=True,
                    detach=False,
                    stdout=True,
                    stderr=True,
                    timeout=self.duration_seconds + 30,
                )

                # Check for crashes
                crashes = list(findings_dir.glob("crashes/*")) if findings_dir.exists() else []
                crash_count = len(crashes)

                if crash_count > 0:
                    return ExecutionResult(
                        result=SandboxResult.FUZZ_CRASH,
                        exit_code=0,
                        fuzz_crashes=crash_count,
                        stdout=f"Found {crash_count} crashes",
                    )
                else:
                    return ExecutionResult(
                        result=SandboxResult.SUCCESS,
                        exit_code=0,
                        fuzz_crashes=0,
                        stdout="No crashes found",
                    )

        except Exception as e:
            logger.error("Fuzzing error: %s", e)
            return ExecutionResult(
                result=SandboxResult.ERROR,
                stderr=str(e),
            )

    def differential_fuzz(
        self,
        vulnerable_code: str,
        patched_code: str,
        language: str,
        seed_inputs: Optional[list[bytes]] = None,
    ) -> tuple[bool, str]:
        """
        Perform differential fuzzing as specified in the proposal.

        A valid "chosen" sample must satisfy:
        - crashes_on_original == True (vulnerability is real)
        - crashes_on_patch == False (patch actually fixes it)

        Args:
            vulnerable_code: Original vulnerable source
            patched_code: Model-generated patch
            language: Programming language
            seed_inputs: Initial seed corpus

        Returns:
            (is_valid_chosen, explanation)
        """
        # Fuzz the original
        original_result = self.fuzz(vulnerable_code, language, seed_inputs)

        # Fuzz the patch
        patch_result = self.fuzz(patched_code, language, seed_inputs)

        crashes_on_original = original_result.result == SandboxResult.FUZZ_CRASH
        crashes_on_patch = patch_result.result == SandboxResult.FUZZ_CRASH

        if crashes_on_original and not crashes_on_patch:
            return True, (
                f"VALID: Original crashed ({original_result.fuzz_crashes} crashes), "
                f"patch is clean."
            )
        elif not crashes_on_original:
            return False, (
                "INVALID: Original code did not crash under fuzzing. "
                "Vulnerability may not be triggerable via fuzzing."
            )
        elif crashes_on_patch:
            return False, (
                f"INVALID: Patch still crashes ({patch_result.fuzz_crashes} crashes). "
                f"Vulnerability not fixed."
            )
        else:
            return False, "INVALID: Unexpected fuzzing state."
