"""
Automated DPO Preference-Pair Synthesis Pipeline.

Implements Algorithm 1 from the doctoral research proposal:
the closed-loop pipeline that generates (chosen, rejected) pairs
by compiling and fuzzing model-generated patches in Docker sandboxes.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generator, Optional

from sentinel.alignment.sandbox import (
    DockerSandbox,
    ExecutionResult,
    FuzzerHarness,
    SandboxResult,
)
from sentinel.config import DPOSynthesisConfig

logger = logging.getLogger(__name__)


@dataclass
class VulnerableSnippet:
    """Input to the DPO synthesis pipeline."""
    snippet_id: str
    vulnerable_code: str
    language: str
    cwe_id: str = ""
    test_code: str = ""
    description: str = ""


@dataclass
class PreferencePair:
    """
    A single (chosen, rejected) pair for DPO training.

    Corresponds to the output of Algorithm 1:
    {prompt: <vuln_code>, chosen: <working_secure_patch>, rejected: <failed_attempt>}
    """
    pair_id: str
    prompt: str                   # The vulnerable code snippet
    chosen: str = ""              # Patch that compiles, passes tests, passes fuzzing
    rejected: str = ""            # Patch that fails at some stage
    chosen_reason: str = ""       # Why chosen is valid
    rejected_reason: str = ""     # Why rejected failed
    language: str = ""
    cwe_id: str = ""
    snippet_id: str = ""

    # Validation metadata
    chosen_compiles: bool = False
    chosen_tests_pass: bool = False
    chosen_fuzz_clean: bool = False
    rejected_compiles: bool = False
    rejected_tests_pass: bool = False

    def to_training_format(self) -> dict[str, str]:
        """Convert to the format expected by trl.DPOTrainer."""
        return {
            "prompt": self.prompt,
            "chosen": self.chosen,
            "rejected": self.rejected,
        }

    def to_dict(self) -> dict[str, Any]:
        """Full serialization including metadata."""
        return {
            "pair_id": self.pair_id,
            "prompt": self.prompt,
            "chosen": self.chosen,
            "rejected": self.rejected,
            "chosen_reason": self.chosen_reason,
            "rejected_reason": self.rejected_reason,
            "language": self.language,
            "cwe_id": self.cwe_id,
            "snippet_id": self.snippet_id,
            "chosen_compiles": self.chosen_compiles,
            "chosen_tests_pass": self.chosen_tests_pass,
            "chosen_fuzz_clean": self.chosen_fuzz_clean,
            "rejected_compiles": self.rejected_compiles,
            "rejected_tests_pass": self.rejected_tests_pass,
        }


class ModelInterface:
    """
    Interface for generating patch candidates from the SFT-trained model.

    In production, this wraps the HuggingFace model inference.
    """

    def __init__(
        self,
        model_path: str,
        temperature: float = 0.7,
        max_new_tokens: int = 2048,
        num_return_sequences: int = 1,
    ) -> None:
        self.model_path = model_path
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.num_return_sequences = num_return_sequences
        self._model: Any = None
        self._tokenizer: Any = None

    def load(self) -> None:
        """Load the model and tokenizer."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype="auto",
                device_map="auto",
            )
            logger.info("Loaded model from %s", self.model_path)
        except ImportError:
            logger.warning("transformers not installed. Using mock model.")

    def generate_patch(self, vulnerable_code: str, language: str, cwe_id: str = "") -> str:
        """
        Generate a patch candidate for the given vulnerable code.

        Args:
            vulnerable_code: The vulnerable source code
            language: Programming language
            cwe_id: Optional CWE identifier for context

        Returns:
            Generated patch code
        """
        prompt = self._build_prompt(vulnerable_code, language, cwe_id)

        if self._model is not None and self._tokenizer is not None:
            inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=True,
            )
            response = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extract code from response
            return self._extract_code(response)
        else:
            # Mock response for testing
            return f"// Mock patch for {language} ({cwe_id})\n{vulnerable_code}"

    def _build_prompt(self, vulnerable_code: str, language: str, cwe_id: str = "") -> str:
        """Build the prompt for patch generation."""
        cwe_context = f" The vulnerability is classified as {cwe_id}." if cwe_id else ""
        return (
            f"You are a security engineer. The following {language} code contains "
            f"a vulnerability.{cwe_context}\n\n"
            f"Vulnerable code:\n```{language}\n{vulnerable_code}\n```\n\n"
            f"Write a secure patch that fixes the vulnerability while preserving "
            f"the original functionality. Output only the patched code:\n"
            f"```{language}\n"
        )

    @staticmethod
    def _extract_code(response: str) -> str:
        """Extract code block from model response."""
        # Find code between ``` markers
        parts = response.split("```")
        if len(parts) >= 3:
            code_block = parts[-2]
            # Remove language tag from first line
            lines = code_block.strip().split("\n")
            if lines and not lines[0].strip().startswith(("{", "#", "/", "import", "from", "def", "class")):
                lines = lines[1:]
            return "\n".join(lines)
        return response


@dataclass
class PipelineStats:
    """Statistics for the DPO synthesis pipeline."""
    total_snippets: int = 0
    total_attempts: int = 0
    compile_failures: int = 0
    test_failures: int = 0
    fuzz_failures: int = 0
    valid_chosen: int = 0
    valid_pairs: int = 0
    elapsed_seconds: float = 0.0

    @property
    def chosen_rate(self) -> float:
        return self.valid_chosen / max(self.total_attempts, 1)

    @property
    def pair_rate(self) -> float:
        return self.valid_pairs / max(self.total_snippets, 1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_snippets": self.total_snippets,
            "total_attempts": self.total_attempts,
            "compile_failures": self.compile_failures,
            "test_failures": self.test_failures,
            "fuzz_failures": self.fuzz_failures,
            "valid_chosen": self.valid_chosen,
            "valid_pairs": self.valid_pairs,
            "chosen_rate": round(self.chosen_rate, 4),
            "pair_rate": round(self.pair_rate, 4),
            "elapsed_seconds": round(self.elapsed_seconds, 1),
        }


class DPOSynthesisPipeline:
    """
    Automated DPO preference-pair synthesis pipeline.

    Implements Algorithm 1 from the doctoral research proposal.
    For each vulnerable code snippet:
    1. Generate N patch candidates from the SFT model
    2. Compile each candidate in Docker sandbox
    3. Run tests against compiled code
    4. Run differential fuzzing (original vs. patched)
    5. Label: compile+test+fuzz pass → Chosen, any failure → Rejected
    6. Pair one chosen with one rejected for DPO training
    """

    def __init__(
        self,
        config: DPOSynthesisConfig,
        model: ModelInterface,
        sandbox: Optional[DockerSandbox] = None,
    ) -> None:
        self.config = config
        self.model = model
        self.sandbox = sandbox or DockerSandbox(
            image=config.sandbox_image,
            timeout_seconds=config.sandbox_timeout_seconds,
        )
        self.fuzzer = FuzzerHarness(
            sandbox=self.sandbox,
            fuzzer=config.fuzz_tool,
            duration_seconds=config.fuzz_duration_seconds,
        )
        self.stats = PipelineStats()

    def run(
        self,
        snippets: list[VulnerableSnippet],
    ) -> Generator[PreferencePair, None, None]:
        """
        Execute the full synthesis pipeline over a list of vulnerable snippets.

        Yields validated PreferencePairs.
        """
        start_time = time.time()

        for snippet in snippets:
            self.stats.total_snippets += 1
            pair = self._process_snippet(snippet)
            if pair is not None:
                self.stats.valid_pairs += 1
                yield pair

            if self.stats.valid_pairs >= self.config.target_pairs:
                logger.info("Reached target of %d pairs", self.config.target_pairs)
                break

        self.stats.elapsed_seconds = time.time() - start_time
        logger.info("Pipeline complete. Stats: %s", json.dumps(self.stats.to_dict(), indent=2))

    def _process_snippet(self, snippet: VulnerableSnippet) -> Optional[PreferencePair]:
        """
        Process a single snippet: generate candidates, validate, pair.

        Returns a PreferencePair if both chosen and rejected are found.
        """
        chosen_candidate: Optional[str] = None
        chosen_reason: str = ""
        rejected_candidate: Optional[str] = None
        rejected_reason: str = ""

        for attempt in range(self.config.max_retries_per_prompt):
            self.stats.total_attempts += 1

            # Step 1: Generate patch candidate (Algorithm 1, line 1)
            candidate = self.model.generate_patch(
                snippet.vulnerable_code,
                snippet.language,
                snippet.cwe_id,
            )

            # Step 2: Compile (Algorithm 1, lines 2-4)
            compile_result = self.sandbox.compile_code(candidate, snippet.language)
            if not compile_result.passed:
                self.stats.compile_failures += 1
                if rejected_candidate is None:
                    rejected_candidate = candidate
                    rejected_reason = f"Compilation failed: {compile_result.stderr[:200]}"
                continue

            # Step 3: Run tests (Algorithm 1, lines 5-7)
            if snippet.test_code:
                test_result = self.sandbox.run_tests(
                    candidate, snippet.test_code, snippet.language
                )
                if not test_result.passed:
                    self.stats.test_failures += 1
                    if rejected_candidate is None:
                        rejected_candidate = candidate
                        rejected_reason = f"Tests failed: {test_result.stderr[:200]}"
                    continue

            # Step 4: Differential fuzzing (Algorithm 1, lines 9-13)
            if self.config.differential_fuzzing:
                is_valid, fuzz_explanation = self.fuzzer.differential_fuzz(
                    snippet.vulnerable_code,
                    candidate,
                    snippet.language,
                )
                if not is_valid:
                    self.stats.fuzz_failures += 1
                    if rejected_candidate is None:
                        rejected_candidate = candidate
                        rejected_reason = f"Fuzzing failed: {fuzz_explanation}"
                    continue

            # All checks passed — this is a CHOSEN candidate
            self.stats.valid_chosen += 1
            chosen_candidate = candidate
            chosen_reason = "Compiles + tests pass + fuzz clean"
            break

        # Build preference pair if we have both chosen and rejected
        if chosen_candidate is not None and rejected_candidate is not None:
            pair_id = hashlib.sha256(
                f"{snippet.snippet_id}:{chosen_candidate[:100]}".encode()
            ).hexdigest()[:12]

            return PreferencePair(
                pair_id=pair_id,
                prompt=snippet.vulnerable_code,
                chosen=chosen_candidate,
                rejected=rejected_candidate,
                chosen_reason=chosen_reason,
                rejected_reason=rejected_reason,
                language=snippet.language,
                cwe_id=snippet.cwe_id,
                snippet_id=snippet.snippet_id,
                chosen_compiles=True,
                chosen_tests_pass=True,
                chosen_fuzz_clean=True,
            )

        return None

    def save_pairs(
        self,
        pairs: list[PreferencePair],
        output_path: Optional[Path] = None,
    ) -> Path:
        """Save generated preference pairs to disk in JSONL format."""
        path = output_path or self.config.output_dir / "preference_pairs.jsonl"
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            for pair in pairs:
                f.write(json.dumps(pair.to_dict()) + "\n")

        # Also save in trl-compatible format
        trl_path = path.with_suffix(".trl.jsonl")
        with open(trl_path, "w") as f:
            for pair in pairs:
                f.write(json.dumps(pair.to_training_format()) + "\n")

        logger.info("Saved %d pairs to %s (+ trl format: %s)", len(pairs), path, trl_path)
        return path
