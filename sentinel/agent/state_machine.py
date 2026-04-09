"""
Deterministic State Machine Orchestrator.

Implements the dual-model architecture from Section 5.2 of the specification.
Gemma 4 31B (Architect) handles strategic reasoning; a fast code-centric
model (Executor) handles patch generation. They are governed by a
deterministic state machine — no free-form inter-model communication.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State definitions
# ---------------------------------------------------------------------------
class State(Enum):
    """States in the SENTINEL orchestration state machine."""
    INTAKE = auto()
    ANALYSIS = auto()
    ARCHITECT_REASONING = auto()
    TOOL_EXPLORATION = auto()
    VULNERABILITY_REPORT = auto()
    PATCH_GENERATION = auto()
    EXECUTOR_CODEGEN = auto()
    SANDBOX_VALIDATION = auto()
    ERROR_FEEDBACK = auto()
    HUMAN_REVIEW = auto()
    AUTO_REMEDIATE = auto()
    COMPLETED = auto()
    FAILED = auto()


class Transition(Enum):
    """Events that trigger state transitions."""
    TARGET_RECEIVED = auto()
    METADATA_LOADED = auto()
    TOOL_CALL_ISSUED = auto()
    TOOL_RESULT_RETURNED = auto()
    VULNERABILITY_CONFIRMED = auto()
    NO_VULNERABILITY_FOUND = auto()
    FIX_SPEC_GENERATED = auto()
    PATCH_GENERATED = auto()
    VALIDATION_PASSED = auto()
    VALIDATION_FAILED = auto()
    HUMAN_APPROVED = auto()
    HUMAN_REJECTED = auto()
    PATCH_DEPLOYED = auto()
    MAX_RETRIES_EXCEEDED = auto()
    TIMEOUT = auto()


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------
@dataclass
class ToolCall:
    """A structured tool invocation from the Architect model."""
    tool_name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    call_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])


@dataclass
class ToolResult:
    """Result returned from a tool execution."""
    call_id: str
    tool_name: str
    output: str = ""
    error: str = ""
    success: bool = True
    tokens_used: int = 0


@dataclass
class VulnerabilityFinding:
    """Structured vulnerability finding from the Architect."""
    finding_id: str = ""
    cwe_id: str = ""
    cwe_name: str = ""
    severity: str = ""
    cvss_vector: str = ""
    location: str = ""
    vulnerable_code: str = ""
    secure_alternative: str = ""
    explanation: str = ""
    confidence: float = 0.0
    attack_vector: str = ""

    def to_json_schema(self) -> dict[str, Any]:
        """Format as the JSON schema passed to the Executor via API Bridge."""
        return {
            "finding_id": self.finding_id,
            "cwe": self.cwe_id,
            "severity": self.severity,
            "location": self.location,
            "vulnerable_code": self.vulnerable_code,
            "fix_strategy": self.explanation,
            "target_language": self._detect_language(),
        }

    def _detect_language(self) -> str:
        """Infer language from file extension in location."""
        ext_map = {
            ".py": "python", ".js": "javascript", ".ts": "typescript",
            ".c": "c", ".cpp": "cpp", ".go": "go", ".rs": "rust",
            ".java": "java", ".rb": "ruby", ".php": "php",
        }
        for ext, lang in ext_map.items():
            if self.location.endswith(ext) or f"{ext}:" in self.location:
                return lang
        return "unknown"


@dataclass
class SessionContext:
    """
    The mutable state of an active SENTINEL session.

    Contains the agent's working memory, accumulated findings,
    and execution trace for audit logging.
    """
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    state: State = State.INTAKE
    target_repo: str = ""
    target_branch: str = "main"

    # Working memory
    current_hypothesis: str = ""
    retrieved_fragments: list[str] = field(default_factory=list)
    tool_history: list[tuple[ToolCall, ToolResult]] = field(default_factory=list)
    findings: list[VulnerabilityFinding] = field(default_factory=list)

    # Counters
    tool_calls_made: int = 0
    executor_retries: int = 0
    total_tokens_consumed: int = 0

    # Timing
    start_time: float = field(default_factory=time.time)
    state_timestamps: dict[str, float] = field(default_factory=dict)

    # Configuration limits
    max_tool_calls: int = 200
    max_executor_retries: int = 3
    session_timeout_minutes: int = 30

    @property
    def elapsed_minutes(self) -> float:
        return (time.time() - self.start_time) / 60.0

    @property
    def is_timed_out(self) -> bool:
        return self.elapsed_minutes >= self.session_timeout_minutes

    @property
    def tool_budget_remaining(self) -> int:
        return self.max_tool_calls - self.tool_calls_made

    def record_state_change(self, new_state: State) -> None:
        """Record a state transition with timestamp."""
        self.state = new_state
        self.state_timestamps[new_state.name] = time.time()

    def add_tool_interaction(self, call: ToolCall, result: ToolResult) -> None:
        """Record a tool call/result pair."""
        self.tool_history.append((call, result))
        self.tool_calls_made += 1
        self.total_tokens_consumed += result.tokens_used

    def to_audit_log(self) -> dict[str, Any]:
        """Generate audit log entry for this session."""
        return {
            "session_id": self.session_id,
            "target_repo": self.target_repo,
            "final_state": self.state.name,
            "findings_count": len(self.findings),
            "tool_calls_made": self.tool_calls_made,
            "executor_retries": self.executor_retries,
            "total_tokens": self.total_tokens_consumed,
            "elapsed_minutes": round(self.elapsed_minutes, 2),
            "state_trace": [
                {"state": s, "timestamp": t}
                for s, t in self.state_timestamps.items()
            ],
        }


# ---------------------------------------------------------------------------
# State machine transition table
# ---------------------------------------------------------------------------
# fmt: off
TRANSITION_TABLE: dict[tuple[State, Transition], State] = {
    (State.INTAKE,              Transition.TARGET_RECEIVED):        State.ANALYSIS,
    (State.ANALYSIS,            Transition.METADATA_LOADED):        State.ARCHITECT_REASONING,
    (State.ARCHITECT_REASONING, Transition.TOOL_CALL_ISSUED):       State.TOOL_EXPLORATION,
    (State.TOOL_EXPLORATION,    Transition.TOOL_RESULT_RETURNED):   State.ARCHITECT_REASONING,
    (State.ARCHITECT_REASONING, Transition.VULNERABILITY_CONFIRMED): State.VULNERABILITY_REPORT,
    (State.ARCHITECT_REASONING, Transition.NO_VULNERABILITY_FOUND): State.COMPLETED,
    (State.ARCHITECT_REASONING, Transition.TIMEOUT):                State.COMPLETED,
    (State.VULNERABILITY_REPORT, Transition.FIX_SPEC_GENERATED):    State.PATCH_GENERATION,
    (State.PATCH_GENERATION,    Transition.PATCH_GENERATED):        State.EXECUTOR_CODEGEN,
    (State.EXECUTOR_CODEGEN,    Transition.PATCH_GENERATED):        State.SANDBOX_VALIDATION,
    (State.SANDBOX_VALIDATION,  Transition.VALIDATION_PASSED):      State.HUMAN_REVIEW,
    (State.SANDBOX_VALIDATION,  Transition.VALIDATION_FAILED):      State.ERROR_FEEDBACK,
    (State.ERROR_FEEDBACK,      Transition.TOOL_RESULT_RETURNED):   State.ARCHITECT_REASONING,
    (State.ERROR_FEEDBACK,      Transition.MAX_RETRIES_EXCEEDED):   State.FAILED,
    (State.HUMAN_REVIEW,        Transition.HUMAN_APPROVED):         State.AUTO_REMEDIATE,
    (State.HUMAN_REVIEW,        Transition.HUMAN_REJECTED):         State.ARCHITECT_REASONING,
    (State.AUTO_REMEDIATE,      Transition.PATCH_DEPLOYED):         State.COMPLETED,
}
# fmt: on


class StateMachine:
    """
    Deterministic state machine governing the SENTINEL pipeline.

    No probabilistic transitions — every state change is triggered
    by an explicit event. This ensures reproducible, auditable behavior.
    """

    def __init__(
        self,
        transition_table: Optional[dict[tuple[State, Transition], State]] = None,
    ) -> None:
        self.transitions = transition_table or TRANSITION_TABLE
        self._hooks: dict[State, list[Callable[[SessionContext], None]]] = {}

    def transition(
        self,
        context: SessionContext,
        event: Transition,
    ) -> State:
        """
        Execute a state transition.

        Args:
            context: The current session context
            event: The triggering event

        Returns:
            The new state

        Raises:
            ValueError: If the transition is not defined
        """
        current = context.state
        key = (current, event)

        if key not in self.transitions:
            raise ValueError(
                f"Invalid transition: {current.name} + {event.name}. "
                f"Valid events from {current.name}: "
                f"{[t.name for (s, t) in self.transitions if s == current]}"
            )

        new_state = self.transitions[key]
        old_state = context.state
        context.record_state_change(new_state)

        logger.info(
            "[Session %s] %s --%s--> %s",
            context.session_id[:8],
            old_state.name,
            event.name,
            new_state.name,
        )

        # Execute hooks
        if new_state in self._hooks:
            for hook in self._hooks[new_state]:
                hook(context)

        return new_state

    def register_hook(
        self,
        state: State,
        callback: Callable[[SessionContext], None],
    ) -> None:
        """Register a callback to execute when entering a state."""
        if state not in self._hooks:
            self._hooks[state] = []
        self._hooks[state].append(callback)

    def get_valid_transitions(self, state: State) -> list[Transition]:
        """Get all valid transitions from the given state."""
        return [t for (s, t) in self.transitions if s == state]

    def is_terminal(self, state: State) -> bool:
        """Check if a state is terminal (no outgoing transitions)."""
        return state in (State.COMPLETED, State.FAILED)


class Orchestrator:
    """
    High-level orchestrator for the SENTINEL dual-model architecture.

    Manages the interaction between:
    - Architect (Gemma 4 31B): Strategic reasoning + tool calls
    - API Bridge: Deterministic format conversion
    - Executor (DeepSeek-Coder): Code generation
    - Sandbox: Compilation + testing + fuzzing
    - HOTL: Human approval gate
    """

    def __init__(
        self,
        config: Optional[Any] = None,
    ) -> None:
        self.state_machine = StateMachine()
        self.config = config
        self._sessions: dict[str, SessionContext] = {}

    def create_session(
        self,
        target_repo: str,
        target_branch: str = "main",
        max_tool_calls: int = 200,
        timeout_minutes: int = 30,
    ) -> SessionContext:
        """Create a new analysis session."""
        context = SessionContext(
            target_repo=target_repo,
            target_branch=target_branch,
            max_tool_calls=max_tool_calls,
            session_timeout_minutes=timeout_minutes,
        )
        self._sessions[context.session_id] = context
        logger.info(
            "Created session %s for repo: %s",
            context.session_id[:8],
            target_repo,
        )
        return context

    def step(self, context: SessionContext, event: Transition) -> State:
        """Advance the session by one step."""
        # Check timeouts
        if context.is_timed_out:
            logger.warning("Session %s timed out", context.session_id[:8])
            return self.state_machine.transition(context, Transition.TIMEOUT)

        # Check tool budget
        if context.tool_budget_remaining <= 0 and event == Transition.TOOL_CALL_ISSUED:
            logger.warning("Session %s exhausted tool budget", context.session_id[:8])
            return self.state_machine.transition(context, Transition.TIMEOUT)

        return self.state_machine.transition(context, event)

    def format_for_executor(self, finding: VulnerabilityFinding) -> str:
        """
        API Bridge: Format a vulnerability finding into a prompt for
        the Executor model.

        This is the deterministic Node 2 (API Bridge) from the spec.
        """
        schema = finding.to_json_schema()
        return (
            f"You are a code security engineer. Generate a minimal patch to fix "
            f"the following vulnerability.\n\n"
            f"Vulnerability: {schema['cwe']} ({schema['severity']})\n"
            f"Location: {schema['location']}\n"
            f"Language: {schema['target_language']}\n\n"
            f"Vulnerable code:\n```\n{schema['vulnerable_code']}\n```\n\n"
            f"Fix strategy: {schema['fix_strategy']}\n\n"
            f"Generate ONLY the patched code. Do not include explanations.\n"
            f"```{schema['target_language']}\n"
        )

    def format_error_feedback(
        self,
        stderr: str,
        stdout: str,
        finding: VulnerabilityFinding,
    ) -> str:
        """
        API Bridge: Format error feedback from failed execution back
        into a prompt for the Architect to reformulate strategy.

        This is Node 4 (Evaluation → Error routing) from the spec.
        """
        # Clean and truncate stderr
        clean_stderr = stderr.strip()[:2000]
        clean_stdout = stdout.strip()[:1000]

        return (
            f"The previous patch attempt for {finding.cwe_id} at "
            f"{finding.location} failed.\n\n"
            f"Error output:\n```\n{clean_stderr}\n```\n\n"
            f"Standard output:\n```\n{clean_stdout}\n```\n\n"
            f"Please reformulate the fix strategy and identify what "
            f"went wrong with the previous approach."
        )

    def get_session(self, session_id: str) -> Optional[SessionContext]:
        """Retrieve a session by ID."""
        return self._sessions.get(session_id)

    def get_audit_log(self, session_id: str) -> Optional[dict[str, Any]]:
        """Get the audit log for a session."""
        context = self._sessions.get(session_id)
        if context is None:
            return None
        return context.to_audit_log()
