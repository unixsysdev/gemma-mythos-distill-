"""CLI: Run the SENTINEL agent for vulnerability analysis."""

from __future__ import annotations

import json
import logging
from pathlib import Path


def main() -> None:
    """Main entry point for sentinel-agent."""
    import argparse

    parser = argparse.ArgumentParser(
        description="SENTINEL: Autonomous Vulnerability Auditor",
    )
    parser.add_argument(
        "target", type=str,
        help="Target repository URL or local path",
    )
    parser.add_argument(
        "--branch", type=str, default="main",
        help="Target branch",
    )
    parser.add_argument(
        "--model", type=str, default="google/gemma-4-31b",
        help="Architect model path",
    )
    parser.add_argument(
        "--executor-model", type=str, default="deepseek-ai/deepseek-coder-v2",
        help="Executor model path",
    )
    parser.add_argument(
        "--max-tool-calls", type=int, default=200,
        help="Maximum tool invocations per session",
    )
    parser.add_argument(
        "--timeout", type=int, default=30,
        help="Session timeout in minutes",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("reports"),
        help="Output directory for reports",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    from sentinel.agent.state_machine import Orchestrator, State, Transition

    orchestrator = Orchestrator()
    session = orchestrator.create_session(
        target_repo=args.target,
        target_branch=args.branch,
        max_tool_calls=args.max_tool_calls,
        timeout_minutes=args.timeout,
    )

    print(f"\n{'='*60}")
    print(f"SENTINEL Autonomous Vulnerability Auditor")
    print(f"{'='*60}")
    print(f"  Session:    {session.session_id[:12]}...")
    print(f"  Target:     {args.target}")
    print(f"  Branch:     {args.branch}")
    print(f"  Architect:  {args.model}")
    print(f"  Executor:   {args.executor_model}")
    print(f"  Tool limit: {args.max_tool_calls}")
    print(f"  Timeout:    {args.timeout} minutes")
    print(f"{'='*60}")

    # Initialize state machine
    orchestrator.step(session, Transition.TARGET_RECEIVED)
    print(f"\n  State: {session.state.name}")

    # In production, the agent loop would run here:
    # while not state_machine.is_terminal(session.state):
    #     architect_output = architect_model.generate(context)
    #     tool_calls = parse_tool_calls(architect_output)
    #     for call in tool_calls:
    #         result = tool_registry.execute(call)
    #         session.add_tool_interaction(call, result)
    #     ...

    # Save audit log
    args.output.mkdir(parents=True, exist_ok=True)
    audit_log = orchestrator.get_audit_log(session.session_id)
    if audit_log:
        audit_path = args.output / f"audit_{session.session_id[:12]}.json"
        with open(audit_path, "w") as f:
            json.dump(audit_log, f, indent=2, default=str)
        print(f"\n  Audit log: {audit_path}")


if __name__ == "__main__":
    main()
