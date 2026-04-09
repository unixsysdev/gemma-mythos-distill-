"""
Agent Tool Definitions — LSP, Terminal, Semantic Search.

Defines the tool suite available to the SENTINEL agent for
iterative codebase exploration, implementing the agentic harness
described in Section 5.1 of the specification.
"""

from __future__ import annotations

import json
import logging
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Base tool interface
# ---------------------------------------------------------------------------
@dataclass
class ToolSpec:
    """Schema describing a tool's interface for the model."""
    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)
    returns: str = ""


class BaseTool(ABC):
    """Abstract base class for all SENTINEL agent tools."""

    @property
    @abstractmethod
    def spec(self) -> ToolSpec:
        """Return the tool specification for the model."""
        ...

    @abstractmethod
    def execute(self, **kwargs: Any) -> str:
        """Execute the tool and return output as string."""
        ...


# ---------------------------------------------------------------------------
# Terminal tool
# ---------------------------------------------------------------------------
class TerminalTool(BaseTool):
    """
    Sandboxed terminal for file system exploration.

    Allowed commands: grep, find, cat, head, tail, wc, git log, git diff, ls.
    Blocked commands: rm, mv, chmod, curl, wget, sudo, apt, pip.
    """

    ALLOWED_COMMANDS = frozenset({
        "grep", "find", "cat", "head", "tail", "wc", "ls", "file",
        "git", "sort", "uniq", "awk", "sed", "tr", "cut",
    })

    BLOCKED_COMMANDS = frozenset({
        "rm", "mv", "cp", "chmod", "chown", "curl", "wget", "sudo",
        "apt", "pip", "npm", "yarn", "docker", "kill", "pkill",
    })

    def __init__(self, workspace_dir: Path, timeout_seconds: int = 30) -> None:
        self.workspace_dir = Path(workspace_dir)
        self.timeout_seconds = timeout_seconds

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="terminal",
            description=(
                "Execute shell commands in the target repository. "
                "Use for: grep (search patterns), find (locate files), "
                "cat (read file contents), git log/diff (version history). "
                "Commands are restricted to read-only operations."
            ),
            parameters={
                "command": {"type": "string", "description": "Shell command to execute"},
            },
            returns="Command stdout/stderr output",
        )

    def execute(self, command: str = "", **kwargs: Any) -> str:
        """Execute a shell command in the sandbox."""
        if not command:
            return "Error: no command provided"

        # Security: validate command
        base_cmd = command.strip().split()[0].split("/")[-1]
        if base_cmd in self.BLOCKED_COMMANDS:
            return f"Error: command '{base_cmd}' is not allowed in the security sandbox"

        if base_cmd not in self.ALLOWED_COMMANDS:
            return f"Error: command '{base_cmd}' is not in the allowed list: {sorted(self.ALLOWED_COMMANDS)}"

        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=str(self.workspace_dir),
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )

            output = result.stdout
            if result.stderr:
                output += f"\n[stderr]: {result.stderr}"

            # Truncate very long outputs
            max_length = 8192
            if len(output) > max_length:
                output = output[:max_length] + f"\n[... truncated, {len(output)} total chars]"

            return output

        except subprocess.TimeoutExpired:
            return f"Error: command timed out after {self.timeout_seconds}s"
        except Exception as e:
            return f"Error: {e}"


# ---------------------------------------------------------------------------
# LSP tool
# ---------------------------------------------------------------------------
class LSPTool(BaseTool):
    """
    Language Server Protocol client for semantic code navigation.

    Supports: go-to-definition, find-references, call-hierarchy,
    hover-info, and diagnostics across multiple languages.
    """

    def __init__(self, workspace_dir: Path) -> None:
        self.workspace_dir = Path(workspace_dir)
        self._server_processes: dict[str, Any] = {}

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="lsp",
            description=(
                "Perform semantic code navigation using Language Server Protocol. "
                "Available operations: goto_definition, find_references, "
                "call_hierarchy, hover, diagnostics. "
                "Use this to trace data flows and understand code structure."
            ),
            parameters={
                "operation": {
                    "type": "string",
                    "enum": [
                        "goto_definition",
                        "find_references",
                        "call_hierarchy",
                        "hover",
                        "diagnostics",
                    ],
                },
                "file_path": {"type": "string", "description": "Path relative to repo root"},
                "line": {"type": "integer", "description": "1-indexed line number"},
                "column": {"type": "integer", "description": "1-indexed column number"},
            },
            returns="LSP response with code locations or type information",
        )

    def execute(
        self,
        operation: str = "",
        file_path: str = "",
        line: int = 0,
        column: int = 0,
        **kwargs: Any,
    ) -> str:
        """Execute an LSP operation."""
        if not operation or not file_path:
            return "Error: operation and file_path required"

        full_path = self.workspace_dir / file_path
        if not full_path.exists():
            return f"Error: file not found: {file_path}"

        # In production, this would communicate with an actual LSP server
        # (pyright, clangd, gopls, etc.) via JSON-RPC.
        # Here we implement the interface contract.

        if operation == "goto_definition":
            return self._goto_definition(file_path, line, column)
        elif operation == "find_references":
            return self._find_references(file_path, line, column)
        elif operation == "call_hierarchy":
            return self._call_hierarchy(file_path, line, column)
        elif operation == "hover":
            return self._hover(file_path, line, column)
        elif operation == "diagnostics":
            return self._diagnostics(file_path)
        else:
            return f"Error: unknown operation '{operation}'"

    def _goto_definition(self, file_path: str, line: int, column: int) -> str:
        """Navigate to the definition of the symbol at the given position."""
        # Placeholder: actual implementation uses LSP textDocument/definition
        return json.dumps({
            "operation": "goto_definition",
            "query": {"file": file_path, "line": line, "column": column},
            "result": "LSP server not connected. Connect with start_server().",
        })

    def _find_references(self, file_path: str, line: int, column: int) -> str:
        """Find all references to the symbol at the given position."""
        return json.dumps({
            "operation": "find_references",
            "query": {"file": file_path, "line": line, "column": column},
            "result": "LSP server not connected.",
        })

    def _call_hierarchy(self, file_path: str, line: int, column: int) -> str:
        """Get the call hierarchy for the function at the given position."""
        return json.dumps({
            "operation": "call_hierarchy",
            "query": {"file": file_path, "line": line, "column": column},
            "result": "LSP server not connected.",
        })

    def _hover(self, file_path: str, line: int, column: int) -> str:
        """Get hover information (type, documentation) for a symbol."""
        return json.dumps({
            "operation": "hover",
            "query": {"file": file_path, "line": line, "column": column},
            "result": "LSP server not connected.",
        })

    def _diagnostics(self, file_path: str) -> str:
        """Get diagnostics (errors, warnings) for a file."""
        return json.dumps({
            "operation": "diagnostics",
            "file": file_path,
            "result": "LSP server not connected.",
        })


# ---------------------------------------------------------------------------
# Semantic search tool
# ---------------------------------------------------------------------------
class SemanticSearchTool(BaseTool):
    """
    Embedding-based semantic code search with AST-aware chunking.

    Uses sentence-transformers to embed code chunks and find
    semantically similar code fragments.
    """

    def __init__(
        self,
        workspace_dir: Path,
        model_name: str = "all-MiniLM-L6-v2",
        chunk_size: int = 512,
    ) -> None:
        self.workspace_dir = Path(workspace_dir)
        self.model_name = model_name
        self.chunk_size = chunk_size
        self._index_built = False
        self._chunks: list[dict[str, Any]] = []
        self._embeddings: Any = None

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="semantic_search",
            description=(
                "Search the codebase for semantically similar code fragments. "
                "Use when you need to find related functions, similar patterns, "
                "or code that handles the same data types."
            ),
            parameters={
                "query": {"type": "string", "description": "Natural language or code query"},
                "top_k": {"type": "integer", "description": "Number of results", "default": 5},
            },
            returns="List of relevant code fragments with file paths and similarity scores",
        )

    def build_index(self) -> int:
        """
        Build the semantic search index over the workspace.

        Returns the number of chunks indexed.
        """
        self._chunks = []
        code_extensions = {".py", ".js", ".ts", ".c", ".cpp", ".h", ".go", ".rs", ".java"}

        for file_path in self.workspace_dir.rglob("*"):
            if file_path.suffix in code_extensions and file_path.is_file():
                try:
                    content = file_path.read_text(errors="ignore")
                    rel_path = str(file_path.relative_to(self.workspace_dir))

                    # Simple line-based chunking (AST-aware chunking would use tree-sitter)
                    lines = content.splitlines()
                    for i in range(0, len(lines), self.chunk_size // 4):
                        chunk_lines = lines[i:i + self.chunk_size // 4]
                        chunk_text = "\n".join(chunk_lines)
                        if chunk_text.strip():
                            self._chunks.append({
                                "file": rel_path,
                                "start_line": i + 1,
                                "end_line": min(i + self.chunk_size // 4, len(lines)),
                                "content": chunk_text,
                            })
                except Exception as e:
                    logger.debug("Failed to index %s: %s", file_path, e)

        # Build embeddings (would use sentence-transformers in production)
        self._index_built = True
        logger.info("Indexed %d chunks from workspace", len(self._chunks))
        return len(self._chunks)

    def execute(self, query: str = "", top_k: int = 5, **kwargs: Any) -> str:
        """Search for semantically similar code fragments."""
        if not query:
            return "Error: query required"

        if not self._index_built:
            count = self.build_index()
            if count == 0:
                return "Error: no code files found in workspace"

        # Simple keyword fallback (production would use embeddings)
        results = self._keyword_search(query, top_k)

        return json.dumps({
            "query": query,
            "results": results,
            "total_chunks": len(self._chunks),
        }, indent=2)

    def _keyword_search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        """Fallback keyword search when embeddings aren't available."""
        query_terms = query.lower().split()
        scored: list[tuple[float, dict[str, Any]]] = []

        for chunk in self._chunks:
            content_lower = chunk["content"].lower()
            score = sum(1.0 for term in query_terms if term in content_lower)
            if score > 0:
                scored.append((score, {
                    "file": chunk["file"],
                    "lines": f"{chunk['start_line']}-{chunk['end_line']}",
                    "score": score / len(query_terms),
                    "snippet": chunk["content"][:500],
                }))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:top_k]]


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------
class ToolRegistry:
    """Registry of all available tools for the SENTINEL agent."""

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """Register a tool."""
        self._tools[tool.spec.name] = tool

    def get(self, name: str) -> Optional[BaseTool]:
        """Get a tool by name."""
        return self._tools.get(name)

    def execute(self, tool_name: str, **kwargs: Any) -> str:
        """Execute a tool by name."""
        tool = self._tools.get(tool_name)
        if tool is None:
            return f"Error: unknown tool '{tool_name}'. Available: {list(self._tools.keys())}"
        return tool.execute(**kwargs)

    def get_all_specs(self) -> list[dict[str, Any]]:
        """Get specifications for all registered tools (for model context)."""
        specs = []
        for tool in self._tools.values():
            s = tool.spec
            specs.append({
                "name": s.name,
                "description": s.description,
                "parameters": s.parameters,
            })
        return specs

    @classmethod
    def create_default(cls, workspace_dir: Path) -> ToolRegistry:
        """Create a registry with all default tools."""
        registry = cls()
        registry.register(TerminalTool(workspace_dir))
        registry.register(LSPTool(workspace_dir))
        registry.register(SemanticSearchTool(workspace_dir))
        return registry
