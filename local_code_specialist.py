"""Lightweight optional code-specialist helpers for local file-maintenance tasks."""

from __future__ import annotations

import re
from pathlib import Path

from data_structures import CodeSpecialistResult


def analyze_code_with_stub(
    text: str,
    *,
    request_text: str,
    source_scope: str,
    source_path: str = "",
    code_model: str,
    max_chars: int,
    max_lines: int,
) -> CodeSpecialistResult:
    """Analyze code with deterministic heuristics that stay bounded and explainable."""
    raw_text = str(text)
    clipped_text = raw_text[: max(1, int(max_chars))]
    lines = clipped_text.splitlines()[: max(1, int(max_lines))]
    working_text = "\n".join(lines).strip()
    warnings: list[str] = []
    if len(raw_text) > len(clipped_text):
        warnings.append("code_clipped")
    if len(clipped_text.splitlines()) > len(lines):
        warnings.append("line_cap_reached")
    if not working_text:
        return CodeSpecialistResult(
            status="blocked",
            source_scope=source_scope,
            source_path=source_path,
            request_text=request_text,
            code_backend="stub_code_specialist",
            code_model=code_model,
            warnings=("empty_code_input",),
        )

    detected_language = _detect_language(working_text, source_path=source_path)
    summary, actions = _summarize_code(
        working_text,
        request_text=request_text,
        detected_language=detected_language,
        source_path=source_path,
    )
    return CodeSpecialistResult(
        status="analyzed",
        source_scope=source_scope,
        source_path=source_path,
        request_text=request_text,
        summary=summary,
        suggested_actions=actions,
        code_backend="stub_code_specialist",
        code_model=code_model,
        detected_language=detected_language,
        line_count=len(lines),
        warnings=tuple(warnings),
    )


def analyze_code_file_with_stub(
    path: str | Path,
    *,
    request_text: str,
    code_model: str,
    max_chars: int,
    max_lines: int,
) -> CodeSpecialistResult:
    """Read and analyze one bounded local code file."""
    source_path = Path(path)
    try:
        raw_text = source_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raw_text = source_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return CodeSpecialistResult(
            status="blocked",
            source_scope="file",
            source_path=str(source_path),
            request_text=request_text,
            code_backend="stub_code_specialist",
            code_model=code_model,
            warnings=(f"file_read_failed:{exc}",),
        )
    return analyze_code_with_stub(
        raw_text,
        request_text=request_text,
        source_scope="file",
        source_path=str(source_path),
        code_model=code_model,
        max_chars=max_chars,
        max_lines=max_lines,
    )


def _detect_language(text: str, *, source_path: str) -> str:
    suffix = Path(source_path).suffix.lower()
    if suffix == ".py" or "def " in text or "import " in text:
        return "python"
    if suffix in {".js", ".ts"} or "function " in text or "const " in text:
        return "javascript"
    if suffix in {".ps1"} or "Get-" in text or "$env:" in text:
        return "powershell"
    if suffix in {".json"} or text.strip().startswith("{"):
        return "json"
    if suffix in {".md", ".txt"}:
        return "text"
    return "unknown"


def _summarize_code(
    text: str,
    *,
    request_text: str,
    detected_language: str,
    source_path: str,
) -> tuple[str, tuple[str, ...]]:
    line_count = len(text.splitlines())
    function_count = len(re.findall(r"^\s*(async\s+def|def|function)\s+", text, flags=re.MULTILINE))
    class_count = len(re.findall(r"^\s*class\s+", text, flags=re.MULTILINE))
    todo_count = len(re.findall(r"TODO|FIXME|XXX", text))
    await_count = len(re.findall(r"\bawait\b", text))
    import_count = len(re.findall(r"^\s*(from|import)\s+", text, flags=re.MULTILINE))

    location = source_path or "snippet"
    summary = (
        f"Stub code-specialist review for {location}: {detected_language} with {line_count} line(s), "
        f"{function_count} function-like block(s), {class_count} class(es), {import_count} import(s), "
        f"{await_count} await site(s), and {todo_count} TODO marker(s). "
        f"Request: {request_text or 'Summarize maintenance risks and next steps.'}"
    )

    actions: list[str] = []
    if todo_count:
        actions.append("Review outstanding TODO/FIXME markers before changing behavior.")
    if await_count:
        actions.append("Verify async boundaries and cancellation or timeout handling.")
    if import_count > max(3, function_count + class_count):
        actions.append("Audit imports for unused or overly broad dependencies.")
    if function_count == 0 and class_count == 0:
        actions.append("Confirm whether this file is configuration or glue code before refactoring.")
    actions.append("Add or update a focused test before modifying the highest-risk path.")
    return summary, tuple(actions[:4])

