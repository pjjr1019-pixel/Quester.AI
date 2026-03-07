"""Bounded deterministic verification helpers for checkable questions."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

_ARITHMETIC_PROMPT_RE = re.compile(
    r"(?:what is|calculate|compute|evaluate)\s+([-+*/()%\d\s\.]+)\??",
    re.IGNORECASE,
)
_EVIDENCE_COUNT_RE = re.compile(r"(?:how many|count).*(?:evidence|source)", re.IGNORECASE)
_PYTHON_EXPR_PATTERNS = (
    re.compile(r"python(?: expression)?\s*[:\-]\s*`([^`]+)`", re.IGNORECASE),
    re.compile(r"python(?: expression)?\s*[:\-]\s*(.+)$", re.IGNORECASE),
    re.compile(r"`([^`]+)`\s*(?:in python|python)\??$", re.IGNORECASE),
    re.compile(r"what does\s+`([^`]+)`\s+return\??$", re.IGNORECASE),
)
_PYTHON_CODE_BLOCK_RE = re.compile(r"```python\s*(.*?)```", re.IGNORECASE | re.DOTALL)
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_MAX_AST_NODES = 32
_MAX_EXEC_AST_NODES = 128
_MAX_EXEC_LINES = 40
_ALLOWED_CALLS = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "round": round,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
}
_ALLOWED_EXEC_BUILTINS = {
    **_ALLOWED_CALLS,
    "dict": dict,
    "enumerate": enumerate,
    "range": range,
    "set": set,
}


@dataclass(slots=True, frozen=True)
class ToolVerificationResult:
    """One deterministic verifier outcome."""

    verifier_type: str
    matched: bool
    expected_answer: str
    actual_answer: str
    score: float
    detail: str
    supporting_evidence_ids: tuple[str, ...] = ()


@dataclass(slots=True, frozen=True)
class EvidenceSupportResult:
    """Deterministic grounding estimate for an answer against retrieved evidence."""

    score: float
    supporting_evidence_ids: tuple[str, ...]


def evaluate_arithmetic_question(question: str) -> str | None:
    """Return a deterministic arithmetic answer for simple questions when possible."""
    match = _ARITHMETIC_PROMPT_RE.search(question.strip())
    if match is None:
        return None
    expression = " ".join(match.group(1).split())
    if not expression or len(expression) > 64:
        return None
    try:
        parsed = ast.parse(expression, mode="eval")
    except SyntaxError:
        return None
    if sum(1 for _ in ast.walk(parsed)) > _MAX_AST_NODES:
        return None
    try:
        value = _evaluate_ast_node(parsed)
    except ValueError:
        return None
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return _normalize_numeric(value)


def expected_evidence_count(question: str, evidence_count: int) -> str | None:
    """Return the expected count when the question explicitly asks for evidence totals."""
    if _EVIDENCE_COUNT_RE.search(question.strip()) is None:
        return None
    return str(int(evidence_count))


def evaluate_python_expression_question(question: str) -> str | None:
    """Return a deterministic result for simple Python-expression questions when possible."""
    expression = _extract_python_expression(question)
    if expression is None or len(expression) > 96:
        return None
    try:
        parsed = ast.parse(expression, mode="eval")
    except SyntaxError:
        return None
    if sum(1 for _ in ast.walk(parsed)) > _MAX_AST_NODES:
        return None
    try:
        value = _evaluate_python_ast(parsed)
    except ValueError:
        return None
    return _render_python_value(value)


def evaluate_python_code_question(question: str) -> str | None:
    """Execute a bounded Python snippet and evaluate the requested expression."""
    code_block = _extract_python_code_block(question)
    expression = _extract_python_expression(question)
    if code_block is None or expression is None:
        return None
    try:
        environment = _execute_bounded_python(code_block)
        value = _evaluate_python_ast_with_environment(expression, environment)
    except ValueError:
        return None
    return _render_python_value(value)


def evaluate_python_unit_test_question(question: str) -> str | None:
    """Return pass/fail for bounded unit-test-style Python snippets."""
    lowered = question.lower()
    if not any(token in lowered for token in ("unit test", "pytest", "test", "assert")):
        return None
    code_block = _extract_python_code_block(question)
    if code_block is None:
        return None
    try:
        environment = _execute_bounded_python(code_block)
    except ValueError:
        return None
    test_functions = [
        value for name, value in environment.items() if callable(value) and name.startswith("test_")
    ]
    if not test_functions:
        return None
    try:
        for test_function in test_functions:
            test_function()
    except AssertionError:
        return "fail"
    except Exception:
        return None
    return "pass"


def verify_expected_answer(
    *,
    verifier_type: str,
    expected_answer: str,
    actual_answer: str,
    supporting_evidence_ids: tuple[str, ...] = (),
) -> ToolVerificationResult:
    """Build a normalized verification result for expected-vs-actual comparisons."""
    matched = bool(actual_answer) and normalize_answer_text(actual_answer) == normalize_answer_text(expected_answer)
    detail = (
        f"{verifier_type} matched expected answer '{expected_answer}'."
        if matched
        else f"{verifier_type} expected '{expected_answer}' but saw '{actual_answer or '<empty>'}'."
    )
    return ToolVerificationResult(
        verifier_type=verifier_type,
        matched=matched,
        expected_answer=expected_answer,
        actual_answer=actual_answer,
        score=1.0 if matched else 0.0,
        detail=detail,
        supporting_evidence_ids=supporting_evidence_ids,
    )


def measure_evidence_support(
    answer_text: str,
    evidence_items: Sequence[tuple[str, str]],
) -> EvidenceSupportResult:
    """Estimate whether answer text is grounded in retrieved evidence."""
    normalized_answer = normalize_answer_text(answer_text)
    if not normalized_answer:
        return EvidenceSupportResult(score=0.0, supporting_evidence_ids=())

    answer_tokens = _tokenize(normalized_answer)
    scored_support: list[tuple[float, str]] = []
    best_score = 0.0
    for evidence_id, content in evidence_items:
        normalized_content = normalize_answer_text(content)
        if not normalized_content:
            continue
        if normalized_answer in normalized_content:
            score = 1.0
        else:
            evidence_tokens = _tokenize(normalized_content)
            if not answer_tokens or not evidence_tokens:
                score = 0.0
            else:
                overlap = len(answer_tokens & evidence_tokens)
                coverage = overlap / max(1, len(answer_tokens))
                score = round(min(0.95, coverage), 3)
        if score > 0.0:
            scored_support.append((score, evidence_id))
            best_score = max(best_score, score)
    scored_support.sort(reverse=True)
    support_ids = tuple(evidence_id for score, evidence_id in scored_support if score >= max(0.35, best_score - 0.15))
    return EvidenceSupportResult(score=round(best_score, 3), supporting_evidence_ids=support_ids)


def measure_candidate_agreement(answer_text: str, peer_answers: Iterable[str]) -> float:
    """Estimate how well one candidate agrees with its peers."""
    normalized_answer = normalize_answer_text(answer_text)
    if not normalized_answer:
        return 0.0
    answer_tokens = _tokenize(normalized_answer)
    peer_scores: list[float] = []
    for peer_answer in peer_answers:
        normalized_peer = normalize_answer_text(peer_answer)
        if not normalized_peer:
            continue
        if normalized_peer == normalized_answer:
            peer_scores.append(1.0)
            continue
        peer_tokens = _tokenize(normalized_peer)
        if not answer_tokens or not peer_tokens:
            continue
        overlap = len(answer_tokens & peer_tokens)
        union = len(answer_tokens | peer_tokens)
        peer_scores.append(round(overlap / max(1, union), 3))
    if not peer_scores:
        return 0.5
    return round(sum(peer_scores) / len(peer_scores), 3)


def normalize_answer_text(value: str) -> str:
    """Normalize lightweight answer text for deterministic comparisons."""
    normalized = " ".join(value.strip().lower().split())
    return normalized.rstrip(".")


def _evaluate_ast_node(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _evaluate_ast_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        operand = _evaluate_ast_node(node.operand)
        return operand if isinstance(node.op, ast.UAdd) else -operand
    if isinstance(node, ast.BinOp):
        left = _evaluate_ast_node(node.left)
        right = _evaluate_ast_node(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            if right == 0:
                raise ValueError("division by zero")
            return left / right
        if isinstance(node.op, ast.FloorDiv):
            if right == 0:
                raise ValueError("division by zero")
            return left // right
        if isinstance(node.op, ast.Mod):
            if right == 0:
                raise ValueError("division by zero")
            return left % right
    raise ValueError(f"unsupported expression node: {type(node).__name__}")


def _normalize_numeric(value: float) -> str:
    rendered = f"{value:.6f}".rstrip("0").rstrip(".")
    return rendered or "0"


def _extract_python_expression(question: str) -> str | None:
    stripped = question.strip()
    for pattern in _PYTHON_EXPR_PATTERNS:
        match = pattern.search(stripped)
        if match is not None:
            candidate = " ".join(match.group(1).split()).rstrip("?")
            return candidate.strip()
    return None


def _extract_python_code_block(question: str) -> str | None:
    match = _PYTHON_CODE_BLOCK_RE.search(question)
    if match is None:
        return None
    code = match.group(1).strip()
    if not code:
        return None
    return "\n".join(line.rstrip() for line in code.splitlines()).strip()


def _execute_bounded_python(code: str) -> dict[str, Any]:
    lines = [line for line in code.splitlines() if line.strip()]
    if not lines or len(lines) > _MAX_EXEC_LINES:
        raise ValueError("python snippet exceeds execution line limit")
    try:
        parsed = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        raise ValueError("python snippet failed to parse") from exc
    if sum(1 for _ in ast.walk(parsed)) > _MAX_EXEC_AST_NODES:
        raise ValueError("python snippet exceeds AST budget")
    _validate_exec_ast(parsed)
    environment: dict[str, Any] = {"__builtins__": _ALLOWED_EXEC_BUILTINS}
    try:
        exec(compile(parsed, "<bounded_python>", "exec"), environment, environment)
    except Exception as exc:
        raise ValueError("bounded python execution failed") from exc
    environment.pop("__builtins__", None)
    return environment


def _validate_exec_ast(node: ast.AST) -> None:
    allowed_node_types = (
        ast.Module,
        ast.FunctionDef,
        ast.arguments,
        ast.arg,
        ast.Return,
        ast.Assign,
        ast.Expr,
        ast.Assert,
        ast.If,
        ast.Pass,
        ast.Load,
        ast.Store,
        ast.Constant,
        ast.Name,
        ast.List,
        ast.Tuple,
        ast.Set,
        ast.Dict,
        ast.UnaryOp,
        ast.UAdd,
        ast.USub,
        ast.Not,
        ast.BinOp,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.FloorDiv,
        ast.Mod,
        ast.BoolOp,
        ast.And,
        ast.Or,
        ast.Compare,
        ast.Eq,
        ast.NotEq,
        ast.Lt,
        ast.LtE,
        ast.Gt,
        ast.GtE,
        ast.Call,
        ast.Subscript,
        ast.Slice,
    )
    for child in ast.walk(node):
        if not isinstance(child, allowed_node_types):
            raise ValueError(f"unsupported Python execution node: {type(child).__name__}")
        if isinstance(child, ast.Call):
            if isinstance(child.func, ast.Name):
                function_name = child.func.id
                if function_name not in _ALLOWED_EXEC_BUILTINS and function_name not in _defined_function_names(node):
                    raise ValueError(f"unsupported Python execution call: {function_name}")
            else:
                raise ValueError("only direct function calls are supported")


def _defined_function_names(node: ast.AST) -> set[str]:
    return {child.name for child in ast.walk(node) if isinstance(child, ast.FunctionDef)}


def _evaluate_python_ast_with_environment(expression: str, environment: dict[str, Any]) -> Any:
    try:
        parsed = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError("python expression failed to parse") from exc
    if sum(1 for _ in ast.walk(parsed)) > _MAX_AST_NODES:
        raise ValueError("python expression exceeds AST budget")
    _validate_python_expression_ast(parsed, allowed_names=set(environment))
    try:
        return eval(compile(parsed, "<bounded_python_expr>", "eval"), environment, environment)
    except Exception as exc:
        raise ValueError("bounded python expression failed") from exc


def _validate_python_expression_ast(node: ast.AST, *, allowed_names: set[str]) -> None:
    allowed_node_types = (
        ast.Expression,
        ast.Load,
        ast.Constant,
        ast.Name,
        ast.List,
        ast.Tuple,
        ast.Set,
        ast.Dict,
        ast.UnaryOp,
        ast.UAdd,
        ast.USub,
        ast.Not,
        ast.BinOp,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.FloorDiv,
        ast.Mod,
        ast.BoolOp,
        ast.And,
        ast.Or,
        ast.Compare,
        ast.Eq,
        ast.NotEq,
        ast.Lt,
        ast.LtE,
        ast.Gt,
        ast.GtE,
        ast.Call,
        ast.Subscript,
        ast.Slice,
    )
    for child in ast.walk(node):
        if not isinstance(child, allowed_node_types):
            raise ValueError(f"unsupported Python expression node: {type(child).__name__}")
        if isinstance(child, ast.Name) and child.id not in allowed_names:
            raise ValueError(f"unknown Python name: {child.id}")
        if isinstance(child, ast.Call):
            if not isinstance(child.func, ast.Name) or child.func.id not in allowed_names:
                raise ValueError("only direct calls to bounded names are supported")


def _evaluate_python_ast(node: ast.AST) -> Any:
    if isinstance(node, ast.Expression):
        return _evaluate_python_ast(node.body)
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.List):
        return [_evaluate_python_ast(item) for item in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_evaluate_python_ast(item) for item in node.elts)
    if isinstance(node, ast.Set):
        return {_evaluate_python_ast(item) for item in node.elts}
    if isinstance(node, ast.Dict):
        return {
            _evaluate_python_ast(key): _evaluate_python_ast(value)
            for key, value in zip(node.keys, node.values, strict=True)
        }
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub, ast.Not)):
        operand = _evaluate_python_ast(node.operand)
        if isinstance(node.op, ast.UAdd):
            return +operand
        if isinstance(node.op, ast.USub):
            return -operand
        return not operand
    if isinstance(node, ast.BinOp):
        left = _evaluate_python_ast(node.left)
        right = _evaluate_python_ast(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.FloorDiv):
            return left // right
        if isinstance(node.op, ast.Mod):
            return left % right
        raise ValueError(f"unsupported binary operator: {type(node.op).__name__}")
    if isinstance(node, ast.BoolOp):
        values = [_evaluate_python_ast(value) for value in node.values]
        if isinstance(node.op, ast.And):
            return all(values)
        if isinstance(node.op, ast.Or):
            return any(values)
        raise ValueError(f"unsupported boolean operator: {type(node.op).__name__}")
    if isinstance(node, ast.Compare):
        left = _evaluate_python_ast(node.left)
        for operator, comparator in zip(node.ops, node.comparators, strict=True):
            right = _evaluate_python_ast(comparator)
            if isinstance(operator, ast.Eq):
                matched = left == right
            elif isinstance(operator, ast.NotEq):
                matched = left != right
            elif isinstance(operator, ast.Lt):
                matched = left < right
            elif isinstance(operator, ast.LtE):
                matched = left <= right
            elif isinstance(operator, ast.Gt):
                matched = left > right
            elif isinstance(operator, ast.GtE):
                matched = left >= right
            else:
                raise ValueError(f"unsupported comparison operator: {type(operator).__name__}")
            if not matched:
                return False
            left = right
        return True
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        function_name = node.func.id
        if function_name not in _ALLOWED_CALLS:
            raise ValueError(f"unsupported function call: {function_name}")
        if node.keywords:
            raise ValueError("keyword arguments are not supported")
        arguments = [_evaluate_python_ast(argument) for argument in node.args]
        return _ALLOWED_CALLS[function_name](*arguments)
    if isinstance(node, ast.Subscript):
        container = _evaluate_python_ast(node.value)
        index = _evaluate_python_ast(node.slice)
        return container[index]
    if isinstance(node, ast.Slice):
        lower = _evaluate_python_ast(node.lower) if node.lower is not None else None
        upper = _evaluate_python_ast(node.upper) if node.upper is not None else None
        step = _evaluate_python_ast(node.step) if node.step is not None else None
        return slice(lower, upper, step)
    raise ValueError(f"unsupported Python expression node: {type(node).__name__}")


def _render_python_value(value: Any) -> str:
    if isinstance(value, float):
        return str(int(value)) if value.is_integer() else _normalize_numeric(value)
    if isinstance(value, str):
        return value
    return repr(value)


def _tokenize(value: str) -> set[str]:
    return set(_TOKEN_RE.findall(value.lower()))
