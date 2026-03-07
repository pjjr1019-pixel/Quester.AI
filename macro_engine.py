"""Macro engine scaffold with round-trip verification hooks."""

from __future__ import annotations

from collections.abc import Iterable

from data_structures import CompressedTrace

class MacroEngine:
    """Minimal macro dictionary and transform helpers for Phase 1."""

    def __init__(self) -> None:
        self._macros: dict[str, list[str]] = {}

    def register_macro(self, name: str, expansion: Iterable[str]) -> None:
        self._macros[name] = list(expansion)

    def compress(self, steps: list[str], task_id: str = "macro_engine_task") -> CompressedTrace:
        """Return a typed compressed trace using a placeholder identity transform."""
        tokens = tuple(steps)
        macros_used = tuple(token for token in tokens if token in self._macros)
        return CompressedTrace(
            task_id=task_id,
            tokens=tokens,
            expanded_preview=tokens,
            macros_used=macros_used,
            confidence=1.0,
            reasoner_notes="MacroEngine identity compression.",
        )

    def expand(self, tokens: list[str]) -> list[str]:
        """Expand known macros and pass unknown tokens through."""
        expanded: list[str] = []
        for token in tokens:
            if token in self._macros:
                expanded.extend(self._macros[token])
            else:
                expanded.append(token)
        return expanded

    def verify_round_trip(self, trace: CompressedTrace) -> bool:
        """Check semantic stability of compress(expand(tokens))."""
        recompressed = self.compress(self.expand(list(trace.tokens)), task_id=trace.task_id)
        return tuple(self.expand(list(recompressed.tokens))) == tuple(self.expand(list(trace.tokens)))
