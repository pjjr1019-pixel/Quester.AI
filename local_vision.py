"""Lightweight optional visual-role helpers for bounded on-step observation."""

from __future__ import annotations

from pathlib import Path

from data_structures import ModelRole, VisionInspectionResult


def inspect_image_with_stub(
    path: str | Path,
    *,
    request_text: str,
    extracted_text: str,
    role: ModelRole,
    vision_model: str,
    max_chars: int = 280,
) -> VisionInspectionResult:
    """Inspect one bounded local image with deterministic stub heuristics."""
    source_path = Path(path)
    warnings: list[str] = []
    file_size = 0
    if not source_path.exists():
        warnings.append("missing_source_image")
    else:
        try:
            file_size = source_path.stat().st_size
        except OSError as exc:
            warnings.append(f"source_stat_failed:{exc}")
    clipped_text = str(extracted_text).strip()[: max(1, int(max_chars))]
    role_label = "vision" if role == ModelRole.VISION else "specialist perception"
    summary = (
        f"Stub {role_label} review for '{source_path.name or source_path}': "
        f"{file_size} byte(s) captured for bounded local inspection."
    )
    if clipped_text:
        summary += f" OCR seed: {clipped_text[:120]}."
    if request_text.strip():
        summary += f" Request: {request_text.strip()}"
    extracted_preview = clipped_text or f"{source_path.stem or 'capture'} visual context"
    return VisionInspectionResult(
        status="inspected" if source_path.exists() else "degraded",
        source_path=str(source_path),
        request_text=request_text,
        role=role,
        inspection_backend=f"stub_{role.value}",
        inspection_model=vision_model,
        summary=summary,
        extracted_text=extracted_preview,
        warnings=tuple(warnings),
        degraded_reason="" if source_path.exists() else "missing_source_image",
    )
