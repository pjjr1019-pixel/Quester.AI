"""Phase 11 demo-pack loader and validation helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Awaitable, Callable

from config import APP_CONFIG, AppConfig
from data_structures import (
    DecoderEntry,
    DemoDocumentFixture,
    DemoPackStatus,
    DemoRuntimePackSummary,
    Macro,
    OpcodeEntry,
    SampleTaskDefinition,
    VerifiedDeepTraceExport,
    coerce_demo_document_fixture,
    coerce_sample_task_definition,
    coerce_verified_deep_trace_export,
)
from storage import StorageManager
from utils import utc_now_iso


class Phase11ContentLoader:
    """Load the opt-in Phase 11 demo content pack from repo fixtures."""

    _STATUS_KEY = "phase11.demo_pack_status"

    def __init__(self, config: AppConfig = APP_CONFIG, *, pack_dir: Path | None = None) -> None:
        self.config = config
        self.pack_dir = pack_dir or (Path(__file__).resolve().parent / "examples" / "phase11")
        self.demo_corpus_path = self.pack_dir / "demo_corpus.json"
        self.sample_tasks_path = self.pack_dir / "sample_tasks.json"
        self.starter_macros_path = self.pack_dir / "starter_macros.json"
        self.starter_runtime_pack_path = self.pack_dir / "starter_runtime_pack.json"
        self.verified_trace_export_path = self.pack_dir / "verified_deep_trace_export.jsonl"

    def load_demo_documents(self) -> tuple[DemoDocumentFixture, ...]:
        payload = self._read_json_object(self.demo_corpus_path)
        documents = tuple(
            coerce_demo_document_fixture(item) for item in payload.get("documents", ())
        )
        self._validate_package(documents=documents)
        return documents

    def load_sample_tasks(self) -> tuple[SampleTaskDefinition, ...]:
        payload = self._read_json_object(self.sample_tasks_path)
        tasks = tuple(
            coerce_sample_task_definition(item) for item in payload.get("tasks", ())
        )
        self._validate_package(sample_tasks=tasks)
        return tasks

    def load_starter_macros(self) -> tuple[Macro, ...]:
        payload = self._read_json_object(self.starter_macros_path)
        macros = tuple(Macro.from_dict(item) for item in payload.get("macros", ()))
        self._validate_package(macros=macros)
        return macros

    def load_starter_runtime_pack(self) -> tuple[tuple[OpcodeEntry, ...], tuple[DecoderEntry, ...], DemoRuntimePackSummary]:
        payload = self._read_json_object(self.starter_runtime_pack_path)
        opcodes = tuple(OpcodeEntry.from_dict(item) for item in payload.get("opcodes", ()))
        decoders = tuple(DecoderEntry.from_dict(item) for item in payload.get("decoders", ()))
        self._validate_package(opcodes=opcodes, decoders=decoders)
        summary = DemoRuntimePackSummary(
            pack_version=str(payload.get("pack_version", "")),
            macro_names=tuple(macro.macro_name for macro in self.load_starter_macros()),
            opcode_names=tuple(opcode.opcode_name for opcode in opcodes),
            decoder_names=tuple(decoder.decoder_name for decoder in decoders),
        )
        return opcodes, decoders, summary

    def load_packaged_verified_trace_exports(self) -> tuple[VerifiedDeepTraceExport, ...]:
        if not self.verified_trace_export_path.exists():
            raise ValueError(f"Missing Phase 11 verified-trace export fixture: {self.verified_trace_export_path}")
        exports: list[VerifiedDeepTraceExport] = []
        with self.verified_trace_export_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    payload = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON on line {line_number} of {self.verified_trace_export_path.name}: {exc}"
                    ) from exc
                exports.append(coerce_verified_deep_trace_export(payload))
        if not exports:
            raise ValueError("Phase 11 verified-trace export fixture is empty.")
        return tuple(exports)

    def get_sample_task(self, sample_id: str) -> SampleTaskDefinition | None:
        normalized = sample_id.strip()
        if not normalized:
            return None
        for sample_task in self.load_sample_tasks():
            if sample_task.sample_id == normalized:
                return sample_task
        return None

    async def load_demo_pack(
        self,
        *,
        storage: StorageManager,
        embed_document: Callable[[str], Awaitable[list[float]]],
        embedding_model_name: str,
    ) -> DemoPackStatus:
        documents = self.load_demo_documents()
        macros = self.load_starter_macros()
        opcodes, decoders, _ = self.load_starter_runtime_pack()
        pack_version = self._package_version()

        for document in documents:
            metadata = {
                "archived": False,
                "corpus_origin": "demo_phase11",
                "corpus_tier": self.config.retrieval.seed_corpus_tier,
                "demo_pack_version": pack_version,
                **document.metadata,
            }
            await storage.ingest_document(
                source_ref=document.source_ref,
                title=document.title,
                content=document.content,
                metadata=metadata,
                embed_document=embed_document,
                embedding_model_name=embedding_model_name,
            )

        for macro in macros:
            await storage.register_macro(macro)
        for opcode in opcodes:
            await storage.register_opcode(opcode)
        for decoder in decoders:
            await storage.register_decoder(decoder)

        await storage.set_kv(
            self._STATUS_KEY,
            {
                "pack_version": pack_version,
                "loaded_at": utc_now_iso(),
            },
        )
        return await self.build_demo_pack_status(storage=storage)

    async def build_demo_pack_status(self, *, storage: StorageManager) -> DemoPackStatus:
        documents = self.load_demo_documents()
        sample_tasks = self.load_sample_tasks()
        macros = self.load_starter_macros()
        opcodes, decoders, summary = self.load_starter_runtime_pack()
        marker = await storage.get_kv(self._STATUS_KEY) or {}

        loaded_document_count = 0
        for document in documents:
            stored = await storage.get_source_document(document.source_ref)
            if stored is not None and str(stored.metadata.get("corpus_origin", "")) == "demo_phase11":
                loaded_document_count += 1

        loaded_macro_names = {macro.macro_name for macro in await storage.list_macros(active_only=False)}
        loaded_opcode_names = {opcode.opcode_name for opcode in await storage.list_opcodes(active_only=False)}
        loaded_decoder_names = {decoder.decoder_name for decoder in await storage.list_decoders(active_only=False)}

        runtime_pack = DemoRuntimePackSummary(
            pack_version=summary.pack_version,
            macro_names=summary.macro_names,
            opcode_names=summary.opcode_names,
            decoder_names=summary.decoder_names,
            loaded_macro_count=sum(1 for macro in macros if macro.macro_name in loaded_macro_names),
            loaded_opcode_count=sum(1 for opcode in opcodes if opcode.opcode_name in loaded_opcode_names),
            loaded_decoder_count=sum(1 for decoder in decoders if decoder.decoder_name in loaded_decoder_names),
        )
        loaded = (
            loaded_document_count == len(documents)
            and runtime_pack.loaded_macro_count == len(macros)
            and runtime_pack.loaded_opcode_count == len(opcodes)
            and runtime_pack.loaded_decoder_count == len(decoders)
        )
        status_detail = (
            "Phase 11 demo pack is loaded."
            if loaded
            else "Phase 11 demo pack is available but not loaded into local storage yet."
        )
        return DemoPackStatus(
            pack_version=self._package_version(),
            loaded=loaded,
            document_count=len(documents),
            loaded_document_count=loaded_document_count,
            sample_task_count=len(sample_tasks),
            runtime_pack=runtime_pack,
            verified_trace_example_path=str(self.verified_trace_export_path),
            loaded_at=str(marker.get("loaded_at", "")),
            status_detail=status_detail,
        )

    async def export_packaged_verified_trace_example(self, export_path: Path) -> Path:
        exports = self.load_packaged_verified_trace_exports()
        export_path.parent.mkdir(parents=True, exist_ok=True)
        with export_path.open("w", encoding="utf-8") as handle:
            for export_row in exports:
                handle.write(json.dumps(export_row.to_dict(), sort_keys=True) + "\n")
        return export_path

    def _package_version(self) -> str:
        versions = {
            self._read_json_object(self.demo_corpus_path).get("pack_version", ""),
            self._read_json_object(self.sample_tasks_path).get("pack_version", ""),
            self._read_json_object(self.starter_macros_path).get("pack_version", ""),
            self._read_json_object(self.starter_runtime_pack_path).get("pack_version", ""),
        }
        versions.discard("")
        if len(versions) != 1:
            raise ValueError(f"Phase 11 fixture pack versions do not match: {sorted(versions)}")
        return next(iter(versions))

    def _read_json_object(self, path: Path) -> dict[str, Any]:
        if not path.exists():
            raise ValueError(f"Missing Phase 11 fixture file: {path}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {path.name}: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"Phase 11 fixture {path.name} must contain a top-level object.")
        return payload

    def _validate_package(
        self,
        *,
        documents: tuple[DemoDocumentFixture, ...] | None = None,
        sample_tasks: tuple[SampleTaskDefinition, ...] | None = None,
        macros: tuple[Macro, ...] | None = None,
        opcodes: tuple[OpcodeEntry, ...] | None = None,
        decoders: tuple[DecoderEntry, ...] | None = None,
    ) -> None:
        documents = documents if documents is not None else tuple(
            coerce_demo_document_fixture(item)
            for item in self._read_json_object(self.demo_corpus_path).get("documents", ())
        )
        sample_tasks = sample_tasks if sample_tasks is not None else tuple(
            coerce_sample_task_definition(item)
            for item in self._read_json_object(self.sample_tasks_path).get("tasks", ())
        )
        macros = macros if macros is not None else tuple(
            Macro.from_dict(item) for item in self._read_json_object(self.starter_macros_path).get("macros", ())
        )
        if opcodes is None or decoders is None:
            runtime_payload = self._read_json_object(self.starter_runtime_pack_path)
            opcodes = opcodes if opcodes is not None else tuple(
                OpcodeEntry.from_dict(item) for item in runtime_payload.get("opcodes", ())
            )
            decoders = decoders if decoders is not None else tuple(
                DecoderEntry.from_dict(item) for item in runtime_payload.get("decoders", ())
            )

        self._package_version()
        self._require_unique("document source refs", (document.source_ref for document in documents))
        self._require_unique("sample ids", (sample_task.sample_id for sample_task in sample_tasks))
        self._require_unique("macro names", (macro.macro_name for macro in macros))
        self._require_unique("opcode names", (opcode.opcode_name for opcode in opcodes))
        self._require_unique("decoder names", (decoder.decoder_name for decoder in decoders))

        sample_ids = {sample_task.sample_id for sample_task in sample_tasks}
        document_refs = {document.source_ref for document in documents}
        for document in documents:
            unknown_task_ids = sorted(set(document.sample_task_ids) - sample_ids)
            if unknown_task_ids:
                raise ValueError(
                    f"Document fixture '{document.source_ref}' references unknown sample ids: {unknown_task_ids}"
                )
        for sample_task in sample_tasks:
            if sample_task.requires_demo_pack:
                unknown_refs = sorted(set(sample_task.required_source_refs) - document_refs)
                if unknown_refs:
                    raise ValueError(
                        f"Sample task '{sample_task.sample_id}' references unknown demo sources: {unknown_refs}"
                    )

    def _require_unique(self, label: str, values: tuple[str, ...] | Any) -> None:
        normalized = [str(value).strip() for value in values if str(value).strip()]
        if len(normalized) != len(set(normalized)):
            raise ValueError(f"Phase 11 fixture {label} must be unique.")
