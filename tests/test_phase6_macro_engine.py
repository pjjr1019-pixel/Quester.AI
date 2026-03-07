"""Phase 6B deterministic macro-engine tests."""

from __future__ import annotations

import unittest

from data_structures import (
    CompressedTrace,
    ContextFrame,
    Macro,
    MacroProposal,
    OperationStep,
    utc_now,
)
from macro_engine import MacroEngine


class MacroEnginePhase6Tests(unittest.TestCase):
    """Validate IR-backed compilation, replay, and proof guards."""

    def test_compress_builds_ir_backed_trace(self) -> None:
        engine = MacroEngine()
        engine.register_macro("@compose_block", ("@read_question", "@compose_answer"))

        trace = engine.compress(["@compose_block"], task_id="macro-ir")

        self.assertEqual(trace.tokens, ("@compose_block",))
        self.assertEqual(trace.expanded_preview, ("@read_question", "@compose_answer"))
        self.assertEqual(trace.macros_used, ("@compose_block",))
        self.assertEqual(trace.ir_version, "1")
        self.assertEqual([step.opcode for step in trace.operation_stream], ["lookup", "emit"])
        self.assertIsNotNone(trace.canonical_graph)
        self.assertEqual(trace.canonical_graph_builder, "macro_engine_v1")
        self.assertTrue(trace.proof_hash)
        self.assertEqual(trace.decode_hints[0].template, "verified_answer")
        assert trace.canonical_graph is not None
        self.assertIn("macro_definition", {entity.entity_type for entity in trace.canonical_graph.entities})
        self.assertIn("macro_expand", {activity.activity_type for activity in trace.canonical_graph.activities})

        payload = trace.to_dict()
        self.assertNotIn("canonical_graph", payload)
        self.assertEqual(CompressedTrace.from_dict(payload), trace)

    def test_expand_supports_nested_macros(self) -> None:
        engine = MacroEngine()
        engine.register_macro("@inner", ("@read_question",))
        engine.register_macro("@outer", ("@inner", "@compose_answer"))

        expanded = engine.expand(["@outer"])

        self.assertEqual(expanded, ["@read_question", "@compose_answer"])

    def test_recursive_macro_expansion_raises(self) -> None:
        engine = MacroEngine()
        engine.register_macro("@a", ("@b",))
        engine.register_macro("@b", ("@a",))

        with self.assertRaises(ValueError):
            engine.expand(["@a"])

    def test_semantically_equivalent_sequences_share_proof_hash(self) -> None:
        engine = MacroEngine()
        engine.register_macro("@macro_step", ("expand 1", "expand 2"))

        trace_with_macro = engine.compress(["@macro_step", "@compose_answer"], task_id="macro-proof")
        trace_literal = engine.compress(["expand 1", "expand 2", "@compose_answer"], task_id="macro-proof")

        self.assertEqual(trace_with_macro.proof_hash, trace_literal.proof_hash)
        self.assertTrue(engine.verify_round_trip(trace_with_macro))
        self.assertTrue(engine.verify_round_trip(trace_literal))

    def test_verify_round_trip_detects_proof_hash_drift(self) -> None:
        engine = MacroEngine()
        trace = engine.compress(["@read_question", "@compose_answer"], task_id="macro-drift")
        tampered = CompressedTrace.from_dict(
            {
                **trace.to_dict(),
                "proof_hash": "drifted-proof-hash",
            }
        )

        self.assertFalse(engine.verify_round_trip(tampered))

    def test_expand_to_graph_replays_operation_stream(self) -> None:
        engine = MacroEngine()
        trace = engine.compress(["@read_question", "@compose_answer"], task_id="macro-replay")

        replayed = engine.expand_to_graph(trace)

        self.assertEqual(
            [activity.activity_type for activity in replayed.activities],
            [step.opcode for step in trace.operation_stream],
        )
        self.assertEqual(
            replayed.entities[-1].value,
            trace.operation_stream[-1].metadata["source_token"],
        )

    def test_parameterized_macros_compress_and_expand_literal_sequences(self) -> None:
        engine = MacroEngine()
        engine.register_macro(
            Macro(
                macro_name="compare_pair",
                expansion=("compare {lhs}", "compare {rhs}"),
                version=1,
                parameters=("lhs", "rhs"),
                opcode_pattern=("compare", "compare"),
                invariants=("deterministic_round_trip",),
                semantic_kind="motif_macro",
            )
        )

        trace = engine.compress(["compare earth", "compare mars"], task_id="macro-params")

        self.assertEqual(trace.tokens, ("@compare_pair(lhs=earth,rhs=mars)",))
        self.assertEqual(trace.expanded_preview, ("compare earth", "compare mars"))
        self.assertEqual(engine.expand(list(trace.tokens)), ["compare earth", "compare mars"])
        self.assertTrue(engine.verify_round_trip(trace))

    def test_verify_round_trip_rejects_noncanonical_commutative_args(self) -> None:
        engine = MacroEngine()
        trace_created_at = utc_now()
        op_stream = (
            OperationStep(
                op_id="op_bind",
                opcode="bind",
                args=("sym_z", "sym_a"),
                output_ref="sym_answer",
                context_frame_id="ctx_primary",
                metadata={"source_token": "bind sym_z sym_a"},
            ),
        )
        graph = engine._build_canonical_graph("noncanonical-bind", op_stream, created_at=trace_created_at)
        context_frames = (
            ContextFrame(
                frame_id="ctx_primary",
                scope="macro_engine",
                confidence=1.0,
                provenance_bundle_id="bundle_macro_engine",
                created_at=trace_created_at,
            ),
        )
        decode_hints = ()
        proof_hash = engine._compute_proof_hash(
            canonical_graph=graph,
            operation_stream=op_stream,
            context_frames=context_frames,
            decode_hints=decode_hints,
        )
        trace = CompressedTrace(
            task_id="noncanonical-bind",
            tokens=("bind sym_z sym_a",),
            expanded_preview=("bind sym_z sym_a",),
            macros_used=(),
            confidence=1.0,
            ir_version="1",
            canonical_graph=graph,
            canonical_graph_builder="macro_engine_v1",
            operation_stream=op_stream,
            context_frames=context_frames,
            proof_hash=proof_hash,
            decode_hints=decode_hints,
            created_at=trace_created_at,
        )

        self.assertFalse(engine.verify_round_trip(trace))

    def test_macro_proposal_validation_marks_missing_invariants_invalid(self) -> None:
        engine = MacroEngine()

        invalid = engine.validate_macro_proposal(
            MacroProposal(
                proposal_id="proposal-invalid",
                macro=Macro(
                    macro_name="invalid_macro",
                    expansion=("@compose_answer",),
                    version=1,
                ),
                reason="Missing validation metadata.",
                examples=("@compose_answer",),
                simulation_score=0.5,
                approved=False,
            )
        )
        valid = engine.validate_macro_proposal(
            MacroProposal(
                proposal_id="proposal-valid",
                macro=Macro(
                    macro_name="valid_macro",
                    expansion=("@compose_answer",),
                    version=1,
                    opcode_pattern=("emit",),
                    invariants=(
                        "deterministic_round_trip",
                        "provenance_preserving",
                        "uncertainty_preserving",
                    ),
                    semantic_kind="token_macro",
                ),
                reason="Fully declared macro proposal.",
                examples=("@compose_answer",),
                simulation_score=0.5,
                approved=False,
            )
        )

        self.assertFalse(invalid.validation_passed)
        self.assertIn("missing proposal invariants", " ".join(invalid.validation_issues))
        self.assertTrue(valid.validation_passed)
        self.assertTrue(valid.proof_fingerprint)


if __name__ == "__main__":
    unittest.main()
