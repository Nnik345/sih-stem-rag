#!/usr/bin/env python3
"""Full hybrid GraphRAG -> evidence gate -> Socratic tutoring, streamed. CLI only.

Order of operations is enforced: retrieval and the evidence gate run *before* the
generator is loaded, and when evidence is insufficient the generator is either
skipped entirely (--strict) or asked to decline rather than invent an answer.

    python scripts/test_rag.py                                    # interactive
    python scripts/test_rag.py -q "how do plants make food" -g 4 -s science
    python scripts/test_rag.py -q "what is a unit fraction" -g 3 -s mathematics --strict
    python scripts/test_rag.py -q "how do I find the area of a rectangle" -g 3 -s mathematics --retrieval-only
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rag.config import ConfigError, load_config  # noqa: E402
from rag.generator import GeneratorError  # noqa: E402
from rag.logging_utils import setup_logging  # noqa: E402
from rag.neo4j_store import Neo4jStore, Neo4jUnavailableError  # noqa: E402
from rag.pipeline import RagResult, SocraticRagPipeline  # noqa: E402
from rag.socratic import TutorState  # noqa: E402

RULE = "=" * 100


def print_diagnostics(result: RagResult) -> None:
    diagnostics = result.retrieval.diagnostics

    print()
    print(RULE)
    print("RETRIEVAL DIAGNOSTICS")
    print(RULE)
    print(f"  query            : {result.query}")
    print(f"  scope            : {result.scope.describe()}")
    print(f"  candidate counts : {diagnostics.channel_counts()}")
    print(
        "  stage timings ms : "
        + ", ".join(f"{k}={v:.1f}" for k, v in diagnostics.timings_ms.items())
    )
    for note in diagnostics.notes:
        print(f"  note             : {note}")

    print()
    print("  Final evidence (after fusion + reranking)")
    if not diagnostics.reranked:
        print("    (none)")
    for index, chunk in enumerate(diagnostics.reranked, start=1):
        rerank = "  --  " if chunk.rerank_score is None else f"{chunk.rerank_score:6.3f}"
        print(
            f"    [E{index}] rerank={rerank} rrf={chunk.rrf_score or 0:.6f} "
            f"sources={chunk.retrieval_source}"
        )
        print(
            f"           G{chunk.grade} {chunk.subject} | role={chunk.source_role} | "
            f"{chunk.source_id} | {chunk.unit_title} | "
            f"{chunk.document_title} | pages {chunk.page_range}"
        )
        print(
            f"           dense#{chunk.dense_rank} ft#{chunk.fulltext_rank} "
            f"graph#{chunk.graph_rank} | licence={chunk.licence} | "
            f"cisce={', '.join(chunk.cisce_outcome_ids) or '(none)'}"
        )
        print(f"           section: {chunk.section_title}")
        print(f"           {chunk.preview(100)}")

    print()
    print(RULE)
    print("EVIDENCE GATE")
    print(RULE)
    verdict = "SUFFICIENT" if result.decision.sufficient else "INSUFFICIENT"
    print(f"  verdict    : {verdict} (confidence label: {result.decision.confidence})")
    for check in result.decision.checks:
        mark = "PASS" if check.passed else "FAIL"
        print(f"  [{mark}] {check.name:26s} {check.detail}")
    for reason in result.decision.reasons:
        print(f"  reason     : {reason}")
    print(
        "  note       : these thresholds are configurable heuristics, not "
        "validated values"
    )


def print_provenance(result: RagResult) -> None:
    if not result.turn.provenance:
        return
    print()
    print(RULE)
    print("SOURCE METADATA (kept internally; no URLs are shown to the student)")
    print(RULE)
    for index, source in enumerate(result.turn.provenance, start=1):
        print(
            f"  [E{index}] grade={source['grade']} subject={source['subject']} "
            f"unit={source['unit_title']}"
        )
        print(
            f"        document={source['document_title']} pages={source['pages']} "
            f"section={source['section_title']}"
        )
        print(
            f"        source={source.get('source_id')} role={source.get('source_role')} "
            f"licence={source.get('licence')}"
        )
        print(
            f"        cisce={source.get('cisce_outcome_ids')} "
            f"alignment={source.get('alignment_status')}"
        )
        print(f"        pdf={source['local_pdf_path']}")
        print(f"        chunk={source['chunk_id']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-q", "--query", help="Student question. Omit to be prompted.")
    parser.add_argument("-g", "--grade", type=int, help="Student grade, e.g. 3")
    parser.add_argument("-s", "--subject", help="science | mathematics")
    parser.add_argument("-u", "--unit", help="Restrict to a unit id")
    parser.add_argument(
        "--state",
        choices=[s.value for s in TutorState if s is not TutorState.INSUFFICIENT_EVIDENCE],
        help="Request a specific tutoring move (default ASK_QUESTION).",
    )
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Stop after the evidence gate; never load the generator.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="On insufficient evidence, skip generation entirely.",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, help="Override generation length."
    )
    parser.add_argument("--json", help="Write the full result record to this JSON file")
    parser.add_argument("--log-level", default="WARNING")
    args = parser.parse_args()

    setup_logging(args.log_level)

    try:
        config = load_config()
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    query, grade, subject = args.query, args.grade, args.subject
    if not query:
        try:
            query = input("Student question: ").strip()
            if grade is None:
                raw = input("Grade (blank for any): ").strip()
                grade = int(raw) if raw else None
            if subject is None:
                raw = input("Subject (science/mathematics, blank for any): ").strip()
                subject = raw or None
        except (EOFError, KeyboardInterrupt):
            print("\nAborted.")
            return 1
    if not query:
        print("No question given.", file=sys.stderr)
        return 2

    if args.max_new_tokens:
        config = config.with_overrides(
            models=replace(
                config.models, generator_max_new_tokens=args.max_new_tokens
            )
        )

    try:
        store = Neo4jStore(config.require_neo4j())
        store.connect()
    except Neo4jUnavailableError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 3

    pipeline: SocraticRagPipeline | None = None
    exit_code = 0
    try:
        pipeline = SocraticRagPipeline(config, store)

        # Stage 1: retrieval + evidence gate. The generator is untouched here.
        result = pipeline.prepare(
            query,
            grade=grade,
            subject=subject,
            unit=args.unit,
            requested_state=TutorState(args.state) if args.state else None,
        )
        print_diagnostics(result)
        print_provenance(result)

        print()
        print(RULE)
        print(f"TUTOR STATE: {result.turn.state.value}")
        print(RULE)

        if args.retrieval_only:
            print(
                "\n--retrieval-only: stopping before generation. "
                "The generator was never loaded."
            )
        elif not result.answered and args.strict:
            print(
                "\nEvidence insufficient and --strict is set: the generator is NOT "
                "called, so no answer can be fabricated.\n"
            )
            print("Structured insufficient-evidence result:")
            print(json.dumps(result.decision.to_dict(), indent=2)[:2000])
        else:
            # Free the embedder and reranker before Qwen3-VL claims memory.
            pipeline.release_retrieval_models()
            print("\nLoading generator (Qwen3-VL-8B-Instruct, partly CPU-offloaded;")
            print("first tokens take a while). Streaming response:\n")
            print("-" * 100)
            try:
                for piece in pipeline.stream_answer(result):
                    print(piece, end="", flush=True)
            except GeneratorError as exc:
                print(f"\n\nERROR: {exc}", file=sys.stderr)
                exit_code = 6
            print()
            print("-" * 100)
            print(
                f"\nGenerated {len(result.response_text)} characters in "
                f"{result.generation_ms / 1000:.1f}s"
            )
            if not result.answered:
                print(
                    "\nNOTE: evidence was judged INSUFFICIENT. The tutor was "
                    "instructed to decline rather than answer from general "
                    "knowledge. Use --strict to skip generation entirely."
                )

        if args.json:
            path = Path(args.json)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(result.to_dict(), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            print(f"\nFull result record written to {path}")

    except Neo4jUnavailableError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 3
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    finally:
        if pipeline is not None:
            pipeline.release_retrieval_models()
        store.close()

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
