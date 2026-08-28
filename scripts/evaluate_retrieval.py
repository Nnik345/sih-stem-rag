#!/usr/bin/env python3
"""Score the hybrid retriever against a manually labelled question set.

This is a scaffold: it computes nothing unless real labels exist in
``data/evaluation/retrieval_questions.jsonl``. Questions that lack the labels a
metric requires are skipped and reported as skipped, never scored as zero, so
the output can never look better (or worse) than the labelling effort justifies.

Usage:
    python scripts/evaluate_retrieval.py
    python scripts/evaluate_retrieval.py --k 10 --no-rerank
    python scripts/evaluate_retrieval.py --questions data/evaluation/my_set.jsonl
    python scripts/evaluate_retrieval.py --per-question --json out.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rag.config import ConfigError, load_config  # noqa: E402
from rag.logging_utils import get_logger, setup_logging  # noqa: E402
from rag.neo4j_store import Neo4jStore, Neo4jUnavailableError  # noqa: E402
from rag.pipeline import HybridRetriever  # noqa: E402
from rag.schemas import RetrievalFilter, RetrievedChunk  # noqa: E402

LOGGER = get_logger("evaluate_retrieval")

DEFAULT_QUESTIONS = PROJECT_ROOT / "data" / "evaluation" / "retrieval_questions.jsonl"
EXAMPLE_QUESTIONS = (
    PROJECT_ROOT / "data" / "evaluation" / "retrieval_questions.example.jsonl"
)


class EvaluationDataError(RuntimeError):
    """The question file is missing or malformed."""


@dataclass
class EvalQuestion:
    """One labelled evaluation record. See data/evaluation/README.md."""

    question_id: str
    question: str
    grade: int | None = None
    subject: str | None = None
    expected_unit: str | None = None
    relevant_chunk_ids: list[str] = field(default_factory=list)
    relevance_grades: dict[str, float] = field(default_factory=dict)
    expected_insufficient: bool = False
    notes: str = ""

    @property
    def has_chunk_labels(self) -> bool:
        return bool(self.relevant_chunk_ids)

    @property
    def has_unit_label(self) -> bool:
        return bool(self.expected_unit)

    def gain(self, chunk_id: str) -> float:
        """Graded relevance: explicit grade, else 1 for labelled, else 0."""
        if chunk_id in self.relevance_grades:
            return float(self.relevance_grades[chunk_id])
        return 1.0 if chunk_id in self.relevant_chunk_ids else 0.0

    def scope(self) -> RetrievalFilter:
        return RetrievalFilter(grade=self.grade, subject=self.subject)


def load_questions(path: Path) -> list[EvalQuestion]:
    """Parse the JSONL question set, failing loudly on malformed records."""
    if not path.exists():
        raise EvaluationDataError(
            f"No evaluation set at {path}.\n"
            f"Create it by hand (see {EXAMPLE_QUESTIONS.relative_to(PROJECT_ROOT)} "
            "for the schema and data/evaluation/README.md for the labelling "
            "procedure). No labels ship with this repository on purpose."
        )

    questions: list[EvalQuestion] = []
    seen: set[str] = set()
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("//"):
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EvaluationDataError(f"{path}:{line_number}: invalid JSON: {exc}")
        if not isinstance(record, dict):
            raise EvaluationDataError(f"{path}:{line_number}: expected a JSON object")

        question_id = str(record.get("question_id") or f"line-{line_number}")
        text = (record.get("question") or "").strip()
        if not text:
            raise EvaluationDataError(
                f"{path}:{line_number}: record '{question_id}' has no 'question'"
            )
        if question_id in seen:
            raise EvaluationDataError(
                f"{path}:{line_number}: duplicate question_id '{question_id}'"
            )
        seen.add(question_id)

        grade = record.get("grade")
        if grade is not None and not isinstance(grade, int):
            raise EvaluationDataError(
                f"{path}:{line_number}: 'grade' must be an integer or null"
            )
        chunk_ids = record.get("relevant_chunk_ids") or []
        if not isinstance(chunk_ids, list):
            raise EvaluationDataError(
                f"{path}:{line_number}: 'relevant_chunk_ids' must be a list"
            )
        grades_map = record.get("relevance_grades") or {}
        if not isinstance(grades_map, dict):
            raise EvaluationDataError(
                f"{path}:{line_number}: 'relevance_grades' must be an object"
            )

        subject = record.get("subject")
        questions.append(
            EvalQuestion(
                question_id=question_id,
                question=text,
                grade=grade,
                subject=str(subject).lower() if subject else None,
                expected_unit=record.get("expected_unit") or None,
                relevant_chunk_ids=[str(c) for c in chunk_ids],
                relevance_grades={str(k): float(v) for k, v in grades_map.items()},
                expected_insufficient=bool(record.get("expected_insufficient", False)),
                notes=str(record.get("notes") or ""),
            )
        )

    if not questions:
        raise EvaluationDataError(f"{path} contains no records")
    return questions


def recall_at_k(question: EvalQuestion, ranked: Sequence[str], k: int) -> float:
    relevant = set(question.relevant_chunk_ids)
    hits = sum(1 for chunk_id in ranked[:k] if chunk_id in relevant)
    return hits / len(relevant)


def reciprocal_rank(question: EvalQuestion, ranked: Sequence[str]) -> float:
    relevant = set(question.relevant_chunk_ids)
    for rank, chunk_id in enumerate(ranked, start=1):
        if chunk_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(question: EvalQuestion, ranked: Sequence[str], k: int) -> float:
    """Graded nDCG@K with the standard 2^gain - 1 numerator."""
    dcg = 0.0
    for rank, chunk_id in enumerate(ranked[:k], start=1):
        gain = question.gain(chunk_id)
        if gain > 0:
            dcg += (2**gain - 1) / math.log2(rank + 1)

    ideal_gains = sorted(
        (question.gain(c) for c in question.relevant_chunk_ids), reverse=True
    )
    idcg = sum(
        (2**gain - 1) / math.log2(rank + 1)
        for rank, gain in enumerate(ideal_gains[:k], start=1)
        if gain > 0
    )
    return dcg / idcg if idcg > 0 else 0.0


def unit_hit_at_k(question: EvalQuestion, chunks: Sequence[RetrievedChunk], k: int) -> float:
    """1.0 if the expected unit appears in the top K, else 0.0.

    Matches either the exact ``unit_id`` or a case-insensitive substring of the
    unit title, so a readable label such as "Sun, Moon" also works.
    """
    expected = (question.expected_unit or "").strip().lower()
    for chunk in chunks[:k]:
        if chunk.unit_id and chunk.unit_id.lower() == expected:
            return 1.0
        if chunk.unit_title and expected in chunk.unit_title.lower():
            return 1.0
    return 0.0


@dataclass
class StageMetrics:
    """Accumulated metrics for one retrieval stage (fused or reranked)."""

    stage: str
    recall: list[float] = field(default_factory=list)
    mrr: list[float] = field(default_factory=list)
    ndcg: list[float] = field(default_factory=list)
    unit_hit: list[float] = field(default_factory=list)

    def summary(self, k: int) -> dict[str, Any]:
        def mean(values: list[float]) -> float | None:
            return round(sum(values) / len(values), 4) if values else None

        return {
            "stage": self.stage,
            f"recall@{k}": mean(self.recall),
            "mrr": mean(self.mrr),
            f"ndcg@{k}": mean(self.ndcg),
            f"unit_hit_rate@{k}": mean(self.unit_hit),
            "questions_with_chunk_labels": len(self.recall),
            "questions_with_unit_labels": len(self.unit_hit),
        }


def evaluate(
    retriever: HybridRetriever,
    questions: Sequence[EvalQuestion],
    *,
    k: int,
    use_reranker: bool,
) -> dict[str, Any]:
    stages = {
        "fused": StageMetrics("fused"),
        "reranked": StageMetrics("reranked"),
    }
    per_question: list[dict[str, Any]] = []
    gate_expectations = {"checked": 0, "correct": 0}

    for index, question in enumerate(questions, start=1):
        LOGGER.info(
            "[%d/%d] %s (%s)",
            index,
            len(questions),
            question.question_id,
            question.scope().describe(),
        )
        response = retriever.retrieve(
            question.question,
            grade=question.grade,
            subject=question.subject,
            rerank=use_reranker,
            final_top_k=k,
        )
        diagnostics = response.diagnostics
        stage_results = {
            "fused": diagnostics.fused,
            "reranked": diagnostics.reranked or response.results,
        }

        record: dict[str, Any] = {
            "question_id": question.question_id,
            "question": question.question,
            "grade": question.grade,
            "subject": question.subject,
            "labels": {
                "chunk_level": question.has_chunk_labels,
                "unit_level": question.has_unit_label,
            },
            "candidate_counts": {
                "dense": len(diagnostics.dense),
                "fulltext": len(diagnostics.fulltext),
                "graph": len(diagnostics.graph),
                "fused": len(diagnostics.fused),
                "reranked": len(diagnostics.reranked),
            },
            "top_chunk_ids": [c.chunk_id for c in stage_results["reranked"][:k]],
            "stages": {},
        }

        for stage_name, chunks in stage_results.items():
            metrics = stages[stage_name]
            ranked_ids = [c.chunk_id for c in chunks]
            stage_record: dict[str, Any] = {}

            if question.has_chunk_labels:
                r = recall_at_k(question, ranked_ids, k)
                rr = reciprocal_rank(question, ranked_ids)
                nd = ndcg_at_k(question, ranked_ids, k)
                metrics.recall.append(r)
                metrics.mrr.append(rr)
                metrics.ndcg.append(nd)
                stage_record |= {
                    f"recall@{k}": round(r, 4),
                    "reciprocal_rank": round(rr, 4),
                    f"ndcg@{k}": round(nd, 4),
                }
            if question.has_unit_label:
                hit = unit_hit_at_k(question, chunks, k)
                metrics.unit_hit.append(hit)
                stage_record[f"unit_hit@{k}"] = hit

            record["stages"][stage_name] = stage_record

        if question.expected_insufficient:
            gate_expectations["checked"] += 1
            # An empty final list is the retriever-level signal; the full
            # evidence gate lives in the RAG pipeline, not the retriever.
            if not response.results:
                gate_expectations["correct"] += 1
            record["expected_insufficient"] = True
            record["returned_candidates"] = len(response.results)

        per_question.append(record)

    unlabelled = [q.question_id for q in questions if not q.has_chunk_labels]
    return {
        "k": k,
        "reranker_used": use_reranker,
        "questions_total": len(questions),
        "questions_without_chunk_labels": unlabelled,
        "stages": [stages["fused"].summary(k), stages["reranked"].summary(k)],
        "insufficient_evidence_expectations": gate_expectations,
        "per_question": per_question,
    }


def print_report(report: dict[str, Any], *, per_question: bool) -> None:
    k = report["k"]
    print()
    print("=" * 78)
    print("RETRIEVAL EVALUATION")
    print("=" * 78)
    print(f"Questions          : {report['questions_total']}")
    print(f"K                  : {k}")
    print(f"Reranker           : {'on' if report['reranker_used'] else 'off'}")

    missing = report["questions_without_chunk_labels"]
    if missing:
        print(
            f"Unlabelled (chunk) : {len(missing)} "
            "-> skipped for Recall/MRR/nDCG, not counted as zero"
        )

    print()
    for stage in report["stages"]:
        print(f"-- {stage['stage'].upper()} " + "-" * (72 - len(stage["stage"])))
        for key, value in stage.items():
            if key == "stage":
                continue
            shown = "n/a (no labels)" if value is None else value
            print(f"   {key:<28} {shown}")
        print()

    gate = report["insufficient_evidence_expectations"]
    if gate["checked"]:
        print(
            f"expected-insufficient questions: {gate['correct']}/{gate['checked']} "
            "returned no candidates"
        )
        print()

    if per_question:
        print("-- PER QUESTION " + "-" * 62)
        for record in report["per_question"]:
            labels = record["labels"]
            tag = (
                "chunk+unit"
                if labels["chunk_level"] and labels["unit_level"]
                else "chunk"
                if labels["chunk_level"]
                else "unit"
                if labels["unit_level"]
                else "UNLABELLED"
            )
            print(f"\n[{record['question_id']}] ({tag}) {record['question']}")
            counts = record["candidate_counts"]
            print(
                f"   candidates: dense={counts['dense']} fulltext={counts['fulltext']} "
                f"graph={counts['graph']} fused={counts['fused']} "
                f"reranked={counts['reranked']}"
            )
            for stage_name, values in record["stages"].items():
                if values:
                    rendered = "  ".join(f"{n}={v}" for n, v in values.items())
                    print(f"   {stage_name:<9} {rendered}")
        print()

    if not any(
        stage.get(f"recall@{k}") is not None for stage in report["stages"]
    ):
        print(
            "No chunk-level metrics were computed: no question carries "
            "relevant_chunk_ids.\n"
            "This is expected until labels are added by hand. See "
            "data/evaluation/README.md."
        )
        print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score the hybrid retriever against manually labelled questions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=DEFAULT_QUESTIONS,
        help=f"JSONL question set (default: {DEFAULT_QUESTIONS.relative_to(PROJECT_ROOT)})",
    )
    parser.add_argument(
        "--k", type=int, default=None, help="Cutoff K (default: final_top_k from config)"
    )
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Score fusion output only; skip loading the reranker",
    )
    parser.add_argument(
        "--per-question", action="store_true", help="Print per-question detail"
    )
    parser.add_argument("--json", type=Path, help="Also write the full report as JSON")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging(args.log_level)

    try:
        questions = load_questions(args.questions)
    except EvaluationDataError as exc:
        print(f"\nERROR: {exc}\n", file=sys.stderr)
        return 2

    try:
        config = load_config()
    except ConfigError as exc:
        print(f"\nERROR: {exc}\n", file=sys.stderr)
        return 2

    k = args.k if args.k is not None else config.retrieval.final_top_k

    retriever = None
    store = None
    try:
        store = Neo4jStore(config.neo4j)
        retriever = HybridRetriever(config, store)
        report = evaluate(
            retriever,
            questions,
            k=k,
            use_reranker=not args.no_rerank,
        )
    except Neo4jUnavailableError as exc:
        print(f"\nERROR: {exc}\n", file=sys.stderr)
        return 3
    finally:
        if retriever is not None:
            retriever.embedder.unload()
            retriever.reranker.unload()
        if store is not None:
            store.close()

    print_report(report, per_question=args.per_question)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Full report written to {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
