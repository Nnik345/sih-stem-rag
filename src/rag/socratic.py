"""Socratic tutoring controller: the only place tutoring behaviour is defined.

Retrieval knows nothing about pedagogy and pedagogy knows nothing about Cypher.
This module owns the system prompt, the evidence block format and the
conversational states, so tutoring behaviour can be changed in one file.

The controller always enters :data:`TutorState.ASK_QUESTION` (or
:data:`TutorState.INSUFFICIENT_EVIDENCE`) because there is no student model yet.
The state machine exists so that later turns can select a different state -- the
design does not assume every interaction is a one-shot answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence

from .config import ModelConfig
from .logging_utils import get_logger
from .schemas import EvidenceDecision, RetrievalFilter, RetrievedChunk

LOGGER = get_logger(__name__)


class TutorState(str, Enum):
    """Tutoring moves the controller can request from the generator."""

    ASK_QUESTION = "ASK_QUESTION"
    GIVE_HINT = "GIVE_HINT"
    CORRECT_MISCONCEPTION = "CORRECT_MISCONCEPTION"
    EXPLAIN_CONCEPT = "EXPLAIN_CONCEPT"
    CONFIRM_STEP = "CONFIRM_STEP"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


_STATE_INSTRUCTIONS: dict[TutorState, str] = {
    TutorState.ASK_QUESTION: (
        "Open with one short, concrete question that gets the student thinking "
        "about the first step. Do not state the final answer."
    ),
    TutorState.GIVE_HINT: (
        "Give exactly one small hint that unblocks the immediate step, then ask "
        "the student to try that step. Do not complete the work for them."
    ),
    TutorState.CORRECT_MISCONCEPTION: (
        "Acknowledge what the student got right, then ask a question that makes "
        "the incorrect assumption visible. Correct it using the evidence only."
    ),
    TutorState.EXPLAIN_CONCEPT: (
        "Explain the single idea the student is missing in two or three short "
        "sentences grounded in the evidence, then check understanding with a "
        "question."
    ),
    TutorState.CONFIRM_STEP: (
        "Confirm whether the student's step is correct and why, referring to the "
        "evidence, then point to the next step as a question."
    ),
    TutorState.INSUFFICIENT_EVIDENCE: (
        "State plainly that the verified curriculum material does not cover this "
        "question, so you will not guess. Offer what the curriculum does cover "
        "nearby, and invite the student to rephrase or ask their teacher."
    ),
}

# Rules that apply to every state. Kept in one place so grounding requirements
# cannot drift between call sites.
_BASE_RULES = (
    "You are a patient Socratic tutor for primary-school STEM students, working "
    "from an official Core Knowledge curriculum.",
    "Ground every factual statement in the CURRICULUM EVIDENCE below. It is your "
    "only source of curriculum fact.",
    "Never invent curriculum facts, definitions, numbers, page references or "
    "lesson content that the evidence does not contain.",
    "Never reveal the complete solution immediately. Guide the student with "
    "questions, hints and small steps so they reach it themselves.",
    "Ask one question at a time and wait for the student.",
    "If the evidence is insufficient, say that the verified curriculum evidence "
    "is insufficient rather than making something up.",
    "Do not cite URLs or invent sources. Refer to material in plain language "
    "(for example \"your unit on plants\") rather than quoting page numbers.",
    "Keep your reply under about 180 words.",
)

_GRADE_GUIDANCE: dict[int, str] = {
    1: (
        "The student is in Grade 1 (age ~6-7). Use very short sentences, everyday "
        "words, and concrete objects they can picture or count."
    ),
    2: (
        "The student is in Grade 2 (age ~7-8). Use short sentences and concrete "
        "examples; introduce a curriculum term only after explaining it simply."
    ),
    3: (
        "The student is in Grade 3 (age ~8-9). Simple sentences are still best; "
        "they can follow two-step reasoning and basic curriculum vocabulary."
    ),
}


@dataclass
class TutorTurn:
    """One tutoring turn: everything the generator needs, and nothing more."""

    question: str
    state: TutorState
    system_prompt: str
    user_prompt: str
    scope: RetrievalFilter
    evidence: list[RetrievedChunk] = field(default_factory=list)
    # Source metadata retained internally so citations can be exposed later.
    provenance: list[dict[str, object]] = field(default_factory=list)

    @property
    def messages(self) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.user_prompt},
        ]


class SocraticController:
    """Builds prompts for the generator from a question plus graded evidence."""

    def __init__(self, *, max_evidence_chars: int = 1000) -> None:
        # Chunks are ~600 tokens; trimming keeps the prompt inside a comfortable
        # window given that Qwen3-VL is partly CPU-offloaded here.
        self.max_evidence_chars = max_evidence_chars

    @classmethod
    def from_config(cls, config: ModelConfig) -> "SocraticController":
        return cls(max_evidence_chars=config.generator_max_evidence_chars)

    # -- prompt pieces ----------------------------------------------------- #

    def system_prompt(
        self, state: TutorState, scope: RetrievalFilter
    ) -> str:
        lines = list(_BASE_RULES)
        if scope.grade is not None and scope.grade in _GRADE_GUIDANCE:
            lines.append(_GRADE_GUIDANCE[scope.grade])
        elif scope.grade is not None:
            lines.append(
                f"The student is in Grade {scope.grade}; keep language and examples "
                f"appropriate to that grade."
            )
        if scope.subject:
            lines.append(f"The subject is {scope.subject}.")
        lines.append(f"Current tutoring move: {state.value}. {_STATE_INSTRUCTIONS[state]}")
        return "\n".join(f"- {line}" for line in lines)

    def format_evidence(self, evidence: Sequence[RetrievedChunk]) -> str:
        """Render evidence with internal provenance labels the model can refer to."""
        if not evidence:
            return "(no curriculum evidence retrieved)"
        blocks = []
        for index, chunk in enumerate(evidence, start=1):
            text = " ".join(chunk.text.split())
            if len(text) > self.max_evidence_chars:
                text = text[: self.max_evidence_chars].rsplit(" ", 1)[0] + " ..."
            header = (
                f"[E{index}] Grade {chunk.grade} {chunk.subject} | "
                f"unit: {chunk.unit_title} | section: {chunk.section_title} | "
                f"pages {chunk.page_range}"
            )
            blocks.append(f"{header}\n{text}")
        return "\n\n".join(blocks)

    def user_prompt(
        self,
        question: str,
        evidence: Sequence[RetrievedChunk],
        *,
        state: TutorState,
        insufficient_reasons: Sequence[str] = (),
    ) -> str:
        parts = [f"STUDENT QUESTION\n{question}"]

        if state is TutorState.INSUFFICIENT_EVIDENCE:
            reasons = "\n".join(f"- {reason}" for reason in insufficient_reasons) or (
                "- No sufficiently relevant curriculum passage was found."
            )
            parts.append(
                "EVIDENCE STATUS\nThe retrieval system judged the available "
                f"curriculum evidence insufficient:\n{reasons}"
            )
            if evidence:
                parts.append(
                    "NEARBY CURRICULUM MATERIAL (may be off-topic; do not present "
                    f"it as an answer)\n{self.format_evidence(evidence)}"
                )
            parts.append(
                "Tell the student you do not have verified curriculum material "
                "for this question. Do not answer from general knowledge."
            )
        else:
            parts.append(f"CURRICULUM EVIDENCE\n{self.format_evidence(evidence)}")
            parts.append(
                "Respond as the Socratic tutor, using only the evidence above."
            )
        return "\n\n".join(parts)

    # -- turn construction ------------------------------------------------- #

    def select_state(
        self,
        decision: EvidenceDecision,
        *,
        requested_state: TutorState | None = None,
    ) -> TutorState:
        """Choose the tutoring move.

        Insufficient evidence always wins: no requested state can override it.
        """
        if not decision.sufficient:
            return TutorState.INSUFFICIENT_EVIDENCE
        return requested_state or TutorState.ASK_QUESTION

    def build_turn(
        self,
        question: str,
        decision: EvidenceDecision,
        *,
        scope: RetrievalFilter,
        fallback_evidence: Sequence[RetrievedChunk] = (),
        requested_state: TutorState | None = None,
    ) -> TutorTurn:
        state = self.select_state(decision, requested_state=requested_state)
        evidence = (
            list(decision.kept_chunks)
            if decision.sufficient
            else list(fallback_evidence)
        )
        turn = TutorTurn(
            question=question,
            state=state,
            system_prompt=self.system_prompt(state, scope),
            user_prompt=self.user_prompt(
                question,
                evidence,
                state=state,
                insufficient_reasons=decision.reasons,
            ),
            scope=scope,
            evidence=evidence,
            provenance=[chunk.provenance() for chunk in evidence],
        )
        LOGGER.info(
            "Socratic turn prepared: state=%s, evidence chunks=%d",
            state.value,
            len(evidence),
        )
        return turn
