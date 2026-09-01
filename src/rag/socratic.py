"""Socratic tutoring controller: the only place tutoring behaviour is defined.

Retrieval knows nothing about pedagogy and pedagogy knows nothing about Cypher.
This module owns the system prompt, the evidence block format and the
conversational states, so tutoring behaviour can be changed in one file.

The controller defaults to :data:`TutorState.GIVE_HINT` (or
:data:`TutorState.INSUFFICIENT_EVIDENCE` when the evidence gate fails).
:data:`TutorState.EXPLAIN_CONCEPT` and :data:`TutorState.CONFIRM_ANSWER` are
available when requested. Student state is not modelled across turns, but the
interface takes a conversation history so it does not assume one-shot answers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Sequence

from .config import ModelConfig
from .logging_utils import get_logger
from .schemas import EvidenceDecision, RetrievalFilter, RetrievedChunk

LOGGER = get_logger(__name__)


class TutorState(str, Enum):
    """Tutoring moves the controller can request from the generator."""

    GIVE_HINT = "GIVE_HINT"
    EXPLAIN_CONCEPT = "EXPLAIN_CONCEPT"
    CONFIRM_ANSWER = "CONFIRM_ANSWER"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


def _is_mathematics(subject: str | None) -> bool:
    return (subject or "").strip().lower() == "mathematics"


def _state_instruction(state: TutorState, subject: str | None) -> str:
    """Per-state, per-subject generator instruction."""
    maths = _is_mathematics(subject)
    if state is TutorState.GIVE_HINT:
        if maths:
            return (
                "Give exactly one hint toward the next working step. Do not "
                "finish the solution or write out the remaining steps."
            )
        return (
            "Give a small hint about the concept the student needs. Do not "
            "lecture or give a full explanation."
        )
    if state is TutorState.EXPLAIN_CONCEPT:
        if maths:
            return (
                "Give the fully worked solution from the curriculum evidence, "
                "showing each step clearly. You may write at the length needed; "
                "the usual word cap and the rule against revealing the complete "
                "solution do not apply in this move. Stay grounded in the evidence."
            )
        return (
            "Fully explain the concept from the curriculum evidence. You may "
            "write at the length needed; the usual word cap and the rule against "
            "revealing the complete solution do not apply in this move. Stay "
            "grounded in the evidence."
        )
    if state is TutorState.CONFIRM_ANSWER:
        extra = ""
        if maths:
            extra = (
                " If they wrote 'is the derivative/differentiation of EXPR = RESULT?', "
                "they are asking whether RESULT is the derivative of EXPR, not whether "
                "EXPR equals RESULT as an identity."
            )
        return (
            "The student message may include both the problem and their attempted "
            "answer. Judge that attempt from the evidence only. If it is correct, "
            "say so. If it is wrong, say what is off and give a hint for how to "
            "proceed — do not dump the full solution."
            + extra
        )
    if state is TutorState.INSUFFICIENT_EVIDENCE:
        return (
            "State plainly that the verified curriculum material does not cover this "
            "question, so you will not guess. Offer what the curriculum does cover "
            "nearby, and invite the student to rephrase or ask their teacher. Do not "
            "give the numeric or algebraic answer from general knowledge while "
            "declining."
        )
    raise ValueError(f"Unknown tutor state: {state}")


# Grounding rules apply to every state so they cannot drift between call sites.
_GROUNDING_RULES = (
    "You are a patient Socratic tutor for CBSE STEM students in classes 1–12, "
    "working from official NCERT Mathematics and Science textbooks "
    "(including EVS, Physics, Chemistry and Biology where those books are the "
    "class science curriculum). NCERT material is copyrighted; do not present "
    "it as your own original curriculum.",
    "Ground every factual statement in the CURRICULUM EVIDENCE below. It is your "
    "only source of curriculum fact.",
    "Never invent curriculum facts, definitions, numbers, page references or "
    "lesson content that the evidence does not contain.",
    "The student may attach a photo of their work or a diagram as input. Use it "
    "together with the evidence text. Do not present, describe, or invent a "
    "textbook figure, and do not generate a drawing.",
    "If the evidence is insufficient, say that the verified curriculum evidence "
    "is insufficient rather than making something up. Do not identify a diagram "
    "from parametric knowledge.",
    "Do not cite URLs or invent sources. Refer to material in plain language "
    "(for example \"your unit on plants\") rather than quoting page numbers.",
)

_APPLY_RULE = (
    "If the evidence states a general rule that applies to the student's "
    "instance, apply that rule to their expression or numbers. Do not refuse "
    "only because their exact example is not printed in the book. Do not use a "
    "rule that is not in the evidence.",
    "If some evidence is from an earlier class than the student is in now, you "
    "may say it was taught in that earlier class. Do not invent text from a "
    "class that was not retrieved.",
)

# Socratic pacing: used for GIVE_HINT and CONFIRM_ANSWER, not EXPLAIN_CONCEPT.
_GUIDED_RULES = (
    "Never reveal the complete solution immediately. Guide the student with "
    "questions, hints and small steps so they reach it themselves.",
    "Ask one question at a time and wait for the student.",
    "Keep your reply under about 180 words.",
)

_SCOPE_GUIDANCE: dict[tuple[int, str], str] = {
    (1, "mathematics"): (
        "The student is in Class 1, using NCERT Joyful-Mathematics. Use very "
        "short sentences, concrete objects they can count or shape with their "
        "hands, and everyday examples. Avoid symbols, letters-as-variables, and "
        "any idea that book has not introduced."
    ),
    (2, "mathematics"): (
        "The student is in Class 2, using NCERT Joyful-Mathematics. Keep language "
        "simple and concrete; one idea per sentence. Number sense, patterns and "
        "spatial ideas come from objects and pictures, not formal notation."
    ),
    (3, "mathematics"): (
        "The student is in Class 3, using NCERT Maths Mela. Simple sentences are "
        "still best; they can follow two-step counting, place value and basic "
        "operations if each step is named."
    ),
    (3, "science"): (
        "The student is in Class 3, using NCERT Our Wondrous World (EVS). Talk "
        "about plants, animals, food, water and the local environment in everyday "
        "language. Do not use middle-school science jargon or chemical symbols."
    ),
    (4, "mathematics"): (
        "The student is in Class 4, using NCERT Math-Mela. They can follow "
        "multi-step arithmetic if each step is named clearly. Fractions and "
        "measurement stay tied to pictures and real quantities."
    ),
    (4, "science"): (
        "The student is in Class 4, using NCERT Our Wondrous World (EVS). Keep "
        "explanations in terms of what they can observe: living things, materials, "
        "weather and community. Avoid Class 6+ science vocabulary."
    ),
    (5, "mathematics"): (
        "The student is in Class 5, using NCERT Math-Mela. They can handle "
        "curriculum vocabulary after a short reminder. Multi-step word problems "
        "are fine if each operation is named before it is used."
    ),
    (5, "science"): (
        "The student is in Class 5, using NCERT Our Wondrous World (EVS). Stay "
        "with observation, grouping and everyday phenomena. Do not jump to "
        "middle-school chapter language from Curiosity."
    ),
    (6, "mathematics"): (
        "The student is in Class 6, using NCERT Ganita Prakash. They can follow a "
        "short chain of reasoning. Introduce a term from that book, then use it. "
        "Integers, fractions, algebra beginnings and basic geometry belong here; "
        "do not use Class 10–12 calculus language."
    ),
    (6, "science"): (
        "The student is in Class 6, using NCERT Curiosity. Introduce a term from "
        "that book (components of food, grouping materials, motion, living and "
        "non-living), then use it. Keep the chain of reasoning short and Socratic."
    ),
    (7, "mathematics"): (
        "The student is in Class 7, using NCERT Ganita Prakash Parts I and II. "
        "They can handle integers, simple equations, lines and angles, and "
        "comparing quantities if each new term is used in a sentence first."
    ),
    (7, "science"): (
        "The student is in Class 7, using NCERT Curiosity. They can follow a "
        "short experiment-style argument (heat, acids and bases, respiration, "
        "electric current) if each new term is used in a sentence first."
    ),
    (8, "mathematics"): (
        "The student is in Class 8, using NCERT Ganita Prakash Parts I and II. "
        "Multi-step explanations are fine if each step is named: rational numbers, "
        "linear equations, mensuration, data handling. Keep the Socratic question "
        "at the end."
    ),
    (8, "science"): (
        "The student is in Class 8, using NCERT Curiosity. Multi-step explanations "
        "are fine for force, friction, chemical effects of current, cell structure "
        "or conservation, if each step is named and the last line is a question."
    ),
    (9, "mathematics"): (
        "The student is in Class 9, using NCERT Ganita Manjari. They can follow a "
        "structured argument: number systems, polynomials, Euclid, coordinate "
        "geometry, linear equations in two variables. Still ask one question at a "
        "time; do not dump a full proof."
    ),
    (9, "science"): (
        "The student is in Class 9, using NCERT Exploration. They can follow a "
        "structured argument from motion, force, atoms and molecules, tissues or "
        "the fundamental unit of life. Still ask one question at a time."
    ),
    (10, "mathematics"): (
        "The student is in Class 10, using NCERT Mathematics. They can use "
        "real numbers, polynomials, trigonometry, similar triangles and quadratic "
        "equations. Do not skip the reasoning that leads to a formula, and do not "
        "jump to Class 12 calculus."
    ),
    (10, "science"): (
        "The student is in Class 10, using NCERT Science. They can use terminology "
        "from light, electricity, acids-bases-salts, life processes and carbon "
        "compounds. Do not skip the reasoning that leads to a formula or a "
        "diagram-based argument."
    ),
    (11, "mathematics"): (
        "The student is in Class 11, using NCERT Mathematics. Formal language for "
        "sets, relations, functions, trigonometry, sequences, conic sections, "
        "limits and derivatives (introductory) is appropriate. Still do not dump "
        "a full worked solution."
    ),
    (11, "physics"): (
        "The student is in Class 11, using NCERT Physics Parts I and II. Formal "
        "language for units, motion in a plane, laws of motion, work-energy, "
        "gravitation, thermodynamics, waves and oscillations is appropriate. "
        "Stay Socratic; do not dump a full derivation."
    ),
    (11, "chemistry"): (
        "The student is in Class 11, using NCERT Chemistry Parts I and II. Formal "
        "language for some basic concepts of chemistry, structure of atom, "
        "periodicity, chemical bonding, states of matter, thermodynamics, "
        "equilibrium, redox and organic basics is appropriate. One idea, then a "
        "question."
    ),
    (11, "biology"): (
        "The student is in Class 11, using NCERT Biology. Formal language for the "
        "living world, biological classification, plant and animal kingdom, "
        "morphology, cell, biomolecules, cell cycle and plant physiology is "
        "appropriate. Guide with questions; do not recite a whole chapter."
    ),
    (12, "mathematics"): (
        "The student is in Class 12, using NCERT Mathematics Parts I and II. They "
        "can follow relations and functions, inverse trigonometry, matrices, "
        "determinants, continuity, differentiability, applications of derivatives, "
        "integrals, differential equations, vectors and probability. Use the "
        "book's wording (for example 'derivative', 'definite integral'). Keep the "
        "reply Socratic and under the word limit."
    ),
    (12, "physics"): (
        "The student is in Class 12, using NCERT Physics Parts I and II. They can "
        "follow electrostatics, current electricity, magnetism, electromagnetic "
        "induction, AC, EM waves, ray and wave optics, dual nature, atoms, nuclei "
        "and semiconductors. Use the book's wording. Keep the reply Socratic; do "
        "not dump a full numerical solution."
    ),
    (12, "chemistry"): (
        "The student is in Class 12, using NCERT Chemistry Parts I and II. They "
        "can follow solutions, electrochemistry, chemical kinetics, d- and f-block, "
        "coordination compounds, haloalkanes, alcohols, aldehydes, amines and "
        "biomolecules. Use NCERT names for reactions and compounds. One step, then "
        "a question."
    ),
    (12, "biology"): (
        "The student is in Class 12, using NCERT Biology. They can follow "
        "reproduction, genetics, evolution, human health, biotechnology, organisms "
        "and populations, ecosystem and biodiversity. Use the book's terms. Guide "
        "with questions rather than reciting a long process."
    ),
}


def scope_guidance(grade: int | None, subject: str | None) -> str:
    """Personalised tutoring line for one legal (class, subject) pair."""
    if grade is None:
        return ""
    key = (grade, (subject or "").lower())
    if key in _SCOPE_GUIDANCE:
        return _SCOPE_GUIDANCE[key]
    if subject:
        return (
            f"The student is in Class {grade}; the subject is {subject}. Keep "
            "language and examples appropriate to that class and subject."
        )
    return (
        f"The student is in Class {grade}; keep language and examples "
        "appropriate to that grade."
    )


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
        lines = list(_GROUNDING_RULES)
        if state is not TutorState.INSUFFICIENT_EVIDENCE:
            lines.extend(_APPLY_RULE)
        if state in (TutorState.GIVE_HINT, TutorState.CONFIRM_ANSWER):
            lines.extend(_GUIDED_RULES)
        guidance = scope_guidance(scope.grade, scope.subject)
        if guidance:
            lines.append(guidance)
        lines.append(
            f"Current tutoring move: {state.value}. "
            f"{_state_instruction(state, scope.subject)}"
        )
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
        attached_figures: Sequence[dict[str, Any]] = (),
    ) -> str:
        parts = [f"STUDENT QUESTION\n{question}"]

        if attached_figures:
            lines = []
            for index, figure in enumerate(attached_figures, start=1):
                page = figure.get("page_number")
                unit = figure.get("unit_title") or ""
                grade = figure.get("grade")
                subject = figure.get("subject") or ""
                page_bit = f" page {page}" if page is not None else ""
                lines.append(
                    f"[Figure {index}] Class {grade} {subject}{page_bit}"
                    + (f" | {unit}" if unit else "")
                    + f" (image_id={figure.get('image_id')})"
                )
            parts.append(
                "ATTACHED TEXTBOOK FIGURES (official NCERT extracts only; "
                "do not invent any other diagram)\n" + "\n".join(lines)
            )
        else:
            parts.append(
                "OUTPUT FIGURES\nNone. Do not sketch, attach, or invent a figure. "
                "If the student uploaded a photo, it is input only."
            )

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
            if state is TutorState.CONFIRM_ANSWER:
                parts.append(
                    "The student message may include both the problem and their "
                    "attempted answer. Judge that attempt from the evidence only."
                )
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
        return requested_state or TutorState.GIVE_HINT

    def build_turn(
        self,
        question: str,
        decision: EvidenceDecision,
        *,
        scope: RetrievalFilter,
        fallback_evidence: Sequence[RetrievedChunk] = (),
        requested_state: TutorState | None = None,
        attached_figures: Sequence[dict[str, Any]] = (),
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
                attached_figures=attached_figures,
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
