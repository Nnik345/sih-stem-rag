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
            pedagogy = (
                "Give one small hint that points to the next useful rule or "
                "working step without applying all the remaining steps. Do not "
                "state or reveal the final answer. End with exactly one guiding "
                "question that helps the student perform that next step."
            )
        else:
            pedagogy = (
                "Give one small hint that points the student toward the relevant "
                "concept without stating the answer or giving a full explanation. "
                "End with exactly one guiding question that helps the student reason "
                "out the next part."
            )
        return (
            pedagogy
            + " Output one JSON object only, no markdown: "
            '{"hint": "...", "guiding_question": "...?"}. '
            "Do not list a sequence of solution steps or a complete explanation."
        )
    if state is TutorState.EXPLAIN_CONCEPT:
        if maths:
            pedagogy = (
                "Explain the mathematical concept clearly. State the relevant "
                "formula or rule when one applies. Then demonstrate it with one "
                "fully worked example that is similar to, but not the same as, "
                "the student's problem: change the values, coefficients or "
                "expression and show every necessary step. Do not solve or reveal "
                "the final answer to the student's exact problem. Do not invent a "
                "formula when the evidence gives only a non-formula rule."
            )
        else:
            pedagogy = (
                "Explain the requested concept clearly and completely at the student's "
                "grade level. Connect the main ideas in a logical order and use a short "
                "example only when the evidence supports it. Do not turn the response "
                "into a quiz or require the student to infer the explanation."
            )
        return (
            pedagogy
            + " Output one JSON object only, no markdown: "
            '{"explanation": "...", "formula_or_rule": "..." or null, '
            '"worked_example": {"problem": "...", "steps": ["..."], "answer": "..."} '
            "or null}. For mathematics, worked_example is required and must differ "
            "from the student's exact problem."
        )
    if state is TutorState.CONFIRM_ANSWER:
        extra = ""
        if maths:
            extra = (
                " For mathematics, silently derive the expected result, simplify "
                "both the expected result and the student's result, and compare "
                "their mathematical meaning rather than their surface form. "
                "Equivalent forms count as correct; omitted zero terms, reordered "
                "terms and valid simplifications are not mistakes. In wording such "
                "as 'is the derivative/differentiation of EXPR = RESULT?', treat "
                "RESULT as the student's proposed derivative of EXPR."
            )
        return (
            "Output one JSON object only, no markdown and no student-facing prose. "
            "Required shape: {\"verdict\": \"correct\" or \"incorrect\", "
            "\"brief_reason\": \"...\", \"mistake_groups\": [{\"mistake\": \"...\", "
            "\"hint\": \"...\"}]}. Complete the full check privately before "
            "writing JSON. Do not expose this private checking process. If the "
            "attempt is correct, verdict is correct, brief_reason is a concise "
            "confirmation, and mistake_groups is an empty list. If it is "
            "incorrect, verdict is incorrect, brief_reason explains why, and "
            "mistake_groups has one entry per distinct mistake (group repeated "
            "instances of the same mistake) with a hint that does not reveal the "
            "corrected final answer or a complete replacement solution. Do not "
            "put Correct., Not quite., or any other verdict prefix in any JSON "
            "field."
            + extra
        )
    if state is TutorState.INSUFFICIENT_EVIDENCE:
        return (
            "State plainly that the verified curriculum material does not cover this "
            "question, so you will not guess. Offer what the curriculum does cover "
            "nearby only when it is genuinely relevant, and invite the student to "
            "rephrase or ask their teacher. Do not answer any part of the question "
            "from general knowledge while declining. "
            "Output one JSON object only, no markdown: "
            '{"decline": "...", "nearby_coverage": "..." or null, "next_step": "..."}.'
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
    "The student cannot see CURRICULUM EVIDENCE or the [E1], [E2], [E3] labels. "
    "Never write phrases such as 'according to the curriculum', 'according to "
    "the evidence' or 'the retrieved material says'. Never mention E1, passage "
    "numbers, chunk ids or page headers. Speak as a teacher in ordinary language.",
)

_APPLY_RULE = (
    "If the evidence states a general rule that applies to the student's "
    "instance, apply that rule to their expression or numbers. Do not refuse "
    "only because their exact example is not printed in the book. Do not use a "
    "rule that is not in the evidence.",
    "A new instance does not need to appear verbatim in the evidence. You may "
    "derive its answer by calculation or logical application of an evidenced "
    "rule. Before making a claim, silently carry out the required reasoning and "
    "check that the claim, calculation and conclusion agree with one another.",
    "If some evidence is from an earlier class than the student is in now, you "
    "may say it was taught in that earlier class. Do not invent text from a "
    "class that was not retrieved.",
)

_CONFIRM_APPLY = (
    "When the current tutoring move is CONFIRM_ANSWER, use the evidenced rule "
    "to perform a complete private check before writing JSON. Compare meaning, "
    "not merely wording or formatting. A different but equivalent expression or "
    "scientifically equivalent statement is correct. Missing detail is not an "
    "error unless that detail is required for the student's claim to be correct.",
    "The JSON verdict must match that private check. Never output a provisional "
    "verdict, student-facing prose, Correct., or Not quite. The student never "
    "sees this JSON.",
)

# Socratic pacing applies only to GIVE_HINT. Other states define their own
# response structure in _state_instruction.
_GUIDED_RULES = (
    "Give only the next useful nudge, not a sequence of hints or steps.",
    "Ask exactly one guiding question and then wait for the student.",
)

_STATE_WORD_LIMITS: dict[TutorState, int] = {
    TutorState.GIVE_HINT: 90,
    TutorState.EXPLAIN_CONCEPT: 450,
    TutorState.CONFIRM_ANSWER: 300,
    TutorState.INSUFFICIENT_EVIDENCE: 130,
}

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
        "non-living), then use it. Keep the chain of reasoning short."
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
        "linear equations, mensuration and data handling."
    ),
    (8, "science"): (
        "The student is in Class 8, using NCERT Curiosity. Multi-step explanations "
        "are fine for force, friction, chemical effects of current, cell structure "
        "or conservation, if each step is named."
    ),
    (9, "mathematics"): (
        "The student is in Class 9, using NCERT Ganita Manjari. They can follow a "
        "structured argument: number systems, polynomials, Euclid, coordinate "
        "geometry and linear equations in two variables."
    ),
    (9, "science"): (
        "The student is in Class 9, using NCERT Exploration. They can follow a "
        "structured argument from motion, force, atoms and molecules, tissues or "
        "the fundamental unit of life."
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
        "limits and derivatives (introductory) is appropriate."
    ),
    (11, "physics"): (
        "The student is in Class 11, using NCERT Physics Parts I and II. Formal "
        "language for units, motion in a plane, laws of motion, work-energy, "
        "gravitation, thermodynamics, waves and oscillations is appropriate. "
        "Show how quantities and principles are connected."
    ),
    (11, "chemistry"): (
        "The student is in Class 11, using NCERT Chemistry Parts I and II. Formal "
        "language for some basic concepts of chemistry, structure of atom, "
        "periodicity, chemical bonding, states of matter, thermodynamics, "
        "equilibrium, redox and organic basics is appropriate."
    ),
    (11, "biology"): (
        "The student is in Class 11, using NCERT Biology. Formal language for the "
        "living world, biological classification, plant and animal kingdom, "
        "morphology, cell, biomolecules, cell cycle and plant physiology is "
        "appropriate. Organise processes and classifications clearly."
    ),
    (12, "mathematics"): (
        "The student is in Class 12, using NCERT Mathematics Parts I and II. They "
        "can follow relations and functions, inverse trigonometry, matrices, "
        "determinants, continuity, differentiability, applications of derivatives, "
        "integrals, differential equations, vectors and probability. Use the "
        "book's wording (for example 'derivative', 'definite integral')."
    ),
    (12, "physics"): (
        "The student is in Class 12, using NCERT Physics Parts I and II. They can "
        "follow electrostatics, current electricity, magnetism, electromagnetic "
        "induction, AC, EM waves, ray and wave optics, dual nature, atoms, nuclei "
        "and semiconductors. Use the book's wording and standard symbols."
    ),
    (12, "chemistry"): (
        "The student is in Class 12, using NCERT Chemistry Parts I and II. They "
        "can follow solutions, electrochemistry, chemical kinetics, d- and f-block, "
        "coordination compounds, haloalkanes, alcohols, aldehydes, amines and "
        "biomolecules. Use NCERT names for reactions and compounds."
    ),
    (12, "biology"): (
        "The student is in Class 12, using NCERT Biology. They can follow "
        "reproduction, genetics, evolution, human health, biotechnology, organisms "
        "and populations, ecosystem and biodiversity. Use the book's terms and "
        "organise multi-stage processes clearly."
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
        if state is TutorState.CONFIRM_ANSWER:
            lines.extend(_CONFIRM_APPLY)
        if state is TutorState.GIVE_HINT:
            lines.extend(_GUIDED_RULES)
        guidance = scope_guidance(scope.grade, scope.subject)
        if guidance:
            lines.append(guidance)
        lines.append(
            f"Keep the complete response within {_STATE_WORD_LIMITS[state]} words."
        )
        lines.append(
            f"Current tutoring move: {state.value}. "
            f"{_state_instruction(state, scope.subject)}"
        )
        return "\n".join(f"- {line}" for line in lines)

    def format_evidence(self, evidence: Sequence[RetrievedChunk]) -> str:
        """Render evidence with internal provenance labels. The student never sees this."""
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
        subject: str | None = None,
        insufficient_reasons: Sequence[str] = (),
        attached_figures: Sequence[dict[str, Any]] = (),
    ) -> str:
        parts = [
            f"STUDENT QUESTION\n{question}",
            "The [E1], [E2], … blocks below are hidden from the student. "
            "Do not mention those labels or say according to the evidence.",
        ]

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
                "do not invent any other diagram). These labels are for you only; "
                "do not say Figure 1 or quote image_id to the student.\n"
                + "\n".join(lines)
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
                "for this question. Do not answer from general knowledge. "
                "Do not mention [E1] labels or that you inspected hidden evidence. "
                "Output JSON only: "
                '{"decline": "...", "nearby_coverage": "..." or null, '
                '"next_step": "..."}.'
            )
        else:
            parts.append(f"CURRICULUM EVIDENCE\n{self.format_evidence(evidence)}")
            if state is TutorState.CONFIRM_ANSWER:
                parts.append(
                    "CONFIRMATION CHECK (private; output JSON only)\n"
                    "1. Interpret the student's proposed answer.\n"
                    "2. Work out the expected answer from the retrieved rule.\n"
                    "3. Compare their meanings, allowing equivalent wording, "
                    "notation and simplification.\n"
                    "4. Lock the verdict, then emit only the JSON object.\n"
                    "If correct: {\"verdict\":\"correct\",\"brief_reason\":\"...\","
                    "\"mistake_groups\":[]}. "
                    "If incorrect: include mistake_groups with mistake and hint; "
                    "group repeated instances of the same mistake; do not reveal "
                    "the corrected final answer. "
                    "Do not write Correct. or Not quite. inside JSON fields."
                )
            elif state is TutorState.GIVE_HINT:
                parts.append(
                    "HINT JSON (private; output JSON only)\n"
                    '{"hint": "...", "guiding_question": "...?"}'
                )
            elif state is TutorState.EXPLAIN_CONCEPT:
                parts.append(
                    "EXPLAIN JSON (private; output JSON only)\n"
                    '{"explanation": "...", "formula_or_rule": "..." or null, '
                    '"worked_example": {"problem": "...", "steps": ["..."], '
                    '"answer": "..."} or null}'
                )
            parts.append(
                "Ground your reply in the hidden curriculum material. "
                "Never name [E1], E2, or 'the evidence' to the student."
            )
        parts.append(
            "FINAL RESPONSE REQUIREMENTS\n"
            f"{_state_instruction(state, subject)} "
            f"Keep the complete response within {_STATE_WORD_LIMITS[state]} words."
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
                subject=scope.subject,
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
