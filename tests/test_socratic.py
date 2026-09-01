"""Per-(grade, subject) Socratic prompts and tutoring states."""

from rag.curriculum_catalog import subjects_for_grade
from rag.schemas import EvidenceDecision, RetrievalFilter
from rag.socratic import SocraticController, TutorState, scope_guidance


def test_every_legal_combo_has_personalised_guidance():
    for grade in range(1, 13):
        for subject in subjects_for_grade(grade):
            text = scope_guidance(grade, subject)
            assert text
            assert f"Class {grade}" in text
            assert "NCERT" in text


def test_class_12_physics_and_maths_prompts_differ():
    physics = scope_guidance(12, "physics")
    maths = scope_guidance(12, "mathematics")
    science6 = scope_guidance(6, "science")
    assert "Physics" in physics
    assert "electrostatics" in physics.lower() or "Electrostatics" in physics
    assert "Mathematics Parts" in maths or "calculus" in maths.lower() or "derivative" in maths
    assert physics != maths
    assert "Curiosity" in science6
    assert science6 != physics


def test_system_prompt_uses_combo_guidance():
    controller = SocraticController()
    physics = controller.system_prompt(
        TutorState.GIVE_HINT, RetrievalFilter(grade=12, subject="physics")
    )
    maths = controller.system_prompt(
        TutorState.GIVE_HINT, RetrievalFilter(grade=12, subject="mathematics")
    )
    assert "Physics Parts" in physics
    assert "Mathematics Parts" in maths
    assert physics != maths


def test_give_hint_maths_vs_science_wording():
    controller = SocraticController()
    maths = controller.system_prompt(
        TutorState.GIVE_HINT, RetrievalFilter(grade=8, subject="mathematics")
    )
    science = controller.system_prompt(
        TutorState.GIVE_HINT, RetrievalFilter(grade=8, subject="science")
    )
    physics = controller.system_prompt(
        TutorState.GIVE_HINT, RetrievalFilter(grade=11, subject="physics")
    )
    assert "working step" in maths
    assert "final answer" in maths
    assert "relevant concept" in science
    assert "guiding question" in science
    assert "relevant concept" in physics
    assert "90 words" in maths
    assert "Ask exactly one guiding question" in science
    assert '"hint"' in maths


def test_explain_concept_maths_vs_science_and_drops_guided_rules():
    controller = SocraticController()
    maths = controller.system_prompt(
        TutorState.EXPLAIN_CONCEPT, RetrievalFilter(grade=10, subject="mathematics")
    )
    science = controller.system_prompt(
        TutorState.EXPLAIN_CONCEPT, RetrievalFilter(grade=10, subject="science")
    )
    biology = controller.system_prompt(
        TutorState.EXPLAIN_CONCEPT, RetrievalFilter(grade=12, subject="biology")
    )
    assert "fully worked example" in maths
    assert "Explain the requested concept" in science
    assert "Explain the requested concept" in biology
    for prompt in (maths, science, biology):
        assert "180 words" not in prompt
        assert "Ask exactly one guiding question" not in prompt
        assert "worked_example" in prompt


def test_confirm_answer_marks_attempt_and_does_not_quiz():
    controller = SocraticController()
    maths = controller.system_prompt(
        TutorState.CONFIRM_ANSWER, RetrievalFilter(grade=9, subject="mathematics")
    )
    science = controller.system_prompt(
        TutorState.CONFIRM_ANSWER, RetrievalFilter(grade=9, subject="science")
    )
    assert "Output one JSON object only" in maths
    assert "mistake_groups" in maths
    assert "cannot see CURRICULUM EVIDENCE" in maths
    assert "EXPR" in maths
    assert "Output one JSON object only" in science
    assert "Ask one question at a time" not in maths
    assert "Ask exactly one guiding question" not in maths


def test_confirm_answer_user_prompt_notes_attempt():
    controller = SocraticController()
    prompt = controller.user_prompt(
        "Find 2+2. I got 5.",
        (),
        state=TutorState.CONFIRM_ANSWER,
    )
    assert "output JSON only" in prompt
    assert "mistake_groups" in prompt
    assert "equivalent" in prompt
    hint_prompt = controller.user_prompt(
        "Find 2+2.",
        (),
        state=TutorState.GIVE_HINT,
    )
    assert "attempted answer" not in hint_prompt


def test_select_state_defaults_to_give_hint():
    controller = SocraticController()
    sufficient = EvidenceDecision(sufficient=True)
    insufficient = EvidenceDecision(sufficient=False, reasons=["no overlap"])
    assert controller.select_state(sufficient) is TutorState.GIVE_HINT
    assert (
        controller.select_state(sufficient, requested_state=TutorState.EXPLAIN_CONCEPT)
        is TutorState.EXPLAIN_CONCEPT
    )
    assert (
        controller.select_state(
            insufficient, requested_state=TutorState.EXPLAIN_CONCEPT
        )
        is TutorState.INSUFFICIENT_EVIDENCE
    )


def test_system_prompt_applies_retrieved_rule_and_does_not_leak_on_insufficient():
    controller = SocraticController()
    hint = controller.system_prompt(
        TutorState.GIVE_HINT, RetrievalFilter(grade=12, subject="mathematics")
    )
    confirm = controller.system_prompt(
        TutorState.CONFIRM_ANSWER, RetrievalFilter(grade=12, subject="mathematics")
    )
    insufficient = controller.system_prompt(
        TutorState.INSUFFICIENT_EVIDENCE, RetrievalFilter(grade=12, subject="mathematics")
    )
    for prompt in (hint, confirm):
        assert "apply that rule" in prompt
        assert "earlier class" in prompt
    assert "private check" in confirm
    assert "apply that rule" not in insufficient
    assert "from general knowledge" in insufficient


def test_every_state_hides_evidence_from_the_student():
    controller = SocraticController()
    scope = RetrievalFilter(grade=11, subject="mathematics")
    for state in TutorState:
        system = controller.system_prompt(state, scope)
        assert "cannot see CURRICULUM EVIDENCE" in system
        assert "according to the evidence" in system
        user = controller.user_prompt("What is a derivative?", (), state=state)
        assert "hidden from the student" in user
        assert "according to the evidence" in user
        if state is TutorState.INSUFFICIENT_EVIDENCE:
            assert "Do not mention [E1] labels" in user
        else:
            assert "Never name [E1]" in user
