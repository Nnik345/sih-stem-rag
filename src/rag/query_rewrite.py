"""Query rewriter: Qwen3-VL-2B-Instruct, loaded only for rewrite then unloaded.

The rewriter turns a student's wording into an NCERT-like retrieval query. It
never answers the question and never infers grade or subject from the text.
Caller-supplied grade/subject may be used only to pick curriculum terminology
(for example Class 12 mathematics "derivative" rather than colloquial
"differentiation").

The 8B tutor lives in :mod:`rag.generator`. This module is a separate small VL
loader so the two checkpoints cannot be confused.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import ModelConfig
from .evidence import has_maths_instance_token, strip_maths_instance_text
from .logging_utils import get_logger
from .model_memory import cuda_max_memory_map, empty_cuda_cache, tighter_max_memory_map

LOGGER = get_logger(__name__)

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
_ALLOWED_INTENTS = frozenset({"explain", "verify", "practice", "other"})
_ALLOWED_INPUT_KINDS = frozenset({"math_problem", "diagram", "other"})
_DIFF_CUE_RE = re.compile(
    r"(?:d\s*/\s*d[xy]|dy\s*/\s*dx|differenti\w*|derivative)",
    re.IGNORECASE,
)
_COMPOSITE_POWER_RE = re.compile(r"\)\s*\^\s*\d+")
_POLYNOMIAL_POWER_RULE_QUERY = (
    "algebra of derivative of functions derivative of x to the power n"
)
_CHAIN_RULE_QUERY = "derivative of composite functions"

# Textbook figures are attached only when the student asked for a visual, not
# because a retrieved page happens to contain an image.
_FIGURE_NEED_RE = re.compile(
    r"(?:"
    r"\b(?:diagrams?|illustrations?|sketches?)\b|"
    r"\b(?:labell?ed)\b|"
    r"\bdrawings?\s+of\b|"
    r"\bdraw\s+(?:me\s+)?(?:a|an|the)\b|"
    r"\bfigures?\s+of\b|"
    r"\bstructure\s+of\b|"
    r"\banatomy\s+of\b|"
    r"\bflowcharts?\b|"
    r"\bcircuit\s+diagram\b|"
    r"\bpictures?\s+of\b|"
    r"\blooks?\s+like\b"
    r")",
    re.IGNORECASE,
)

REWRITE_SYSTEM_PROMPT = """You rewrite a student's STEM question into a short retrieval query for official NCERT English textbooks.

Output JSON only, no markdown, no extra text:
{"retrieval_query": "...", "intent": "explain|verify|practice|other"}

retrieval_query:
- One or two sentences in NCERT-like English, using the textbook's likely terms for the given class and subject.
- Name the curriculum procedure or rule the books would use, not the student's specific polynomial, numbers, or proposed answer. Do not solve the problem. Do not include a computed result, yes/no, or the student's proposed answer.
- Do not put the instance expression into the retrieval query (for example do not search for "x squared plus 3x"). Search for the rule (power rule, sum rule, chain rule, multiplication, and so on).
- Keep the student's meaning. Expand abbreviations and map colloquial wording to curriculum wording (for example Class 12 mathematics: "differentiation" -> "derivative").
- Do not guess or change the class or subject. Use only the class and subject supplied in the user message. Lower-class search is not your job.
- Do not invent chapter numbers, page numbers, or book titles that were not given.
- Mention a diagram in retrieval_query only when the student asked to see one (or the photo is a diagram). Ordinary explanations do not need a figure.

intent (pick one):
- explain: the student wants the idea, definition, or method.
- verify: the student is checking their own work. Typical shapes: "Is the derivative/differentiation of EXPR = RESULT?", "Is my answer ...?", "Did I get this right?". EXPR is the problem; RESULT is their attempt. Do not apply the operation to RESULT. Do not treat this as "operation of EXPR equals operation of RESULT".
- practice: the student wants an exercise or more problems.
- other: none of the above.

Examples:
Student question: Is 12 times 3 equal to 36?
Class 5, mathematics
{"retrieval_query": "multiplication of whole numbers", "intent": "verify"}

Student question: Is the differentiation of x^2 + 3x = 2x + 3?
Class 11, mathematics
{"retrieval_query": "derivative of a polynomial using the power rule and the sum rule", "intent": "verify"}

Student question: Is the derivative of (2x+1)^3 equal to 6(2x+1)^2?
Class 12, mathematics
{"retrieval_query": "chain rule for differentiation of composite functions", "intent": "verify"}
"""

IMAGE_REWRITE_SYSTEM_PROMPT = REWRITE_SYSTEM_PROMPT + """
A student photo is attached. Classify it and include these extra JSON fields:
{"retrieval_query": "...", "intent": "...", "input_kind": "math_problem|diagram|other", "transcribed_question": "..."}

input_kind:
- math_problem: handwritten or printed exercise, possibly with working. Transcribe the full problem and any attempt into transcribed_question (NCERT-like English). retrieval_query still names the curriculum RULE or topic, not the instance expression.
- diagram: labelled figure, apparatus, map, or sketch. retrieval_query is a short curriculum-term description (for example "plant cell labelled diagram"). transcribed_question is a brief description of what is shown.
- other: neither of the above. Describe it; retrieval_query is that description.

transcribed_question is what the tutor should see as the student question. Do not solve the problem. Do not invent labels that are not visible.
If the student also typed text, keep that meaning and merge it with what the photo shows.
"""


def specialize_maths_retrieval_query(
    retrieval_query: str,
    *,
    original: str = "",
    transcribed: str = "",
) -> str:
    """Search the curriculum rule, not the student's specific polynomial.

    The 2B rewriter is asked to do this, but a photo of working often comes back
    as the equation itself. Searching that instance will not hit NCERT, even
    when the power rule is on the page.
    """
    text = (retrieval_query or "").strip()
    if not text:
        text = (transcribed or original or "").strip()
    blob = f"{text} {original or ''} {transcribed or ''}"
    # NCERT never says "power rule". If the rewriter keeps that phrase (or
    # "quadratic polynomial"), search hits Class 10 polynomials and the
    # reranker scores the real Limits-and-Derivatives pages below 0.
    if _DIFF_CUE_RE.search(blob):
        if _COMPOSITE_POWER_RE.search(blob):
            return _CHAIN_RULE_QUERY
        return _POLYNOMIAL_POWER_RULE_QUERY
    stripped = strip_maths_instance_text(text)
    if stripped and not has_maths_instance_token(text):
        return text
    if stripped:
        return stripped
    return text


def question_needs_textbook_figure(
    *,
    query: str,
    transcribed_question: str = "",
    input_kind: str = "other",
) -> bool:
    """True only when a textbook diagram is needed for the answer.

    A photo classified as a diagram counts. So does wording like "diagram of a
    cell". Explanations, calculations, and check-my-work do not, even if the
    matching textbook page has a figure.
    """
    if (input_kind or "").strip().lower() == "diagram":
        return True
    blob = f"{query or ''} {transcribed_question or ''}"
    return bool(_FIGURE_NEED_RE.search(blob))


class RewriteError(RuntimeError):
    """Raised when the rewriter checkpoint is missing or generation fails."""


@dataclass(frozen=True)
class QueryRewriteResult:
    original_query: str
    retrieval_query: str
    intent: str
    fallback: bool
    reason: str = ""
    input_kind: str = "other"
    transcribed_question: str = ""


def build_rewrite_user_prompt(
    question: str,
    *,
    grade: int | None = None,
    subject: str | None = None,
    has_image: bool = False,
) -> str:
    text = (question or "").strip()
    if has_image:
        lines = [
            f"Student text (may be empty): {text or '(none)'}",
            "A photo is attached. Classify it (math_problem, diagram, or other) and rewrite for NCERT retrieval.",
        ]
    else:
        lines = [f"Student question: {text}"]
    if grade is not None:
        lines.append(f"Class (given by the application, do not infer): {grade}")
    if subject:
        lines.append(f"Subject (given by the application, do not infer): {subject}")
    lines.append("Rewrite for NCERT retrieval. JSON only.")
    return "\n".join(lines)


def parse_rewrite_output(raw: str, original_query: str) -> QueryRewriteResult:
    """Parse model JSON. On any failure, fall back to the original question."""
    text = (raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    match = _JSON_OBJECT_RE.search(text)
    if not match:
        return QueryRewriteResult(
            original_query=original_query,
            retrieval_query=original_query,
            intent="other",
            fallback=True,
            reason="rewriter output was not JSON",
        )
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return QueryRewriteResult(
            original_query=original_query,
            retrieval_query=original_query,
            intent="other",
            fallback=True,
            reason="rewriter JSON could not be parsed",
        )
    transcribed_raw = payload.get("transcribed_question")
    transcribed = (
        " ".join(transcribed_raw.split())
        if isinstance(transcribed_raw, str)
        else ""
    )
    kind_raw = payload.get("input_kind")
    if isinstance(kind_raw, str) and kind_raw.strip().lower() in _ALLOWED_INPUT_KINDS:
        input_kind = kind_raw.strip().lower()
    else:
        input_kind = "other"
    fallback_text = (original_query or "").strip() or transcribed
    retrieval = payload.get("retrieval_query")
    if not isinstance(retrieval, str) or not retrieval.strip():
        return QueryRewriteResult(
            original_query=original_query,
            retrieval_query=fallback_text or original_query,
            intent="other",
            fallback=True,
            reason="retrieval_query missing",
            input_kind=input_kind,
            transcribed_question=transcribed,
        )
    intent = payload.get("intent")
    if not isinstance(intent, str) or intent.strip().lower() not in _ALLOWED_INTENTS:
        intent = "other"
    else:
        intent = intent.strip().lower()
    rewritten = " ".join(retrieval.split())
    return QueryRewriteResult(
        original_query=original_query,
        retrieval_query=rewritten,
        intent=intent,
        fallback=False,
        input_kind=input_kind,
        transcribed_question=transcribed,
    )


class QueryRewriter:
    """Lazy-loading wrapper around the local Qwen3-VL-2B-Instruct checkpoint."""

    def __init__(self, model_path: Path, *, vram_reserve_gib: float = 0.5) -> None:
        self.model_path = Path(model_path)
        self.vram_reserve_gib = vram_reserve_gib
        self._model = None
        self._processor = None

    @classmethod
    def from_config(cls, config: ModelConfig) -> "QueryRewriter":
        return cls(config.rewriter_model_path)

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self._model is not None:
            return
        if not self.model_path.is_dir():
            raise RewriteError(
                f"Rewriter model directory not found: {self.model_path}. "
                "Download it with: python scripts/download_retrieval_models.py "
                "or python scripts/download_qwen_models.py"
            )
        try:
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        except ImportError as exc:
            raise RewriteError(
                f"Transformers does not expose Qwen3VLForConditionalGeneration "
                f"({exc}). Qwen3-VL needs the git build: "
                f"pip install git+https://github.com/huggingface/transformers"
            ) from exc

        LOGGER.info("Loading Qwen3-VL-2B rewriter from %s", self.model_path)
        try:
            empty_cuda_cache()
            self._processor = AutoProcessor.from_pretrained(str(self.model_path))
            max_memory = self._max_memory_map()
            load_kwargs: dict[str, Any] = {
                "dtype": "auto",
                "device_map": "auto",
                "low_cpu_mem_usage": True,
            }
            if max_memory is not None:
                load_kwargs["max_memory"] = max_memory
            try:
                model = Qwen3VLForConditionalGeneration.from_pretrained(
                    str(self.model_path), **load_kwargs
                )
            except Exception as capped_error:
                if max_memory is None:
                    raise
                LOGGER.warning(
                    "Capped rewriter load failed (%s); retrying with more CPU offload",
                    capped_error,
                )
                load_kwargs["max_memory"] = tighter_max_memory_map(max_memory)
                empty_cuda_cache()
                model = Qwen3VLForConditionalGeneration.from_pretrained(
                    str(self.model_path), **load_kwargs
                )
            model.eval()
        except Exception as exc:
            self._processor = None
            raise RewriteError(
                f"Could not load the rewriter from {self.model_path}: {exc}"
            ) from exc
        self._model = model
        LOGGER.info("Qwen3-VL-2B rewriter ready")

    def _max_memory_map(self) -> dict[Any, str] | None:
        # 2B is small; keep a light reserve so 8 GiB cards still run it on GPU
        # while 32 GiB cards are unaffected.
        return cuda_max_memory_map(
            vram_reserve_gib=self.vram_reserve_gib,
            cpu_ram_fraction=0.50,
            headroom_fraction=0.15,
            headroom_cap_gib=4.0,
        )

    def unload(self) -> None:
        if self._model is None and self._processor is None:
            return
        LOGGER.info("Releasing Qwen3-VL-2B rewriter")
        self._model = None
        self._processor = None
        empty_cuda_cache()

    def rewrite(
        self,
        question: str,
        *,
        grade: int | None = None,
        subject: str | None = None,
        image_path: str | Path | None = None,
    ) -> QueryRewriteResult:
        """Rewrite ``question`` for retrieval. Falls back to the original text."""
        original = (question or "").strip()
        image = Path(image_path) if image_path else None
        has_image = image is not None and image.is_file()
        if not original and not has_image:
            return QueryRewriteResult(
                original_query=question,
                retrieval_query=question,
                intent="other",
                fallback=True,
                reason="empty question",
            )
        try:
            self.load()
        except RewriteError as exc:
            LOGGER.warning("Rewriter unavailable (%s); using the original query", exc)
            return QueryRewriteResult(
                original_query=original,
                retrieval_query=original or question,
                intent="other",
                fallback=True,
                reason=str(exc),
            )
        assert self._processor is not None and self._model is not None
        system = IMAGE_REWRITE_SYSTEM_PROMPT if has_image else REWRITE_SYSTEM_PROMPT
        user_content: list[dict[str, Any]] = []
        if has_image:
            user_content.append({"type": "image", "image": str(image)})
        user_content.append(
            {
                "type": "text",
                "text": build_rewrite_user_prompt(
                    original, grade=grade, subject=subject, has_image=has_image
                ),
            }
        )
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system}]},
            {"role": "user", "content": user_content},
        ]
        max_new_tokens = 256 if has_image else 128
        try:
            import torch

            inputs = self._processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self._model.device)
            with torch.inference_mode():
                generated = self._model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )
            prompt_len = inputs["input_ids"].shape[-1]
            new_tokens = generated[0, prompt_len:]
            raw = self._processor.tokenizer.decode(
                new_tokens, skip_special_tokens=True
            )
        except Exception as exc:
            LOGGER.warning("Rewriter generation failed (%s); using the original query", exc)
            return QueryRewriteResult(
                original_query=original,
                retrieval_query=original or question,
                intent="other",
                fallback=True,
                reason=str(exc),
            )
        result = parse_rewrite_output(raw, original)
        LOGGER.info(
            "Query rewrite fallback=%s intent=%s kind=%s %r -> %r",
            result.fallback,
            result.intent,
            result.input_kind,
            original,
            result.retrieval_query,
        )
        return result
