"""LLM-as-judge prompts for answer correctness (reference + rubric) and
retrieval sufficiency. The judge depends only on the `LLMClient` protocol, so
tests inject a scripted fake."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from src.agent.llm import LLMClient
from src.retrieval.results import RetrievalResult

JUDGE_SCHEMA_NAME = "judge_verdict"
JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        # Reason first so the verdict is conditioned on the reasoning.
        "reason": {"type": "string"},
        "passed": {"type": "boolean"},
    },
    "required": ["reason", "passed"],
    "additionalProperties": False,
}

REFERENCE_SYSTEM = """\
You are grading an AI system's answer to a question about SEC financial filings \
against a human-verified reference answer.

Judge FACTUAL CORRECTNESS ON WHAT THE QUESTION ASKS:
- PASS if the answer states the key facts the question asks for and they agree \
with the reference (numbers, entities, periods, units).
- Be tolerant of terseness: an answer that gives exactly what was asked without \
the reference's extra context (drivers, background, dates not asked for) still \
passes.
- Be tolerant of formatting: "$23,568 million", "$23.6 billion" and "23,568 \
(in millions)" are equivalent; reasonable rounding is fine.
- Additional detail beyond the reference is fine, but it is held to the same \
standard: an incorrect extra claim fails the answer.
- Check every DERIVED figure the answer states (differences, sums, ratios, \
multiples, % changes) even if the reference doesn't give one: it must be \
arithmetically valid from the underlying figures. Arithmetic or a \
"higher/lower by X" comparison across mismatched currencies or units without \
an explicit conversion is invalid and fails the answer.
- FAIL if a requested fact is missing, a stated fact contradicts the reference \
(wrong number, wrong year, wrong company, wrong currency), a derived figure is \
invalid, or the answer declines/says the information is unavailable when the \
reference answers it.
- Do NOT require exact wording.

Give a brief reason (1-3 sentences) naming the decisive fact, then the verdict."""

RUBRIC_SYSTEM = """\
You are grading an AI system's answer to a question about SEC financial filings \
against a grading rubric written by a human who verified the source filings.

Apply the rubric's pass_condition and fail_condition literally. Use \
required_points, credit_points and scoping_and_honesty to interpret them. \
Figures need only match approximately as the rubric states. Do NOT require \
exact wording. If any fail_condition is met, the answer fails.

credit_points can earn credit but can NEVER cause a fail: omitting a credit \
point is not a failure. Only a missing required_point or a triggered \
fail_condition can fail an answer.

Give a brief reason (1-3 sentences) naming the decisive point(s), then the verdict."""

RETRIEVAL_SYSTEM = """\
You are evaluating the retrieval step of a question-answering system over SEC \
financial filings. You are given a question, a human-verified reference (an \
answer, or a list of key facts a correct answer must contain), and the text \
chunks the system retrieved.

Decide whether the retrieved chunks, taken together, CONTAIN the facts needed \
to produce the reference's key facts for what the question asks. Judge \
by content, not by which filing or chunk it came from: a different filing that \
states the same correct figure counts. Supplementary context in the reference \
that the question didn't ask for is not required.

PASS if every key fact needed is present in the chunks. FAIL if any needed fact \
is absent or only present for the wrong period/company.

Give a brief reason (1-3 sentences) naming any missing fact, then the verdict."""


@dataclass(frozen=True)
class Verdict:
    passed: bool
    reason: str


def _parse_verdict(result: dict[str, Any]) -> Verdict:
    passed = result.get("passed")
    reason = result.get("reason")
    if not isinstance(passed, bool) or not isinstance(reason, str):
        raise TypeError(f"Malformed judge reply: {result!r}")
    return Verdict(passed=passed, reason=reason.strip())


def reference_user_prompt(question: str, reference: str, answer: str) -> str:
    return (
        f"QUESTION:\n{question}\n\n"
        f"REFERENCE ANSWER (human-verified):\n{reference}\n\n"
        f"SYSTEM ANSWER TO GRADE:\n{answer}"
    )


def rubric_user_prompt(question: str, rubric: dict[str, Any], answer: str) -> str:
    return (
        f"QUESTION:\n{question}\n\n"
        f"RUBRIC:\n{json.dumps(rubric, indent=2)}\n\n"
        f"SYSTEM ANSWER TO GRADE:\n{answer}"
    )


def retrieval_user_prompt(
    question: str, reference: str, chunks: Sequence[RetrievalResult]
) -> str:
    rendered = "\n\n".join(
        f"[{chunk.id}] ({chunk.filing.company}, FY{chunk.filing.fiscal_year} "
        f"{chunk.filing.form_type})\n{chunk.text}"
        for chunk in chunks
    )
    return (
        f"QUESTION:\n{question}\n\n"
        f"REFERENCE (human-verified):\n{reference}\n\n"
        f"RETRIEVED CHUNKS:\n{rendered or '(none)'}"
    )


class LLMJudge:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def _ask(self, system: str, user: str) -> Verdict:
        return _parse_verdict(
            self.llm.complete_json(
                system=system,
                user=user,
                schema_name=JUDGE_SCHEMA_NAME,
                schema=JUDGE_SCHEMA,
            )
        )

    def judge_reference(self, question: str, reference: str, answer: str) -> Verdict:
        return self._ask(
            REFERENCE_SYSTEM, reference_user_prompt(question, reference, answer)
        )

    def judge_rubric(
        self, question: str, rubric: dict[str, Any], answer: str
    ) -> Verdict:
        return self._ask(RUBRIC_SYSTEM, rubric_user_prompt(question, rubric, answer))

    def judge_retrieval(
        self, question: str, reference: str, chunks: Sequence[RetrievalResult]
    ) -> Verdict:
        return self._ask(
            RETRIEVAL_SYSTEM, retrieval_user_prompt(question, reference, chunks)
        )
