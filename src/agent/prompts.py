"""Prompts and JSON output schemas for the agent's LLM steps."""

from __future__ import annotations

from src.retrieval.results import RetrievalResult

GRADE_SCHEMA_NAME = "grade_context"
GRADE_SCHEMA = {
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": ["sufficient", "insufficient"]},
        "reason": {"type": "string"},
    },
    "required": ["decision", "reason"],
    "additionalProperties": False,
}

GRADE_SYSTEM = """\
You judge whether excerpts retrieved from SEC filings (10-K / 40-F) contain \
enough information to answer a user's question.

Answer "sufficient" only if the excerpts directly contain the facts needed to \
answer the question for the company and fiscal year it asks about. Answer \
"insufficient" if key facts are missing, the excerpts cover the wrong company \
or year, or they are only loosely related.

In "reason", state concisely what is present and, if insufficient, exactly \
what is missing. The reason is used to rewrite the search query."""

REFORMULATE_SCHEMA_NAME = "reformulate_query"
REFORMULATE_SCHEMA = {
    "type": "object",
    "properties": {"query": {"type": "string"}},
    "required": ["query"],
    "additionalProperties": False,
}

REFORMULATE_SYSTEM = """\
You rewrite search queries for a hybrid (semantic + keyword) search engine over \
SEC annual filings (10-K / 40-F).

Given the user's question, the query that was just tried, and why its results \
were judged insufficient, write one new query more likely to retrieve the \
missing information. Use the vocabulary filings actually use (e.g. line-item \
names, "Risk Factors", "Management's Discussion and Analysis"), keep the \
company name and fiscal year if the question has them, and do not simply \
repeat the previous query. Return only the query."""

GENERATE_SCHEMA_NAME = "grounded_answer"
GENERATE_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "answerable": {"type": "boolean"},
        "cited_chunk_ids": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["answer", "answerable", "cited_chunk_ids"],
    "additionalProperties": False,
}

GENERATE_SYSTEM = """\
You answer questions about companies using ONLY the SEC filing excerpts \
provided. You have no other knowledge of these companies.

Rules:
- Every factual claim must be supported by the excerpts. Do not use outside \
knowledge, do not guess, and do not fill gaps with plausible-sounding figures.
- Cite the excerpt each claim comes from inline, using its id in square \
brackets, e.g. [CCJ_2023_40-F_0436]. List every id you cited in \
"cited_chunk_ids". Only use ids that appear in the excerpts.
- If the excerpts do not contain the answer, set "answerable" to false and say \
plainly that the provided filings do not contain this information. Do not \
attempt an answer. You may briefly note what the excerpts do cover instead.
- The absence of something from the excerpts is not evidence about it. If your \
answer would rest on the filings not mentioning something (e.g. "the excerpts \
don't indicate the company does X"), set "answerable" to false.
- If the excerpts answer only part of the question, answer that part, state \
clearly what is not covered, and set "answerable" to true.
- Be precise with numbers, units, periods, and which company/fiscal year a \
figure belongs to."""

UNANSWERABLE_NO_CONTEXT = (
    "The retrieved filings do not contain information to answer this question."
)


def format_chunks(chunks: list[RetrievalResult]) -> str:
    blocks = []
    for chunk in chunks:
        filing = chunk.filing
        header = (
            f"[{chunk.id}] {filing.company} ({filing.ticker}) {filing.form_type}, "
            f"fiscal year {filing.fiscal_year} - {filing.source_url}"
        )
        blocks.append(f"{header}\n{chunk.text}")
    return "\n\n---\n\n".join(blocks)


def grade_user_prompt(question: str, chunks: list[RetrievalResult]) -> str:
    return f"Question: {question}\n\nExcerpts:\n\n{format_chunks(chunks)}"


def reformulate_user_prompt(
    question: str, previous_query: str, grade_reason: str
) -> str:
    return (
        f"Question: {question}\n"
        f"Previous query: {previous_query}\n"
        f"Why the results were insufficient: {grade_reason}"
    )


def generate_user_prompt(
    question: str, chunks: list[RetrievalResult], grade_reason: str | None
) -> str:
    prompt = f"Question: {question}\n\nExcerpts:\n\n{format_chunks(chunks)}"
    if grade_reason is not None:
        # Retries were exhausted without a "sufficient" grade; surface the
        # grader's concern so the model leans toward declining over guessing.
        prompt += (
            "\n\nNote: a reviewer judged these excerpts possibly insufficient: "
            f"{grade_reason}"
        )
    return prompt
