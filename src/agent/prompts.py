"""Prompts and JSON output schemas for the agent's LLM steps."""

from __future__ import annotations

from src.retrieval.results import RetrievalResult

GRADE_SCHEMA_NAME = "grade_context"
GRADE_SCHEMA = {
    "type": "object",
    "properties": {
        "sub_parts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "part": {"type": "string"},
                    "supported": {"type": "boolean"},
                    "evidence_chunk_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["part", "supported", "evidence_chunk_ids"],
                "additionalProperties": False,
            },
        },
        "decision": {"type": "string", "enum": ["sufficient", "insufficient"]},
        "reason": {"type": "string"},
    },
    "required": ["sub_parts", "decision", "reason"],
    "additionalProperties": False,
}

GRADE_SYSTEM = """\
You judge whether excerpts retrieved from SEC filings (10-K / 40-F) contain \
enough information to answer a user's question.

Step 1 - Decompose. List the distinct sub-parts the question requires an \
answer to. A sub-part is one piece of information the answer must contain:
- each company named or compared (in a comparison, each side is its own \
sub-part),
- each fiscal year or filing the question names (e.g. "in fiscal years 2021 \
and 2022" is two sub-parts, one per year),
- each metric, figure, or fact asked for, for each company and fiscal year it \
applies to,
- for questions asking for several items and an attribute of each (e.g. \
"which segments grew, and by how much"), the items themselves AND the \
requested attribute for each item you identify.
A single-fact question has exactly one sub-part. Do not invent sub-parts the \
question does not ask for (e.g. drivers, context, or explanations).

Step 2 - Check coverage. For each sub-part, decide whether the excerpts \
contain the evidence needed for it, and list the supporting excerpt ids in \
"evidence_chunk_ids".
- Evidence counts if the excerpts contain the inputs needed, even when the \
final value must be computed from them. A sum, difference, percentage change, \
or comparison of figures that are present in the excerpts is supported; the \
computed result does not need to appear verbatim.
- When the question asks for an attribute of each of several items (e.g. "the \
cost of each project"), one figure cannot satisfy "each": if the question \
implies several items each needing a figure and the excerpts give only one, \
that sub-part is not supported.
- Evidence does not count if it is for the wrong company or fiscal year, only \
loosely related, or if the sub-part could only be answered by stating that \
the excerpts don't mention it.

Step 3 - Decide. "sufficient" if and only if every sub-part is supported. \
Otherwise "insufficient".

In "reason", state which sub-parts are supported and, if insufficient, \
exactly which are missing (company, metric, fiscal year). The reason is used \
to rewrite the search query, so name the missing information specifically."""

REFORMULATE_SCHEMA_NAME = "reformulate_query"
REFORMULATE_SCHEMA = {
    "type": "object",
    "properties": {"query": {"type": "string"}},
    "required": ["query"],
    "additionalProperties": False,
}

REFORMULATE_SYSTEM = """\
You write follow-up search queries for a hybrid (semantic + keyword) search \
engine over SEC annual filings (10-K / 40-F).

A previous search for the user's question left some required information \
missing. Excerpts covering the parts already found are kept, so do NOT \
search for those again. Write one new query aimed only at the missing \
information:
- Name the specific company, metric or fact, and fiscal year that is missing \
(e.g. "Acme Corp 2022 long-term debt maturities schedule", "Acme Corp \
warehouse divestiture sale proceeds").
- Use the vocabulary filings actually use (line-item names, "Risk Factors", \
"Management's Discussion and Analysis", "purchase price", "capital \
expenditures").
- Do not restate the whole original question and do not repeat the previous \
query. Return only the query."""

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
    question: str, previous_query: str, grade_reason: str, missing_parts: list[str]
) -> str:
    missing = "\n".join(f"- {part}" for part in missing_parts) or "- (not itemized)"
    return (
        f"Question: {question}\n"
        f"Previous query: {previous_query}\n"
        f"Missing information:\n{missing}\n"
        f"Grader's explanation: {grade_reason}"
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
