"""Run every golden question through the agent, score it, and build the
structured results dict (per-question detail + aggregate summary)."""

from __future__ import annotations

import dataclasses
import logging
import statistics
import subprocess
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.agent.graph import DEFAULT_MAX_RETRIES, Backend, answer_question, make_store
from src.agent.llm import LLMClient, OpenAILLMClient, agent_model, grade_model
from src.agent.state import AgentResult, RetrievalMode
from src.eval.cost import question_cost, summarize_costs
from src.eval.golden import (
    DEFAULT_GOLDEN_PATH,
    GoldenItem,
    file_sha256,
    load_golden_set,
)
from src.eval.judge import LLMJudge
from src.eval.scoring import AnswerScore, score_answer, score_retrieval
from src.ingestion.vector_stores.base import VectorStore

logger = logging.getLogger(__name__)

DEFAULT_JUDGE_MODEL = "gpt-5.4"
DEFAULT_OUTPUT_DIR = Path("eval_results")

# Returns fresh (generator, grader) clients per question so usage is per question.
AgentLLMFactory = Callable[[], tuple[LLMClient, LLMClient]]


@dataclass(frozen=True)
class EvalConfig:
    backend: Backend = "pinecone"
    retrieval_mode: RetrievalMode = "hybrid_reranked"
    judge_model: str = DEFAULT_JUDGE_MODEL
    golden_path: Path = DEFAULT_GOLDEN_PATH
    output_dir: Path = DEFAULT_OUTPUT_DIR
    max_retries: int = DEFAULT_MAX_RETRIES
    top_k: int = 5
    question_ids: tuple[str, ...] = field(default_factory=tuple)  # empty = all


def default_agent_llms() -> tuple[LLMClient, LLMClient]:
    return OpenAILLMClient(agent_model()), OpenAILLMClient(grade_model())


def _usage_records(client: LLMClient, start: int = 0) -> list[dict[str, Any]]:
    """Token records a client accumulated since `start`; fakes without a
    `usage` attribute contribute none."""
    usage = getattr(client, "usage", [])
    return [dataclasses.asdict(record) for record in usage[start:]]


def _usage_count(client: LLMClient) -> int:
    return len(getattr(client, "usage", []))


def rerank_searches(result: AgentResult) -> int:
    return sum(
        1
        for event in result["trace"]
        if event["node"] == "retrieve"
        and event["detail"].get("mode") == "hybrid_reranked"
    )


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def evaluate_question(
    item: GoldenItem,
    config: EvalConfig,
    *,
    store: VectorStore,
    judge: LLMJudge,
    agent_llms: AgentLLMFactory,
) -> dict[str, Any]:
    generator, grader = agent_llms()
    judge_start = _usage_count(judge.llm)
    record: dict[str, Any] = {
        "id": item.id,
        "question": item.question,
        "question_type": item.question_type,
        "eval_type": item.eval_type,
        "scoring_mode": item.scoring_mode,
        "golden_answer": item.draft_answer,
        "agent": None,
        "answer_score": None,
        "retrieval_score": None,
        "error": None,
    }

    result: AgentResult | None = None
    started = time.perf_counter()
    try:
        result = answer_question(
            item.question,
            config.retrieval_mode,
            store=store,
            llm=generator,
            grader_llm=grader,
            max_retries=config.max_retries,
            top_k=config.top_k,
        )
    except Exception as exc:  # one bad question must not sink the run
        logger.exception("Agent failed on %s", item.id)
        record["error"] = f"agent: {type(exc).__name__}: {exc}"
    latency = time.perf_counter() - started

    if result is not None:
        record["agent"] = {
            "answer": result["answer"],
            "answerable": result["answerable"],
            "retry_count": result["retry_count"],
            "citations": result["citations"],
            "retrieved_chunk_ids": [chunk.id for chunk in result["retrieved_chunks"]],
            "latency_s": round(latency, 3),
        }
        try:
            answer_score = score_answer(
                item, result["answer"], result["answerable"], judge
            )
            retrieval_score = score_retrieval(item, result["retrieved_chunks"], judge)
            record["answer_score"] = dataclasses.asdict(answer_score)
            record["retrieval_score"] = (
                dataclasses.asdict(retrieval_score) if retrieval_score else None
            )
        except Exception as exc:
            logger.exception("Judge failed on %s", item.id)
            record["error"] = f"judge: {type(exc).__name__}: {exc}"

    if record["answer_score"] is None:
        # Errors count as failures so they can't silently inflate the pass rate.
        record["answer_score"] = dataclasses.asdict(
            AnswerScore(False, f"ERROR: {record['error']}")
        )

    record["usage"] = {
        "agent": _usage_records(generator) + _usage_records(grader),
        "judge": _usage_records(judge.llm, judge_start),
        "rerank_searches": rerank_searches(result) if result is not None else 0,
    }
    record["cost_usd"] = question_cost(record["usage"])
    return record


def _pass_group(records: list[dict[str, Any]]) -> dict[str, Any]:
    passed = sum(1 for r in records if r["answer_score"]["passed"])
    return {
        "n": len(records),
        "passed": passed,
        "pass_rate": round(passed / len(records), 4),
    }


def _group_by(records: Iterable[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        groups.setdefault(record[key], []).append(record)
    return {name: _pass_group(group) for name, group in groups.items()}


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [r["agent"]["latency_s"] for r in records if r["agent"] is not None]
    retrieval = [
        r["retrieval_score"] for r in records if r["retrieval_score"] is not None
    ]
    retries = [r["agent"]["retry_count"] for r in records if r["agent"] is not None]

    def tokens(kind: str, field_: str) -> int:
        return sum(u[field_] for r in records for u in r["usage"][kind])

    overall = (
        _pass_group(records) if records else {"n": 0, "passed": 0, "pass_rate": None}
    )
    return {
        "n": len(records),
        "n_errors": sum(1 for r in records if r["error"] is not None),
        "answer_passed": overall["passed"],
        "answer_pass_rate": overall["pass_rate"],
        "retrieval_n": len(retrieval),
        "retrieval_passed": sum(1 for s in retrieval if s["passed"]),
        "retrieval_pass_rate": (
            round(sum(1 for s in retrieval if s["passed"]) / len(retrieval), 4)
            if retrieval
            else None
        ),
        "by_question_type": _group_by(records, "question_type"),
        "by_scoring_mode": _group_by(records, "scoring_mode"),
        "latency_s": {
            "total": round(sum(latencies), 3),
            "mean": round(statistics.mean(latencies), 3) if latencies else None,
            "p50": round(statistics.median(latencies), 3) if latencies else None,
            "max": round(max(latencies), 3) if latencies else None,
        },
        "tokens": {
            "agent_prompt": tokens("agent", "prompt_tokens"),
            "agent_completion": tokens("agent", "completion_tokens"),
            "judge_prompt": tokens("judge", "prompt_tokens"),
            "judge_completion": tokens("judge", "completion_tokens"),
        },
        "rerank_searches": sum(r["usage"]["rerank_searches"] for r in records),
        "cost_usd": summarize_costs(r["cost_usd"] for r in records),
        "mean_retry_count": round(statistics.mean(retries), 3) if retries else None,
    }


def run_eval(
    config: EvalConfig,
    *,
    judge: LLMJudge | None = None,
    store: VectorStore | None = None,
    agent_llms: AgentLLMFactory | None = None,
) -> dict[str, Any]:
    """Run the full golden set under `config` and return the results dict.
    `judge`, `store` and `agent_llms` are injectable for tests; a store built
    here is reused across questions and closed at the end."""
    items = load_golden_set(config.golden_path)
    if config.question_ids:
        wanted = set(config.question_ids)
        items = [item for item in items if item.id in wanted]

    judge = judge or LLMJudge(OpenAILLMClient(config.judge_model))
    agent_llms = agent_llms or default_agent_llms
    owns_store = store is None
    if store is None:
        store = make_store(config.backend)

    started_at = datetime.now(UTC)
    records = []
    try:
        for item in items:
            logger.info("Evaluating %s (%s)", item.id, item.scoring_mode)
            record = evaluate_question(
                item, config, store=store, judge=judge, agent_llms=agent_llms
            )
            logger.info(
                "%s -> %s",
                item.id,
                "PASS" if record["answer_score"]["passed"] else "FAIL",
            )
            records.append(record)
    finally:
        if owns_store:
            store.close()
    finished_at = datetime.now(UTC)

    return {
        "run": {
            "run_id": f"{started_at:%Y-%m-%dT%H-%M-%SZ}_{config.backend}_{config.retrieval_mode}",
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "git_commit": _git_commit(),
            "config": {
                "backend": config.backend,
                "retrieval_mode": config.retrieval_mode,
                "judge_model": config.judge_model,
                "agent_model": agent_model(),
                "grade_model": grade_model(),
                "max_retries": config.max_retries,
                "top_k": config.top_k,
                "golden_path": config.golden_path.as_posix(),
                "golden_sha256": file_sha256(config.golden_path),
                "question_ids": list(config.question_ids) or None,
            },
        },
        "summary": summarize(records),
        "questions": records,
    }
