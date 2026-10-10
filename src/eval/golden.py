"""Golden Q&A set: loading and the scoring mode each item is graded under."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

DEFAULT_GOLDEN_PATH = Path("eval_data/golden_set.jsonl")

ScoringMode = Literal["reference", "rubric", "negative"]
NEGATIVE_QUESTION_TYPE = "negative_unanswerable"


@dataclass(frozen=True)
class GoldenItem:
    id: str
    question: str
    question_type: str
    eval_type: str
    draft_answer: str = ""  # absent on some rubric-only items
    relevant_chunk_ids: list[str] = field(default_factory=list)
    supporting_chunk_text: dict[str, str] = field(default_factory=dict)
    answer_rubric: dict[str, Any] | None = None

    @property
    def scoring_mode(self) -> ScoringMode:
        """Negatives are tagged `eval_type: "reference"` in the golden file, so
        `question_type` takes precedence: they're scored on the answerable flag."""
        if self.question_type == NEGATIVE_QUESTION_TYPE:
            return "negative"
        if self.eval_type == "rubric":
            if not self.answer_rubric:
                raise ValueError(f"{self.id}: eval_type 'rubric' but no answer_rubric")
            return "rubric"
        if self.eval_type == "reference":
            return "reference"
        raise ValueError(f"{self.id}: unknown eval_type {self.eval_type!r}")

    @property
    def reference_facts(self) -> str:
        """What the retrieval judge checks the chunks against: the golden answer,
        or for rubric items the rubric's required points only. Credit points are
        optional for passing, so retrieval isn't penalized for missing them."""
        if self.scoring_mode == "rubric":
            assert self.answer_rubric is not None
            points = self.answer_rubric.get("required_points", [])
            return "\n".join(f"- {point}" for point in points)
        return self.draft_answer

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GoldenItem:
        return cls(
            id=data["id"],
            question=data["question"],
            question_type=data["question_type"],
            eval_type=data["eval_type"],
            draft_answer=data.get("draft_answer", ""),
            relevant_chunk_ids=list(data.get("relevant_chunk_ids", [])),
            supporting_chunk_text=dict(data.get("supporting_chunk_text", {})),
            answer_rubric=data.get("answer_rubric"),
        )


def load_golden_set(path: Path = DEFAULT_GOLDEN_PATH) -> list[GoldenItem]:
    items = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = GoldenItem.from_dict(json.loads(line))
                _ = item.scoring_mode  # fail fast on a malformed item, not mid-run
                items.append(item)
    return items


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
