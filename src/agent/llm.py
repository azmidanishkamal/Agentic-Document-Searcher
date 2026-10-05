"""LLM client used by the agent's grade/reformulate/generate nodes.

Nodes depend only on the `LLMClient` protocol, so tests can inject a scripted
fake and never touch the network."""

from __future__ import annotations

import json
import os
from typing import Any, Protocol

from openai import OpenAI

AGENT_LLM_MODEL_ENV_VAR = "AGENT_LLM_MODEL"
AGENT_GRADE_MODEL_ENV_VAR = "AGENT_GRADE_MODEL"
# Mid-tier rather than mini/nano so answer quality doesn't confound retrieval
# comparisons in the eval harness.
DEFAULT_AGENT_LLM_MODEL = "gpt-5.4"


def agent_model() -> str:
    """Model for the generate step."""
    return os.environ.get(AGENT_LLM_MODEL_ENV_VAR) or DEFAULT_AGENT_LLM_MODEL


def grade_model() -> str:
    """Model for the lightweight grade/reformulate steps; falls back to the
    generate model unless overridden."""
    return os.environ.get(AGENT_GRADE_MODEL_ENV_VAR) or agent_model()


class LLMClient(Protocol):
    def complete_json(
        self, *, system: str, user: str, schema_name: str, schema: dict[str, Any]
    ) -> dict[str, Any]:
        """Return the model's reply parsed as JSON conforming to `schema`."""
        ...


class OpenAILLMClient:
    """`LLMClient` backed by OpenAI chat completions with strict JSON-schema output."""

    def __init__(self, model: str, client: OpenAI | None = None) -> None:
        self.model = model
        self.client = client or OpenAI()

    def complete_json(
        self, *, system: str, user: str, schema_name: str, schema: dict[str, Any]
    ) -> dict[str, Any]:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": schema_name, "schema": schema, "strict": True},
            },
        )
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError(f"{self.model} returned no content for {schema_name}")
        return json.loads(content)
