"""
Shared fixtures and configuration for deepeval LLM evaluation tests.
"""

import asyncio
import os

import pytest
import requests
from deepeval.metrics import GEval
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCaseParams


class OllamaJudgeModel(DeepEvalBaseLLM):
    """Use the configured Ollama endpoint as the DeepEval judge."""

    def __init__(self, model=None, base_url=None, api_key=None):
        self.base_url = (
            base_url or os.getenv("OLLAMA_BASE_URL", "https://ollama.com")
        ).rstrip("/")
        self.api_key = (
            api_key if api_key is not None else os.getenv("OLLAMA_API_KEY", "")
        )
        super().__init__(
            model
            or os.getenv("DEEPEVAL_OLLAMA_MODEL")
            or os.getenv("OLLAMA_MODEL_CLASSIFY", "gemma4:31b")
        )

    def load_model(self):
        return self

    def generate(self, prompt, schema=None, **kwargs):
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key

        payload = {
            "model": self.name,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }
        if schema:
            payload["format"] = schema.model_json_schema()

        response = requests.post(
            f"{self.base_url}/api/chat",
            headers=headers,
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        content = response.json()["message"]["content"]
        return schema.model_validate_json(content) if schema else content

    async def a_generate(self, prompt, schema=None, **kwargs):
        return await asyncio.to_thread(
            self.generate, prompt, schema=schema, **kwargs
        )

    def get_model_name(self):
        return f"{self.name} (Ollama)"

    def supports_structured_outputs(self):
        return True


judge_model = OllamaJudgeModel()


# ---------------------------------------------------------------------------
# Reusable GEval metric factories
# ---------------------------------------------------------------------------

def json_schema_metric(schema_description: str):
    """Creates a GEval metric that checks JSON schema compliance."""
    return GEval(
        name="JSON Schema Compliance",
        criteria=(
            "Evaluate whether the actual output is valid JSON that conforms to "
            "the required schema. Only check structure, key names, and data "
            "types — do NOT penalize for specific values. "
            + schema_description
        ),
        evaluation_params=[
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ],
        threshold=0.5,
        model=judge_model,
    )


def output_correctness_metric():
    """Creates a GEval metric that checks factual/logical correctness."""
    return GEval(
        name="Output Correctness",
        criteria=(
            "Determine whether the actual output is logically correct and "
            "reasonable given the input text. The analysis should make sense "
            "for the provided input."
        ),
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ],
        threshold=0.5,
        model=judge_model,
    )


def answer_relevancy_metric():
    """Creates a GEval metric that checks whether the output is topically
    relevant to the input.  Unlike AnswerRelevancyMetric (which assumes a
    Q&A format), this works for classification and analysis endpoints where
    the output is structured metadata about the input text."""
    return GEval(
        name="Answer Relevancy",
        criteria=(
            "Evaluate whether the actual output is topically relevant to the "
            "input text. The labels, categories, or analysis in the output "
            "should directly relate to the subject matter of the input. "
            "Structured metadata (labels, categories, confidence scores) that "
            "accurately describes the input text should be considered relevant."
        ),
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ],
        threshold=0.5,
        model=judge_model,
    )
