import json
import re

from dotenv import load_dotenv
from deepeval.models import DeepEvalBaseLLM
from langchain_groq import ChatGroq

load_dotenv()

JUDGE_MODEL_ID = "qwen/qwen3.8-27b"

# Tune this downward if you still hit 413.
# 12,000 characters is roughly 3,000-4,000 tokens for normal English/JSON.
MAX_PROMPT_CHARS = 12_000


def _strip_think(text: str) -> str:
    return re.sub(r"", "", text, flags=re.DOTALL).strip()


def _truncate_prompt(prompt: str, max_chars: int = MAX_PROMPT_CHARS) -> str:
    if len(prompt) <= max_chars:
        return prompt

    keep = max_chars // 2
    return (
        prompt[:keep]
        + "\n\n[TRUNCATED: middle portion of the agent trace removed to fit model input limits]\n\n"
        + prompt[-keep:]
    )


def _extract_json(text: str):
    text = _strip_think(text)

    # Prefer a fenced JSON block if present.
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        return json.loads(fenced.group(1))

    # Fall back to the first balanced-looking JSON object.
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in model output:\n{text[:1000]}")

    return json.loads(match.group(0))


class GroqJudge(DeepEvalBaseLLM):
    def __init__(self, model_name: str = JUDGE_MODEL_ID):
        self.model_name = model_name
        self.client = ChatGroq(
            model=model_name,
            temperature=0,
            max_retries=6,
        )

    def load_model(self):
        return self.client

    def generate(self, prompt: str, schema=None):
        prompt = _truncate_prompt(prompt)

        if schema is None:
            return _strip_think(self.client.invoke(prompt).content)

        try:
            return self.client.with_structured_output(schema).invoke(prompt)
        except Exception:
            text = _strip_think(
                self.client.invoke(prompt + "\nReturn only valid JSON.").content
            )
            data = _extract_json(text)
            return schema(**data)

    async def a_generate(self, prompt: str, schema=None):
        return self.generate(prompt, schema)

    def get_model_name(self):
        return self.model_name


eval_model = GroqJudge()
