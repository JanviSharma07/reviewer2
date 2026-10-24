"""Small wrapper around the Groq API that always returns parsed JSON."""

import json
import time

from groq import BadRequestError, Groq, RateLimitError

# Open-weight models on Groq's free tier (the Llama 3 models were retired on 16 Aug 2026).
# 120B is smarter; 20B is faster with higher free limits, good for testing.
MODELS = ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]


class LLM:
    def __init__(self, api_key: str, model: str = MODELS[0]):
        self.client = Groq(api_key=api_key)
        self.model = model

    def ask_json(self, system: str, user: str, max_tokens: int = 1500) -> dict:
        extra = {}
        if "gpt-oss" in self.model:
            # GPT-OSS "thinks" before answering, and that thinking counts toward max_tokens.
            # Keep thinking short and leave room for the actual answer.
            extra["reasoning_effort"] = "low"
            max_tokens += 1500
        last_error = None
        for attempt in range(4):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": user}],
                    temperature=0.1,
                    max_tokens=max_tokens,
                    response_format={"type": "json_object"},
                    **extra,
                )
                content = resp.choices[0].message.content
                if not content:
                    raise json.JSONDecodeError("empty answer", "", 0)
                return json.loads(content)
            except RateLimitError as exc:
                # Free tier has per-minute limits: wait and retry.
                last_error = exc
                retry_after = exc.response.headers.get("retry-after") if exc.response else None
                try:
                    wait = min(float(retry_after), 30) if retry_after else 5 * (attempt + 1)
                except ValueError:
                    wait = 5 * (attempt + 1)
                time.sleep(wait)
            except (BadRequestError, json.JSONDecodeError) as exc:
                # Usually the model produced invalid JSON; try again.
                last_error = exc
                if isinstance(exc, BadRequestError) and "json" not in str(exc).lower():
                    raise
        raise RuntimeError(f"The model did not return a valid answer: {last_error}")
