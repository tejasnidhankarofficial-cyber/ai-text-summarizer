"""LLM Client integration module with latency, token usage tracking, and provider neutrality."""

import os
import time
import logging
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError, AuthenticationError, RateLimitError, APIConnectionError

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# Pricing per 1M tokens for estimation (USD) - default gpt-4o-mini
PRICING_PER_1M = {
    "gpt-4o-mini": {"prompt": 0.15, "completion": 0.60},
    "gpt-4o": {"prompt": 2.50, "completion": 10.00},
    "gpt-3.5-turbo": {"prompt": 0.50, "completion": 1.50},
    "default": {"prompt": 0.15, "completion": 0.60},
}


class LLMClient:
    """Provider-neutral LLM client wrapper with timing and token usage instrumentation."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "").strip()
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
        self.base_url = os.getenv("OPENAI_BASE_URL", None)
        self.mock_mode = os.getenv("MOCK_LLM", "false").lower() in ("true", "1", "yes")

        self.client = None
        if self.api_key and not self.mock_mode:
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def is_configured(self) -> bool:
        """Check if an API key is configured or running in mock mode."""
        return bool(self.api_key) or self.mock_mode

    def _estimate_cost(self, prompt_tokens: int, completion_tokens: int, model: str) -> float:
        """Calculate estimated cost in USD."""
        rates = PRICING_PER_1M.get(model, PRICING_PER_1M["default"])
        cost = (prompt_tokens / 1_000_000 * rates["prompt"]) + (
            completion_tokens / 1_000_000 * rates["completion"]
        )
        return round(cost, 6)

    def summarize(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 350,
        temperature: float = 0.3
    ) -> Dict[str, Any]:
        """
        Executes the summarization prompt against the LLM, tracking latency and tokens.

        Returns:
            Dict containing:
                - summary (str)
                - latency_ms (float)
                - prompt_tokens (int)
                - completion_tokens (int)
                - total_tokens (int)
                - estimated_cost_usd (float)
                - model (str)
        """
        start_time = time.perf_counter()

        # Check for mock mode or missing API key
        if self.mock_mode or not self.client:
            if not self.mock_mode and not self.api_key:
                raise ValueError(
                    "OPENAI_API_KEY is not configured. Please add your API key to .env, "
                    "or set MOCK_LLM=true for local simulation."
                )

            # Simulated mock response for local testing
            time.sleep(0.45)  # Simulate network latency
            duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
            mock_summary = (
                "- [MOCK MODE] Key takeaway 1: High fidelity summary generated without consuming API tokens.\n"
                "- [MOCK MODE] Key takeaway 2: Input processing and length validation passed successfully.\n"
                "- [MOCK MODE] Key takeaway 3: To use live OpenAI generation, set OPENAI_API_KEY in your .env file."
            )
            prompt_tokens = len(user_prompt.split()) + 40
            completion_tokens = len(mock_summary.split())
            total_tokens = prompt_tokens + completion_tokens

            return {
                "summary": mock_summary,
                "latency_ms": duration_ms,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "estimated_cost_usd": self._estimate_cost(prompt_tokens, completion_tokens, self.model),
                "model": f"{self.model} (mock)",
            }

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            duration_ms = round((time.perf_counter() - start_time) * 1000, 2)

            summary_text = response.choices[0].message.content or ""
            usage = response.usage

            prompt_tokens = usage.prompt_tokens if usage else 0
            completion_tokens = usage.completion_tokens if usage else 0
            total_tokens = usage.total_tokens if usage else (prompt_tokens + completion_tokens)

            cost = self._estimate_cost(prompt_tokens, completion_tokens, self.model)

            logger.info(
                f"Summarization succeeded | Model: {self.model} | "
                f"Latency: {duration_ms}ms | Total Tokens: {total_tokens} | Cost: ${cost:.6f}"
            )

            return {
                "summary": summary_text.strip(),
                "latency_ms": duration_ms,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "estimated_cost_usd": cost,
                "model": self.model,
            }

        except AuthenticationError as e:
            logger.error(f"OpenAI Authentication error: {e}")
            raise ValueError(f"Invalid OpenAI API Key: {str(e)}") from e
        except RateLimitError as e:
            logger.error(f"OpenAI Rate limit error: {e}")
            raise RuntimeError("OpenAI rate limit or quota exceeded. Please check your account usage/credits.") from e
        except APIConnectionError as e:
            logger.error(f"OpenAI Connection error: {e}")
            raise ConnectionError(f"Failed to connect to OpenAI API: {str(e)}") from e
        except OpenAIError as e:
            logger.error(f"OpenAI General API error: {e}")
            raise RuntimeError(f"OpenAI API Error: {str(e)}") from e
        except Exception as e:
            logger.error(f"Unexpected error in LLM summarization: {e}")
            raise e
