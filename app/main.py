"""FastAPI Backend Application for AI Text Summarizer."""

import os
import logging
from typing import Literal
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from dotenv import load_dotenv

from app.prompts import build_summarize_prompt
from app.llm import LLMClient

# Load environment configuration
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("ai_text_summarizer")

app = FastAPI(
    title="AI Text Summarizer API",
    description="High-performance, faithful text summarization API powered by LLMs with style, length, and cost instrumentation.",
    version="1.0.0",
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize LLM client
llm_client = LLMClient()

MIN_INPUT_CHARS = 50
MAX_INPUT_CHARS = 30000


class SummarizeRequest(BaseModel):
    text: str = Field(
        ...,
        description="The source text to summarize.",
        min_length=MIN_INPUT_CHARS,
        max_length=MAX_INPUT_CHARS,
    )
    format_style: Literal["bullets", "paragraph", "tldr"] = Field(
        default="bullets",
        description="Summary format: 'bullets', 'paragraph', or 'tldr'."
    )
    length: Literal["short", "medium", "detailed"] = Field(
        default="medium",
        description="Desired length constraint: 'short', 'medium', or 'detailed'."
    )

    @field_validator("text")
    @classmethod
    def validate_meaningful_content(cls, v: str) -> str:
        stripped = v.strip()
        if len(stripped.split()) < 10:
            raise ValueError(
                f"Input text must contain at least 10 words (received {len(stripped.split())} words)."
            )
        return stripped


class SummarizeResponse(BaseModel):
    summary: str
    format_style: str
    length: str
    model: str
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost_usd: float
    input_char_count: int
    input_word_count: int


class HealthResponse(BaseModel):
    status: str
    model: str
    api_key_configured: bool
    mock_mode: bool
    version: str


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check() -> HealthResponse:
    """Check service health and API configuration status."""
    return HealthResponse(
        status="healthy",
        model=llm_client.model,
        api_key_configured=bool(llm_client.api_key),
        mock_mode=llm_client.mock_mode,
        version="1.0.0",
    )


@app.post(
    "/summarize",
    response_model=SummarizeResponse,
    status_code=status.HTTP_200_OK,
    tags=["Summarization"],
    summary="Summarize input text with length and style constraints"
)
def summarize_text(payload: SummarizeRequest) -> SummarizeResponse:
    """
    Validates input text and generates a faithful summary with latency and token usage tracking.
    """
    input_char_count = len(payload.text)
    input_word_count = len(payload.text.split())

    logger.info(
        f"Incoming summarize request: {input_word_count} words ({input_char_count} chars), "
        f"style='{payload.format_style}', length='{payload.length}'"
    )

    # Build prompt
    system_prompt, user_prompt, max_tokens = build_summarize_prompt(
        text=payload.text,
        format_style=payload.format_style,
        length=payload.length,
    )

    try:
        result = llm_client.summarize(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=max_tokens,
        )
    except ValueError as e:
        logger.warning(f"Validation or configuration error: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except ConnectionError as e:
        logger.error(f"Network error: {e}")
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(e))
    except RuntimeError as e:
        logger.error(f"Runtime error: {e}")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))
    except Exception as e:
        logger.exception("Unhandled server error during summarization")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred during summarization: {str(e)}"
        )

    return SummarizeResponse(
        summary=result["summary"],
        format_style=payload.format_style,
        length=payload.length,
        model=result["model"],
        latency_ms=result["latency_ms"],
        prompt_tokens=result["prompt_tokens"],
        completion_tokens=result["completion_tokens"],
        total_tokens=result["total_tokens"],
        estimated_cost_usd=result["estimated_cost_usd"],
        input_char_count=input_char_count,
        input_word_count=input_word_count,
    )


if __name__ == "__main__":
    import uvicorn
    host = os.getenv("FASTAPI_HOST", "127.0.0.1")
    port = int(os.getenv("FASTAPI_PORT", "8000"))
    uvicorn.run("app.main:app", host=host, port=port, reload=True)
