"""Unit and integration tests for AI Text Summarizer."""

import os
import glob
import pytest
from fastapi.testclient import TestClient

# Set mock mode before importing app
os.environ["MOCK_LLM"] = "true"

from app.main import app
from app.prompts import build_summarize_prompt
from app.llm import LLMClient

client = TestClient(app)


def test_prompts_generation():
    """Verify prompt builder formats system and user prompts with style/length controls."""
    sample_text = "Artificial intelligence enables automation and decision support across modern enterprises."
    
    # Test bullets & short
    sys_p, user_p, max_tok = build_summarize_prompt(sample_text, format_style="bullets", length="short")
    assert "STRICT FACTUALITY GUIDELINES" in sys_p
    assert "Target length: 3 concise bullet points" in user_p
    assert max_tok == 150
    assert sample_text in user_p

    # Test paragraph & detailed
    sys_p, user_p, max_tok = build_summarize_prompt(sample_text, format_style="paragraph", length="detailed")
    assert "prose paragraph" in user_p
    assert max_tok == 700

    # Test tldr
    sys_p, user_p, max_tok = build_summarize_prompt(sample_text, format_style="tldr", length="medium")
    assert "TL;DR" in user_p


def test_health_endpoint():
    """Verify /health endpoint returns healthy status and metadata."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "model" in data
    assert "mock_mode" in data
    assert data["version"] == "1.0.0"


def test_summarize_success_mock():
    """Verify /summarize works with valid payload in mock mode."""
    long_text = (
        "The transition toward renewable power sources has accelerated globally over the past decade. "
        "Solar photovoltaic panels and wind turbines are now cost-competitive with traditional fossil fuels. "
        "However, integrating intermittent generation requires substantial investments in battery energy storage systems."
    )
    payload = {
        "text": long_text,
        "format_style": "bullets",
        "length": "medium",
    }
    response = client.post("/summarize", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "summary" in data
    assert len(data["summary"]) > 0
    assert data["format_style"] == "bullets"
    assert data["length"] == "medium"
    assert data["latency_ms"] >= 0
    assert data["prompt_tokens"] > 0
    assert data["completion_tokens"] > 0
    assert data["total_tokens"] > 0
    assert data["input_char_count"] == len(long_text)


def test_summarize_validation_too_short():
    """Verify /summarize rejects inputs with fewer than 50 characters."""
    payload = {
        "text": "Short text.",
        "format_style": "bullets",
        "length": "short",
    }
    response = client.post("/summarize", json=payload)
    assert response.status_code == 422  # Pydantic validation error


def test_summarize_validation_not_enough_words():
    """Verify /summarize rejects inputs with fewer than 10 words even if characters >= 50."""
    payload = {
        "text": "A" * 60,  # 60 characters but 1 word
        "format_style": "bullets",
        "length": "short",
    }
    response = client.post("/summarize", json=payload)
    assert response.status_code == 422


def test_all_sample_files_exist_and_readable():
    """Verify all 10 sample files exist, have content, and are valid inputs."""
    sample_files = sorted(glob.glob("samples/*.txt"))
    assert len(sample_files) == 10, f"Expected 10 sample files, found {len(sample_files)}"

    for file_path in sample_files:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            assert len(content) >= 50, f"{file_path} is too short ({len(content)} chars)"
            assert len(content.split()) >= 10, f"{file_path} has too few words"
