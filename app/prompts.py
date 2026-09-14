"""Prompt templates and prompt generation logic for the AI Text Summarizer."""

from typing import Tuple, Dict, Any

STYLE_INSTRUCTIONS: Dict[str, str] = {
    "bullets": (
        "Format the summary as concise, high-impact bullet points. "
        "Each bullet point must highlight a distinct key insight, fact, or conclusion from the source text. "
        "Do not repeat ideas across bullet points."
    ),
    "paragraph": (
        "Format the summary as a cohesive, well-structured prose paragraph (or multiple paragraphs if detailed). "
        "Use natural transitions between sentences to ensure smooth readability while preserving logical flow."
    ),
    "tldr": (
        "Format the summary as an ultra-concise Executive TL;DR (1-2 sentences maximum). "
        "Deliver the single most critical takeaway and core outcome immediately."
    ),
}

LENGTH_CONFIG: Dict[str, Dict[str, Any]] = {
    "short": {
        "description": "Short & Concise (~50-80 words). Provide only the essential core point.",
        "max_tokens": 150,
        "target_bullets": "3 concise bullet points",
        "target_sentences": "2-3 crisp sentences",
    },
    "medium": {
        "description": "Medium & Balanced (~120-200 words). Cover main findings and supporting context.",
        "max_tokens": 350,
        "target_bullets": "5 informative bullet points",
        "target_sentences": "4-6 well-developed sentences across 1-2 paragraphs",
    },
    "detailed": {
        "description": "Detailed & Comprehensive (~250-400 words). Thorough breakdown of background, findings, and implications.",
        "max_tokens": 700,
        "target_bullets": "7-10 detailed bullet points with supporting details",
        "target_sentences": "2-3 comprehensive paragraphs covering background, evidence, and conclusions",
    },
}

SYSTEM_PROMPT = (
    "You are an expert AI executive summarizer. Your goal is to provide faithful, accurate, "
    "and objective summaries of user-provided texts.\n\n"
    "STRICT FACTUALITY GUIDELINES:\n"
    "1. Rely ONLY on the facts explicitly mentioned in the provided text.\n"
    "2. Do NOT invent, assume, extrapolate, or bring in external knowledge.\n"
    "3. Avoid editorializing or adding opinions not present in the source.\n"
    "4. Retain important metrics, dates, and quantitative figures when present.\n"
    "5. If the source text is ambiguous or inconclusive on a topic, reflect that accurately."
)


def build_summarize_prompt(
    text: str,
    format_style: str = "bullets",
    length: str = "medium"
) -> Tuple[str, str, int]:
    """
    Constructs the system prompt, user prompt, and max_tokens setting for the LLM.

    Args:
        text: The source text to summarize.
        format_style: 'bullets', 'paragraph', or 'tldr'.
        length: 'short', 'medium', or 'detailed'.

    Returns:
        Tuple of (system_prompt, user_prompt, max_tokens).
    """
    style_rule = STYLE_INSTRUCTIONS.get(format_style, STYLE_INSTRUCTIONS["bullets"])
    length_spec = LENGTH_CONFIG.get(length, LENGTH_CONFIG["medium"])
    max_tokens = length_spec["max_tokens"]

    if format_style == "bullets":
        target_constraint = f"Target length: {length_spec['target_bullets']}."
    elif format_style == "tldr":
        target_constraint = "Target length: 1-2 impactful sentences."
    else:
        target_constraint = f"Target length: {length_spec['target_sentences']}."

    user_prompt = f"""Please summarize the following text according to these instructions:

--- FORMAT INSTRUCTIONS ---
- Style: {style_rule}
- Length Constraint: {length_spec['description']} ({target_constraint})
- Objectivity: Strictly faithful to the source material without hallucinations or omissions of critical points.

--- SOURCE TEXT ---
{text.strip()}

--- SUMMARY ---
"""
    return SYSTEM_PROMPT, user_prompt, max_tokens
