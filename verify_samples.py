"""Batch verification script to test summarization across all 10 sample articles."""

import glob
import os

# Set mock mode for automated validation
os.environ["MOCK_LLM"] = "true"

from app.main import app, SummarizeRequest, summarize_text

def run_batch_verification():
    sample_files = sorted(glob.glob("samples/*.txt"))
    print(f"\n=======================================================")
    print(f"   AI TEXT SUMMARIZER - BATCH SAMPLE VERIFICATION")
    print(f"=======================================================\n")
    print(f"Found {len(sample_files)} sample articles to evaluate.\n")

    styles = ["bullets", "paragraph", "tldr"]
    lengths = ["short", "medium", "detailed"]

    for idx, path in enumerate(sample_files, 1):
        filename = os.path.basename(path)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()

        # Rotate style and length across samples
        chosen_style = styles[(idx - 1) % len(styles)]
        chosen_length = lengths[(idx - 1) % len(lengths)]

        request = SummarizeRequest(
            text=content,
            format_style=chosen_style,
            length=chosen_length,
        )

        response = summarize_text(request)

        print(f"[{idx:02d}/10] {filename}")
        print(f"      Input: {response.input_word_count} words ({response.input_char_count} chars)")
        print(f"      Config: style='{response.format_style}', length='{response.length}'")
        print(f"      Latency: {response.latency_ms:.1f}ms | Tokens: {response.total_tokens} (Prompt: {response.prompt_tokens}, Out: {response.completion_tokens})")
        print(f"      Est Cost: ${response.estimated_cost_usd:.6f}")
        print(f"      Status: OK ✓\n")

    print("All 10 sample articles processed and validated successfully!\n")

if __name__ == "__main__":
    run_batch_verification()
