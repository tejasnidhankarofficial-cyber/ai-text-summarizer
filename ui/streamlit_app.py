"""Streamlit Frontend for AI Text Summarizer."""

import os
import glob
import requests
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="AI Text Summarizer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern styling and aesthetics
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        padding: 1.8rem 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
        border: 1px solid #334155;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    .main-header h1 {
        margin: 0;
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .main-header p {
        margin: 0.4rem 0 0 0;
        color: #94a3b8;
        font-size: 0.95rem;
    }

    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 0.9rem 1.2rem;
        text-align: center;
    }
    .metric-value {
        font-size: 1.4rem;
        font-weight: 700;
        color: #0f172a;
    }
    .metric-label {
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748b;
        margin-top: 0.2rem;
    }

    .summary-box {
        background-color: #f1f5f9;
        border-left: 4px solid #3b82f6;
        padding: 1.25rem 1.5rem;
        border-radius: 8px;
        font-size: 1rem;
        line-height: 1.65;
        color: #1e293b;
        margin-top: 0.5rem;
    }

    .stButton button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s ease-in-out;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Backend configuration
API_URL = os.getenv("BACKEND_API_URL", "http://127.0.0.1:8000")


def check_api_health():
    """Verify backend connectivity."""
    try:
        res = requests.get(f"{API_URL}/health", timeout=2.5)
        if res.status_code == 200:
            return True, res.json()
        return False, None
    except Exception:
        return False, None


def load_sample_files():
    """Discover sample text files in the samples directory."""
    sample_files = sorted(glob.glob("samples/*.txt"))
    samples = {}
    for path in sample_files:
        name = os.path.basename(path).replace(".txt", "").replace("_", " ").title()
        try:
            with open(path, "r", encoding="utf-8") as f:
                samples[name] = f.read()
        except Exception:
            continue
    return samples


# Initialize session state
if "source_text" not in st.session_state:
    st.session_state["source_text"] = ""
if "last_result" not in st.session_state:
    st.session_state["last_result"] = None

# Header Banner
st.markdown(
    """
    <div class="main-header">
        <h1>⚡ AI Text Summarizer</h1>
        <p>Production-ready summarization engine powered by FastAPI, Pydantic, and OpenAI. Delivers faithful summaries in seconds with token & latency analytics.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar Controls
with st.sidebar:
    st.header("⚙️ Configuration")

    # API Health Badge
    is_healthy, health_data = check_api_health()
    if is_healthy:
        st.success(f"Backend Online | Model: `{health_data.get('model', 'N/A')}`")
        if not health_data.get("api_key_configured") and not health_data.get("mock_mode"):
            st.warning("⚠️ OPENAI_API_KEY is not configured in .env. Real generation requires an API key or MOCK_LLM=true.")
    else:
        st.error(f"Backend Offline at `{API_URL}`. Start FastAPI server using: `uvicorn app.main:app --reload`")

    st.divider()

    # Format Style Selector
    format_style = st.selectbox(
        "📝 Summary Format",
        options=["bullets", "paragraph", "tldr"],
        format_func=lambda x: {
            "bullets": "Bullet Points (Key Takeaways)",
            "paragraph": "Cohesive Narrative Paragraph",
            "tldr": "Executive TL;DR (1-2 sentences)",
        }[x],
        index=0,
        help="Select the structural representation of the summary.",
    )

    # Length Selector
    length = st.select_slider(
        "📏 Summary Length",
        options=["short", "medium", "detailed"],
        value="medium",
        format_func=lambda x: {
            "short": "Short (~50-80 words)",
            "medium": "Medium (~120-200 words)",
            "detailed": "Detailed (~250-400 words)",
        }[x],
        help="Controls the output detail and token budget.",
    )

    st.divider()

    # Sample Loader
    st.subheader("📚 Test Samples")
    samples = load_sample_files()
    if samples:
        selected_sample = st.selectbox(
            "Load pre-made sample article:",
            options=["-- Select a sample article --"] + list(samples.keys()),
        )
        if st.button("Load Selected Sample", use_container_width=True):
            if selected_sample != "-- Select a sample article --":
                st.session_state["source_text"] = samples[selected_sample]
                st.rerun()

    st.divider()
    st.caption("AI Text Summarizer • Portfolio Project")


# Main Area
col_input, col_output = st.columns([1, 1], gap="large")

with col_input:
    st.subheader("📄 Source Text")

    input_text = st.text_area(
        label="Input text",
        value=st.session_state["source_text"],
        placeholder="Paste your article, meeting notes, research paper, or report here (minimum 50 characters)...",
        height=380,
        label_visibility="collapsed",
    )

    # Update session state text
    st.session_state["source_text"] = input_text

    # Text statistics bar
    char_count = len(input_text)
    word_count = len(input_text.split())

    col_stats, col_actions = st.columns([1, 1])
    with col_stats:
        st.caption(f"📊 **{word_count}** words | **{char_count}** characters")

    with col_actions:
        if st.button("🧹 Clear Input", use_container_width=True):
            st.session_state["source_text"] = ""
            st.session_state["last_result"] = None
            st.rerun()

    summarize_button = st.button(
        "⚡ Generate Summary",
        type="primary",
        use_container_width=True,
        disabled=(char_count < 50),
    )

    if char_count > 0 and char_count < 50:
        st.info("💡 Minimum input length is 50 characters to produce a meaningful summary.")

with col_output:
    st.subheader("✨ Summary & Metrics")

    if summarize_button:
        if char_count < 50:
            st.error("Input text is too short. Please provide at least 50 characters.")
        else:
            with st.spinner("Analyzing text and generating summary..."):
                payload = {
                    "text": input_text,
                    "format_style": format_style,
                    "length": length,
                }
                try:
                    res = requests.post(f"{API_URL}/summarize", json=payload, timeout=45)
                    if res.status_code == 200:
                        st.session_state["last_result"] = res.json()
                    else:
                        detail = res.json().get("detail", res.text)
                        st.error(f"Error ({res.status_code}): {detail}")
                except requests.exceptions.ConnectionError:
                    st.error(
                        f"Could not connect to FastAPI server at `{API_URL}`. "
                        "Please ensure the backend is running: `uvicorn app.main:app --reload`"
                    )
                except Exception as e:
                    st.error(f"Request failed: {str(e)}")

    # Display results if available
    result = st.session_state["last_result"]
    if result:
        # Metrics row
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-value">{result['latency_ms']:.0f} ms</div>
                    <div class="metric-label">Latency</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with m2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-value">{result['prompt_tokens']}</div>
                    <div class="metric-label">Prompt Tokens</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with m3:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-value">{result['completion_tokens']}</div>
                    <div class="metric-label">Output Tokens</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with m4:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-value">${result['estimated_cost_usd']:.5f}</div>
                    <div class="metric-label">Est. Cost</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

        # Summary output container
        st.markdown(f"**Format:** `{result['format_style']}` | **Length:** `{result['length']}` | **Model:** `{result['model']}`")
        st.markdown(result["summary"])

        st.divider()

        # Action bar
        col_down1, col_down2 = st.columns([1, 1])
        with col_down1:
            st.download_button(
                label="📥 Download Markdown Summary",
                data=f"# AI Summary\n\n**Source text word count:** {result['input_word_count']}\n**Model:** {result['model']}\n**Latency:** {result['latency_ms']} ms\n\n---\n\n{result['summary']}\n",
                file_name="summary.md",
                mime="text/markdown",
                use_container_width=True,
            )
        with col_down2:
            compression_ratio = round((1 - (len(result["summary"].split()) / max(result["input_word_count"], 1))) * 100, 1)
            st.metric(
                label="Text Reduction",
                value=f"{max(0, compression_ratio)}%",
                help="Percentage of words compressed by the summarizer.",
            )
    else:
        st.info("👉 Paste an article on the left or select a preloaded sample, then click **Generate Summary**.")
