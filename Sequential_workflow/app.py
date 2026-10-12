from typing import TypedDict

import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END

load_dotenv()


class PipelineState(TypedDict):
    raw_input: str
    edited_text: str
    script_text: str
    final_output: str


llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.7)


def editor_node(state: PipelineState):
    """Stage 1: Cleans up the raw text."""
    prompt = f"""

Clean the given raw text.

Rules:
- Remove extra spaces, blank lines, HTML tags, and scraping noise.
- Fix broken line breaks and formatting.
- Remove obvious duplicate text.
- Preserve headings, lists, names, dates, numbers, URLs, code, and important facts.
- Do not summarize, rewrite, or add information.
- Do not change the original meaning.

Return ONLY the cleaned text.

Raw text:

{state['raw_input']}

"""
    response = llm.invoke(prompt)
    return {"edited_text": response.content.strip()}


def scriptwriter_node(state: PipelineState):
    """Stage 2: Formats the clean text into an engaging video script."""
    prompt = f"""

Convert the given clean text into a short, engaging video script.

Rules:
- Start with a strong hook.
- Use simple, natural spoken language.
- Organize it logically: Hook → Introduction → Main Content → Conclusion.
- Keep all important facts and meaning.
- Do not invent information.
- Do not unnecessarily repeat content.
- Write for narration, not like an article.

Return ONLY the final video script.

Clean text:

{state['edited_text']}

"""
    response = llm.invoke(prompt)
    return {"script_text": response.content.strip()}


def translator_node(state: PipelineState):
    """Stage 3: Translates the script into natural Hinglish."""
    prompt = f"""

Convert the given text into natural, conversational Hinglish.

Rules:
- Use a natural mix of Hindi and English, like people commonly speak in India.
- Keep important English technical terms in English.
- Preserve the original meaning, facts, names, numbers, and examples.
- Do not add or remove information.
- Do not translate word-by-word.
- Make it sound natural when spoken aloud.
- Keep the language simple and easy to understand.
- Do not use overly formal Hindi.

Return ONLY the final Hinglish text.

Text:
{state['script_text']}

"""
    response = llm.invoke(prompt)
    return {"final_output": response.content.strip()}


graph = StateGraph(PipelineState)
graph.add_node("editor", editor_node)
graph.add_node("scriptwriter", scriptwriter_node)
graph.add_node("translator", translator_node)

graph.add_edge(START, "editor")
graph.add_edge("editor", "scriptwriter")
graph.add_edge("scriptwriter", "translator")
graph.add_edge("translator", END)

pipeline_app = graph.compile()


# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="ScriptFlow", page_icon="🎬", layout="centered")

st.markdown(
    """
<style>
#MainMenu, footer {visibility: hidden;}
.block-container {padding-top: 2.2rem; max-width: 820px;}

.hero {
    padding: 1.8rem 2rem; border-radius: 20px; margin-bottom: 1.6rem;
    background: linear-gradient(135deg, #6d28d9 0%, #2563eb 55%, #06b6d4 100%);
    color: #fff; box-shadow: 0 10px 30px rgba(37, 99, 235, .25);
}
.hero h1 {margin: 0; font-size: 2.1rem; color: #fff; letter-spacing: -.5px;}
.hero p {margin: .4rem 0 0; font-size: 1.02rem; opacity: .93;}

.steps {display: flex; gap: .6rem; margin: 1rem 0 .4rem;}
.step {
    flex: 1; padding: .75rem .9rem; border-radius: 14px;
    border: 1px solid rgba(128,128,128,.25); background: rgba(128,128,128,.06);
}
.step .n {font-size: .72rem; opacity: .6; text-transform: uppercase; letter-spacing: .08em;}
.step .t {font-weight: 600; font-size: .95rem; margin-top: .1rem;}
.step .s {font-size: .78rem; opacity: .65; margin-top: .15rem;}
.step.run {border-color: #f59e0b; background: rgba(245,158,11,.10);}
.step.run .s {color: #f59e0b; opacity: 1;}
.step.done {border-color: #10b981; background: rgba(16,185,129,.10);}
.step.done .s {color: #10b981; opacity: 1;}

.stTextArea textarea {border-radius: 14px; font-size: 1rem; line-height: 1.55;}
.stButton > button[kind="primary"] {
    border-radius: 12px; font-weight: 600; padding: .65rem 1rem;
    background: linear-gradient(90deg, #6d28d9, #2563eb); border: none;
}
.stButton > button[kind="primary"]:hover {filter: brightness(1.1);}
.stTabs [data-baseweb="tab"] {font-weight: 600;}
</style>

<div class="hero">
  <h1>🎬 ScriptFlow</h1>
  <p>Paste raw text and get a clean, narration-ready video script in Hinglish.</p>
</div>
""",
    unsafe_allow_html=True,
)

STAGES = [
    ("editor", "Editor", "Cleaning text"),
    ("scriptwriter", "Script writer", "Writing script"),
    ("translator", "Translator", "Hinglish"),
]


def render_steps(status: dict) -> str:
    html = '<div class="steps">'
    for i, (key, title, _) in enumerate(STAGES, 1):
        state = status.get(key, "wait")
        label = {"wait": "Waiting", "run": "Running…", "done": "Done ✓"}[state]
        html += (
            f'<div class="step {state}"><div class="n">Stage {i}</div>'
            f'<div class="t">{title}</div><div class="s">{label}</div></div>'
        )
    return html + "</div>"


raw_text = st.text_area(
    "Your text",
    height=280,
    placeholder="Paste your article, notes or scraped text here…",
)

run = st.button("✨ Generate script", type="primary", use_container_width=True, disabled=not raw_text.strip())

steps_box = st.empty()
steps_box.markdown(render_steps(st.session_state.get("status", {})), unsafe_allow_html=True)

if run:
    status = {"editor": "run"}
    steps_box.markdown(render_steps(status), unsafe_allow_html=True)
    result = {"raw_input": raw_text.strip()}
    order = [k for k, _, _ in STAGES]

    try:
        for chunk in pipeline_app.stream({"raw_input": raw_text.strip()}, stream_mode="updates"):
            for node, update in chunk.items():
                result.update(update)
                status[node] = "done"
                idx = order.index(node) + 1
                if idx < len(order):
                    status[order[idx]] = "run"
                steps_box.markdown(render_steps(status), unsafe_allow_html=True)
        st.session_state.result = result
        st.session_state.status = status
    except Exception as e:
        st.error(f"Something went wrong: {e}")

result = st.session_state.get("result")
if result:
    st.markdown("### Result")
    tab_final, tab_script, tab_clean = st.tabs(["🎯 Hinglish script", "📝 English script", "🧹 Cleaned text"])
    with tab_final:
        with st.container(border=True):
            st.markdown(result["final_output"].replace("\n", "  \n"))
        with st.expander("Copy"):
            st.code(result["final_output"], language=None, wrap_lines=True)
    with tab_script:
        with st.container(border=True):
            st.markdown(result["script_text"].replace("\n", "  \n"))
    with tab_clean:
        with st.container(border=True):
            st.markdown(result["edited_text"].replace("\n", "  \n"))