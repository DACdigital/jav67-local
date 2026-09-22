"""Streamlit UI for ranking emotion options against a phrase with a local Ollama embedding model."""

import pandas as pd
import streamlit as st

from main import (
    DEFAULT_DIMENSIONS,
    DEFAULT_HOST,
    DEFAULT_INSTRUCTION,
    DEFAULT_MODEL,
    DEFAULT_OPTIONS,
    DEFAULT_SHARPNESS,
    MatchRequest,
    Option,
    make_embedder,
    match,
    rescore,
)

st.set_page_config(page_title="Phrase vs options", layout="wide")


@st.cache_resource
def embedder(model: str, dimensions: int, host: str):
    return make_embedder(model, dimensions, host)


with st.sidebar:
    st.header("Ollama")
    model = st.text_input("Model", DEFAULT_MODEL)
    dimensions = st.number_input("Dimensions", min_value=1, value=DEFAULT_DIMENSIONS, step=1)
    host = st.text_input("Host", DEFAULT_HOST)

st.title("Phrase vs options")

phrase = st.text_input("Phrase", "I've had enough!")

use_instruction = st.checkbox("Prepend instruction to the phrase", value=True)
instruction = st.text_area(
    "Instruction",
    DEFAULT_INSTRUCTION,
    disabled=not use_instruction,
    height=70,
    help="Qwen3 embedding models expect the query as 'Instruct: <task>\\nQuery: <text>'.",
)

st.subheader("Options")
default_rows = pd.DataFrame([{"label": o.label, "description": o.description or ""} for o in DEFAULT_OPTIONS])
edited = st.data_editor(
    default_rows,
    num_rows="dynamic",
    width="stretch",
    column_config={
        "label": st.column_config.TextColumn("Label", required=True),
        "description": st.column_config.TextColumn("Description (optional)"),
    },
    key="options_editor",
)

options = [
    Option(label=str(row["label"]).strip(), description=(str(row["description"]).strip() or None))
    for _, row in edited.iterrows()
    if str(row["label"]).strip() and str(row["label"]) != "nan"
]

if st.button("Run", type="primary", disabled=not phrase.strip()):
    if len(options) < 2:
        st.error("Add at least two options.")
        st.stop()
    request = MatchRequest(
        phrase=phrase,
        options=options,
        instruction=instruction if use_instruction else None,
        sharpness=DEFAULT_SHARPNESS,
    )
    try:
        with st.spinner("Embedding..."):
            st.session_state["result"] = match(request, embedder(model, int(dimensions), host))
            st.session_state["query_text"] = request.query_text
    except Exception as exc:
        st.session_state.pop("result", None)
        st.error(f"Embedding failed: {exc}")
        st.stop()

if "result" in st.session_state:
    st.subheader("Results")

    sharpness = st.slider(
        "Sharpness",
        min_value=0.0,
        max_value=50.0,
        value=float(st.session_state.get("sharpness", DEFAULT_SHARPNESS)),
        step=0.5,
        help="Softmax inverse temperature over cosine similarities. 0 spreads confidence evenly; "
        "higher values concentrate it on the best option. Changing it does not re-embed.",
        key="sharpness",
    )
    result = rescore(st.session_state["result"], sharpness)

    left, right = st.columns([1, 1])

    with left:
        st.metric("Best match", result.best.option, delta=f"{result.best.confidence:.1%} confidence")
        scores = pd.DataFrame(
            [
                {
                    "option": s.option,
                    "confidence": s.confidence,
                    "scaled": s.scaled,
                    "similarity": s.cosine_similarity,
                    "distance": s.cosine_distance,
                    "embed ms": s.embed_ms,
                }
                for s in result.scores
            ]
        )
        st.dataframe(
            scores,
            hide_index=True,
            width="stretch",
            column_config={
                "confidence": st.column_config.ProgressColumn("confidence", min_value=0, max_value=1, format="percent"),
                "scaled": st.column_config.NumberColumn("scaled 0-1", format="%.3f"),
                "similarity": st.column_config.NumberColumn(format="%.4f"),
                "distance": st.column_config.NumberColumn(format="%.4f"),
                "embed ms": st.column_config.NumberColumn(format="%.1f"),
            },
        )
        st.caption(f"Margin between top two similarities: {result.margin:.4f}")
        st.bar_chart(scores.set_index("option")["confidence"], horizontal=True, x_label="confidence")

    with right:
        t = result.timings
        n = len(result.scores)
        st.markdown("**Timings**")
        timings = pd.DataFrame(
            [
                {"step": "phrase embedding", "ms": t.phrase_embed_ms},
                {"step": f"options embedding ({n} x {t.options_embed_ms / n:.1f} ms)", "ms": t.options_embed_ms},
                {"step": "scoring", "ms": t.scoring_ms},
                {"step": "total", "ms": t.total_ms},
            ]
        )
        st.dataframe(
            timings,
            hide_index=True,
            width="stretch",
            column_config={"ms": st.column_config.NumberColumn(format="%.2f")},
        )
        st.caption(f"Query text sent to the model:\n\n```\n{st.session_state['query_text']}\n```")

    with st.expander("Raw result (JSON)"):
        st.json(result.model_dump())
