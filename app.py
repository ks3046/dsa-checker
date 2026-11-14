import os

import streamlit as st

try:
    if "GEMINI_API_KEY" in st.secrets:
        os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]
except Exception:
    pass
from rag import Retriever, answer  # noqa: E402

MAX_QUESTIONS_PER_SESSION = 10  # basic cost guard; also set a spend limit in your API console

st.set_page_config(page_title="DSA Doubt Solver", page_icon="📚")
st.title("📚 DSA Doubt Solver")
st.caption(
    "Answers come only from a small set of Wikipedia articles on data structures and "
    "algorithms. If the answer isn't in them, it says so."
)


@st.cache_resource
def get_retriever() -> Retriever:
    return Retriever()


if "count" not in st.session_state:
    st.session_state.count = 0

question = st.text_input("Ask a DSA question", placeholder="How does binary search work?")

if st.button("Ask") and question.strip():
    if st.session_state.count >= MAX_QUESTIONS_PER_SESSION:
        st.warning("Question limit reached for this session. Refresh to start again.")
    else:
        st.session_state.count += 1
        with st.spinner("Searching sources..."):
            result = answer(question.strip(), get_retriever())
        if result["refused"]:
            st.info(result["answer"])
        else:
            st.markdown(result["answer"])
        if result["sources"]:
            with st.expander("Sources used"):
                for s in result["sources"]:
                    st.markdown(f"**[{s['n']}] {s['title']}** (score {s['score']:.1f})")
                    st.write(s["text"])

st.caption(f"Questions used: {st.session_state.count}/{MAX_QUESTIONS_PER_SESSION}")
