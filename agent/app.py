import streamlit as st

from agent import run_agent

st.set_page_config(page_title="AI Engineer Wiki", page_icon="📚")
st.title("📚 AI Engineer Wiki")
st.caption("Operate the compiled wiki: ingest, query, audit, generate, company prep, cheatsheets.")

if "history" not in st.session_state:
    st.session_state.history = []

if st.sidebar.button("Clear History"):
    st.session_state.history = []
    st.rerun()

st.sidebar.markdown("### Example Commands")
st.sidebar.markdown(
    """
- `Ingest raw/transformers/attention-is-all-you-need.md`
- `What does the wiki say about KV cache?`
- `Run a full wiki audit`
- `Generate Q&A on LoRA`
- `Update company prep for Capital One`
- `Make a cheatsheet for positional encoding`
"""
)

for msg in st.session_state.history:
    if isinstance(msg.get("content"), str):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

if query := st.chat_input("Ask a wiki operation or question..."):
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Working..."):
            try:
                answer, st.session_state.history = run_agent(query, st.session_state.history)
            except Exception as exc:  # noqa: BLE001
                answer = f"Error: {exc}"
        st.markdown(answer)
