"""Streamlit UI: wiki chat + adaptive interview + rating history.

Spec: interview-agent-spec.md §3.4 (Phase 4). Text is the canonical record;
voice input is intentionally omitted.
"""
import json
import os
from pathlib import Path

import streamlit as st

from interview import (
    InterviewSession,
    LLMInterviewer,
    STYLES,
    load_skill_ratings,
    resolve_scope,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

st.set_page_config(page_title="AI Engineer Wiki", page_icon="📚", layout="wide")
st.title("📚 AI Engineer Wiki")

chat_tab, interview_tab, history_tab = st.tabs(["💬 Chat", "🎤 Interview", "📈 History"])


# ---------------------------------------------------------------------------
# Chat tab (existing agent loop)
# ---------------------------------------------------------------------------

with chat_tab:
    st.caption("Operate the compiled wiki: ingest, query, audit, generate Q&A, cheatsheets.")

    if "history" not in st.session_state:
        st.session_state.history = []

    with st.sidebar:
        if st.button("Clear chat history"):
            st.session_state.history = []
            st.rerun()
        st.markdown("### Example Commands")
        st.markdown(
            """
- `Ingest raw/transformers/attention-is-all-you-need.md`
- `What does the wiki say about KV cache?`
- `Run a full wiki audit`
- `Generate Q&A on LoRA`
- `Make a cheatsheet for positional encoding`
"""
        )

    for msg in st.session_state.history:
        if isinstance(msg.get("content"), str):
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    if query := st.chat_input("Ask a wiki operation or question..."):
        from agent import run_agent

        with st.chat_message("user"):
            st.markdown(query)
        with st.chat_message("assistant"):
            with st.spinner("Working..."):
                try:
                    answer, st.session_state.history = run_agent(
                        query, st.session_state.history)
                except Exception as exc:  # noqa: BLE001
                    answer = f"Error: {exc}"
            st.markdown(answer)


# ---------------------------------------------------------------------------
# Interview tab (OP-6 + OP-7)
# ---------------------------------------------------------------------------

def _start_session(topic, style, company, n_questions, level, weakest):
    try:
        scope = resolve_scope(topic=topic or None, company=company or None,
                              weakest=weakest)
    except ValueError as exc:
        st.error(str(exc))
        return
    interviewer = None
    if os.getenv("OPENAI_API_KEY"):
        try:
            interviewer = LLMInterviewer()
        except Exception as exc:  # noqa: BLE001
            st.warning(f"LLM unavailable — using the question bank only ({exc})")
    session = InterviewSession(
        scope=scope, style=style, max_questions=n_questions,
        interviewer=interviewer, start_level=level or None)
    question = session.next_question()
    if question is None:
        st.error("No questions available for this scope — generate Q&A for "
                 "the topic first, or set an OPENAI_API_KEY so questions can "
                 "be generated live.")
        return
    st.session_state.iv = {"session": session, "question": question,
                           "transcript_path": None, "report": None}


with interview_tab:
    iv = st.session_state.get("iv")

    if iv is None:
        st.subheader("Start a mock interview")
        col1, col2, col3 = st.columns(3)
        with col1:
            topic = st.text_input("Topic (wiki slug)", placeholder="kv-cache")
            weakest = st.checkbox("My weakest topics instead")
        with col2:
            style = st.selectbox("Style", STYLES, index=0)
            _COMPANY_PRESETS = ["(none)", "capital-one", "massmutual",
                                "fidelity", "exxon", "other…"]
            _preset = st.radio("Company", _COMPANY_PRESETS, horizontal=True,
                               index=0)
            company = (st.text_input("Company slug", placeholder="e.g. google")
                       if _preset == "other…"
                       else ("" if _preset == "(none)" else _preset))
        with col3:
            n_questions = st.slider("Questions", 1, 15, 5)
            level = st.select_slider(
                "Starting difficulty (0 = adaptive)", options=[0, 1, 2, 3, 4, 5])
        if st.button("Start interview", type="primary",
                     disabled=not (topic or weakest)):
            _start_session(topic.strip(), style, company.strip(),
                           n_questions, level, weakest)
            st.rerun()

    elif iv["transcript_path"] is None:
        session: InterviewSession = iv["session"]
        st.subheader(f"{session.scope.topic} — {session.style}")
        st.progress(len(session.turns) / session.max_questions,
                    text=f"Question {len(session.turns) + 1} of "
                         f"{session.max_questions} · "
                         f"{session.elapsed_minutes()} min elapsed")
        question = iv["question"]
        st.markdown(f"**Q{len(session.turns) + 1}.** {question.text}")
        answer = st.text_area("Your answer", key=f"answer_{len(session.turns)}",
                              height=160)
        col1, col2, col3 = st.columns([1, 1, 4])
        submitted = col1.button("Submit answer", type="primary")
        skipped = col2.button("Skip")
        ended = col3.button("End session")
        if submitted or skipped or ended:
            if submitted or skipped:
                session.record_answer(question, "" if skipped else answer.strip())
                iv["question"] = session.next_question()
            if ended or iv["question"] is None:
                if session.turns:
                    iv["transcript_path"] = session.save_transcript()
                else:
                    st.session_state.iv = None
            st.rerun()

    else:
        path = iv["transcript_path"]
        rel = path.relative_to(REPO_ROOT)
        st.success(f"Session complete — transcript saved to `{rel}`")

        if iv["report"] is None:
            col1, col2 = st.columns([1, 3])
            if col1.button("Assess now (OP-7)", type="primary",
                           disabled=not os.getenv("OPENAI_API_KEY")):
                from assess import assess_transcript

                with st.spinner("Grading against the wiki…"):
                    try:
                        result = assess_transcript(path)
                        iv["report"] = result.report_path.read_text(
                            encoding="utf-8")
                    except Exception as exc:  # noqa: BLE001
                        st.error(f"Assessment failed: {exc}")
                st.rerun()
            if not os.getenv("OPENAI_API_KEY"):
                col2.caption("Set OPENAI_API_KEY to grade this session.")
        else:
            st.markdown(iv["report"])

        if st.button("New interview"):
            st.session_state.iv = None
            st.rerun()


# ---------------------------------------------------------------------------
# History tab — ratings radar + trends (spec §3.4)
# ---------------------------------------------------------------------------

with history_tab:
    ratings = load_skill_ratings()
    concepts = ratings.get("concepts", {})

    if not concepts:
        st.info("No assessed sessions yet. Run an interview, then assess it.")
    else:
        st.subheader("Concept ratings")
        col1, col2 = st.columns([1, 1])

        with col1:
            try:
                import plotly.graph_objects as go

                names = list(concepts)
                values = [concepts[c].get("rating", 1200) for c in names]
                fig = go.Figure(go.Scatterpolar(
                    r=values + values[:1], theta=names + names[:1],
                    fill="toself", name="rating"))
                fig.update_layout(
                    polar={"radialaxis": {"range": [900, 1900]}},
                    showlegend=False, height=420, margin=dict(t=30, b=30))
                st.plotly_chart(fig, use_container_width=True)
            except ImportError:
                st.caption("`pip install plotly` for the radar chart.")

        with col2:
            rows = [{
                "concept": c,
                "rating": e.get("rating", 1200),
                "sessions": e.get("sessions", 0),
                "last assessed": e.get("last_assessed", "—"),
            } for c, e in sorted(concepts.items(),
                                 key=lambda kv: kv[1].get("rating", 1200))]
            st.dataframe(rows, use_container_width=True, hide_index=True)
            st.caption("Elo scale: 1000 ≈ level-1 recall · 1800 ≈ level-5 "
                       "open-ended design. Low session counts mean noisy "
                       "ratings (K=32 for the first 5 sessions).")

        st.subheader("Rating trends")
        trend_data = {c: e.get("trend", []) for c, e in concepts.items()
                      if e.get("trend")}
        if trend_data:
            import pandas as pd

            longest = max(len(t) for t in trend_data.values())
            frame = pd.DataFrame({
                c: ([None] * (longest - len(t)) + t)
                for c, t in trend_data.items()
            })
            frame.index.name = "assessment #"
            st.line_chart(frame)

        st.subheader("Assessment log")
        log_path = REPO_ROOT / "state" / "assessment_log.jsonl"
        if log_path.exists():
            entries = [json.loads(line)
                       for line in log_path.read_text(encoding="utf-8").splitlines()
                       if line.strip()]
            for e in reversed(entries[-20:]):
                with st.expander(
                        f"{e.get('date')} — {e.get('topic')} "
                        f"({e.get('style')}) · overall {e.get('overall')}/4"):
                    st.json(e)
