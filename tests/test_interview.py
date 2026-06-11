"""Phase 1 tests for OP-6 INTERVIEW (interview-agent-spec.md §3.1, §4.1, §4.2).

Covers: scope resolution, Elo-band difficulty picker (±150), QA-bank parsing,
question dedup within a session, transcript frontmatter validity, and the
no-overwrite rule for raw/interviews/.

All tests run offline: no LLMInterviewer is constructed.
"""
import re

import pytest

from interview import (
    BAND,
    DEFAULT_RATING,
    LEVEL_RATINGS,
    InterviewSession,
    Question,
    concept_rating,
    eligible_levels,
    load_skill_ratings,
    normalize_question,
    parse_interview_request,
    parse_qa_bank,
    pick_level,
    resolve_scope,
)


# ---------------------------------------------------------------------------
# Ratings + difficulty picker
# ---------------------------------------------------------------------------

def test_load_skill_ratings_missing_file_degrades_to_defaults(tmp_path):
    ratings = load_skill_ratings(tmp_path / "nope.json")
    assert ratings["concepts"] == {}
    assert concept_rating(ratings, "kv-cache") == DEFAULT_RATING


def test_load_skill_ratings_corrupt_file_degrades_to_defaults(tmp_path):
    bad = tmp_path / "skill_ratings.json"
    bad.write_text("{not json", encoding="utf-8")
    assert load_skill_ratings(bad)["concepts"] == {}


@pytest.mark.parametrize("rating", range(850, 1951, 25))
def test_pick_level_stays_within_band(rating):
    level = pick_level(rating)
    assert abs(LEVEL_RATINGS[level] - rating) <= BAND


def test_pick_level_clamps_outside_band():
    assert pick_level(600) == 1
    assert pick_level(2400) == 5


def test_pick_level_default_rating_is_level_2():
    # Fresh concept (1200) must land on level 2 — the only level in band.
    assert pick_level(DEFAULT_RATING) == 2
    assert eligible_levels(DEFAULT_RATING) == [2]


def test_pick_level_tie_breaks_harder():
    # 1300 is equidistant from levels 2 (1200) and 3 (1400).
    assert pick_level(1300) == 3


# ---------------------------------------------------------------------------
# Scope resolution
# ---------------------------------------------------------------------------

def test_resolve_scope_topic_loads_concept_and_qa_pages():
    scope = resolve_scope(topic="kv-cache")
    assert "kv-cache" in scope.concepts
    assert "kv-cache" in scope.concept_pages
    assert scope.concept_pages["kv-cache"].lstrip().startswith("---")
    # transformers-qa.md links [[kv-cache]] in its related frontmatter
    assert any("transformers-qa" in p for p in scope.qa_pages)


def test_resolve_scope_includes_related_concepts():
    scope = resolve_scope(topic="kv-cache")
    # kv-cache's frontmatter relates it to attention-mechanism et al.
    assert len(scope.concepts) > 1
    for slug in scope.concepts:
        assert slug in scope.concept_pages


def test_resolve_scope_unknown_topic_raises():
    with pytest.raises(ValueError, match="No wiki coverage"):
        resolve_scope(topic="definitely-not-a-real-topic-xyz")


def test_resolve_scope_weakest_requires_ratings():
    with pytest.raises(ValueError, match="No skill ratings"):
        resolve_scope(weakest=True, ratings={"version": 1, "concepts": {}})


def test_resolve_scope_weakest_picks_lowest_rated():
    ratings = {
        "version": 1,
        "concepts": {
            "kv-cache": {"rating": 1500},
            "flash-attention": {"rating": 1100},
            "attention-mechanism": {"rating": 1300},
        },
    }
    scope = resolve_scope(weakest=True, n_weakest=2, ratings=ratings)
    assert scope.concepts[:2] == ["flash-attention", "kv-cache"] or \
        set(scope.concepts[:2]) == {"flash-attention", "attention-mechanism"}
    assert "flash-attention" == scope.concepts[0]


def test_resolve_scope_missing_company_is_graceful():
    scope = resolve_scope(topic="kv-cache", company="capital-one")
    assert scope.company == "capital-one"
    assert scope.company_page is None  # wiki/companies/ does not exist


# ---------------------------------------------------------------------------
# QA bank parsing
# ---------------------------------------------------------------------------

def test_parse_qa_bank_extracts_leveled_questions():
    scope = resolve_scope(topic="kv-cache")
    bank = parse_qa_bank(scope)
    assert bank, "expected questions parsed from wiki/qa/"
    assert all(q.level in (2, 3, 4) for q in bank)
    assert all(q.source.startswith("wiki/qa/") for q in bank)
    assert not any(q.generated for q in bank)
    # The known KV-cache question is attributed to the kv-cache concept.
    kv_questions = [q for q in bank if q.concept == "kv-cache"]
    assert kv_questions


# ---------------------------------------------------------------------------
# Session: dedup + adaptivity
# ---------------------------------------------------------------------------

def _session(**kwargs) -> InterviewSession:
    scope = resolve_scope(topic="kv-cache")
    defaults = dict(style="drill", max_questions=5,
                    ratings={"version": 1, "concepts": {}})
    defaults.update(kwargs)
    return InterviewSession(scope=scope, **defaults)


def test_session_never_repeats_a_question():
    session = _session(max_questions=50)
    seen = set()
    while True:
        q = session.next_question()
        if q is None:
            break
        key = normalize_question(q.text)
        assert key not in seen, f"duplicate question: {q.text}"
        seen.add(key)
        session.record_answer(q, "some answer")
    assert seen, "session asked no questions at all"


def test_first_question_is_explicitly_about_the_topic():
    # Generic bank questions fall back to the topic concept by attribution;
    # they must not outrank questions that actually mention the concept.
    session = _session(max_questions=1)
    q = session.next_question()
    assert q is not None
    assert q.concept == "kv-cache"
    assert q.explicit, f"expected an explicit kv-cache question, got: {q.text}"
    assert "kv cache" in q.text.lower() or "kv-cache" in q.text.lower()


def test_session_stops_at_max_questions():
    session = _session(max_questions=3)
    count = 0
    while (q := session.next_question()) is not None:
        session.record_answer(q, "answer")
        count += 1
    assert count <= 3


def test_session_escalates_on_correct_and_deescalates_on_wrong():
    session = _session()
    concept = session.scope.concepts[0]
    start = session.working_levels[concept]
    session._adapt(concept, "correct")
    assert session.working_levels[concept] == min(5, start + 1)
    session._adapt(concept, "wrong")
    session._adapt(concept, "wrong")
    assert session.working_levels[concept] == max(1, start - 1)
    # unknown/partial verdicts leave the level unchanged
    level = session.working_levels[concept]
    session._adapt(concept, "unknown")
    assert session.working_levels[concept] == level


def test_session_levels_seeded_within_band_of_ratings():
    ratings = {"version": 1, "concepts": {"kv-cache": {"rating": 1620}}}
    session = _session(ratings=ratings)
    assert abs(LEVEL_RATINGS[session.working_levels["kv-cache"]] - 1620) <= BAND


# ---------------------------------------------------------------------------
# Transcript
# ---------------------------------------------------------------------------

REQUIRED_FM = ("type:", "date:", "topic:", "style:", "duration_min:",
               "questions:", "concepts_touched:", "assessed:")


def _run_scripted_session(tmp_path, n=5):
    session = _session(max_questions=n)
    while (q := session.next_question()) is not None:
        session.record_answer(q, f"scripted answer about {q.concept}")
    return session, session.save_transcript(tmp_path)


def test_transcript_frontmatter_is_valid(tmp_path):
    session, path = _run_scripted_session(tmp_path)
    text = path.read_text(encoding="utf-8")
    assert text.startswith("---\n")
    fm_end = text.find("\n---", 4)
    fm = text[4:fm_end]
    for key in REQUIRED_FM:
        assert re.search(rf"^{key}", fm, flags=re.MULTILINE), f"missing {key}"
    assert "type: interview-transcript" in fm
    assert "assessed: false" in fm
    assert re.search(r"^date: \d{4}-\d{2}-\d{2}$", fm, flags=re.MULTILINE)
    m = re.search(r"^questions: (\d+)$", fm, flags=re.MULTILINE)
    assert int(m.group(1)) == len(session.turns)
    m = re.search(r"^concepts_touched: \[(.*)\]$", fm, flags=re.MULTILINE)
    touched = [c.strip() for c in m.group(1).split(",") if c.strip()]
    assert touched == session.concepts_touched()
    # company key omitted when no company was set
    assert "company:" not in fm


def test_transcript_filename_pattern_and_no_overwrite(tmp_path):
    session, first = _run_scripted_session(tmp_path, n=2)
    assert re.match(r"^\d{4}-\d{2}-\d{2}-kv-cache-drill\.md$", first.name)
    second = session.save_transcript(tmp_path)
    assert second != first, "raw transcripts must never be overwritten"
    assert second.name.endswith("-2.md")
    assert first.exists() and second.exists()


def test_transcript_body_has_question_blocks(tmp_path):
    session, path = _run_scripted_session(tmp_path, n=3)
    text = path.read_text(encoding="utf-8")
    blocks = re.findall(r"^## Q(\d+) \(concept: ([a-z0-9-]+), level: (\d)",
                        text, flags=re.MULTILINE)
    assert len(blocks) == len(session.turns)
    assert text.count("**Candidate:**") >= len(session.turns)


def test_generated_questions_are_tagged(tmp_path):
    session = _session(max_questions=1)
    q = Question(concept="kv-cache", level=5, text="Design X under constraint Y.",
                 source="generated", generated=True)
    session.record_answer(q, "an answer")
    text = session.save_transcript(tmp_path).read_text(encoding="utf-8")
    assert "generated:true" in text


# ---------------------------------------------------------------------------
# Request parsing
# ---------------------------------------------------------------------------

def test_parse_interview_request_full_form():
    req = parse_interview_request(
        "Interview me on transformers, 30 minutes, company=capital-one")
    assert req["topic"] == "transformers"
    assert req["duration_min"] == 30
    assert req["company"] == "capital-one"


def test_parse_interview_request_weakest():
    req = parse_interview_request("Interview me on my weakest topics")
    assert req["weakest"] is True
    assert req["topic"] is None


def test_parse_interview_request_system_design_hard():
    req = parse_interview_request("Run a system design interview, hard mode")
    assert req["style"] == "system-design"
    assert req["start_level"] == 4


def test_parse_interview_request_drill_with_count():
    req = parse_interview_request("drill me on kv-cache, 5 questions")
    assert req["style"] == "drill"
    assert req["topic"] == "kv-cache"
    assert req["max_questions"] == 5


def test_parse_interview_request_ignores_non_interview_text():
    assert parse_interview_request("what does the wiki say about rope?") is None
