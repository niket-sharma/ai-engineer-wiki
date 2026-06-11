"""Phase 2 tests for OP-7 ASSESS (interview-agent-spec.md §3.2, §4.1, §4.3, §4.4).

Covers: Elo math golden cases, transcript parsing, ratings updates (K-factor
schedule, trend, sessions), maintenance-queue task generation (score ≤ 2 and
wiki_gap), JSONL append integrity, report rendering with the study-plan
section, and the phase done-when: assessing a fixture transcript produces a
report, updated ratings, and ≥ 1 queued task — all offline via a fake grader.
"""
import json
import re

import pytest

from assess import (
    Grade,
    K_EARLY,
    K_LATE,
    append_assessment_log,
    assess_transcript,
    elo_update,
    expected_score,
    find_unassessed,
    grade_transcript,
    k_factor,
    mark_assessed,
    outcome_for,
    parse_transcript,
    queue_follow_ups,
    render_report,
    update_ratings,
)

FIXTURE = """---
type: interview-transcript
date: 2026-06-11
topic: kv-cache
style: drill
duration_min: 12
questions: 3
concepts_touched: [kv-cache, flash-attention]
assessed: false
---

# Interview Transcript — kv-cache (drill)

## Q1 (concept: kv-cache, level: 2, source: wiki/qa/transformers-qa.md)
**Interviewer:** What is a KV cache and why is it important?

**Candidate:** It stores K and V projections so decoding is O(n) instead of O(n^2).

## Q2 (concept: kv-cache, level: 3, generated:true)
**Interviewer:** Walk me through the memory cost of a KV cache.

**Candidate:** I think RoPE is applied to V so the cache shrinks.

**Interviewer (follow-up 1):** Why would that shrink anything?

**Candidate:** (no answer)

## Q3 (concept: flash-attention, level: 3, source: wiki/qa/transformers-qa.md)
**Interviewer:** What problem does Flash Attention solve?

**Candidate:** (no answer)
"""


class FakeGrader:
    """Deterministic grader: Q1 strong, Q2 weak + misconception + wiki_gap,
    Q3 unanswered."""

    def grade(self, turn, rubric_page):
        if turn.number == 1:
            return {"score": 4, "gaps": [], "misconceptions": [],
                    "wiki_gap": False}
        if turn.number == 2:
            return {"score": 1,
                    "gaps": ["missed cache size formula (Technical Detail)"],
                    "misconceptions": ["claimed RoPE is applied to V"],
                    "wiki_gap": True}
        return {"score": 3, "gaps": [], "misconceptions": [], "wiki_gap": False}


@pytest.fixture
def fixture_transcript(tmp_path):
    path = tmp_path / "2026-06-11-kv-cache-drill.md"
    path.write_text(FIXTURE, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Elo golden cases
# ---------------------------------------------------------------------------

def test_expected_score_symmetry():
    assert expected_score(1200, 1200) == 0.5
    assert expected_score(1200, 1000) + expected_score(1000, 1200) == pytest.approx(1.0)


def test_elo_win_at_equal_rating_gains_half_k():
    assert elo_update(1200, 1200, 1.0, 32) == 1216


def test_elo_loss_to_stronger_opponent_costs_little():
    # E = 1/(1+10^0.5) ≈ 0.2403 → 1200 + 32·(0 − 0.2403) ≈ 1192
    assert elo_update(1200, 1400, 0.0, 32) == 1192


def test_elo_draw_against_weaker_opponent_loses_points():
    # E ≈ 0.7597 → 1200 + 32·(0.5 − 0.7597) ≈ 1192
    assert elo_update(1200, 1000, 0.5, 32) == 1192


def test_outcome_mapping():
    assert outcome_for(0) == 0.0 and outcome_for(1) == 0.0
    assert outcome_for(2) == 0.5
    assert outcome_for(3) == 1.0 and outcome_for(4) == 1.0


def test_k_factor_schedule():
    assert all(k_factor(s) == K_EARLY for s in range(5))
    assert k_factor(5) == K_LATE and k_factor(50) == K_LATE


# ---------------------------------------------------------------------------
# Transcript parsing
# ---------------------------------------------------------------------------

def test_parse_transcript_meta_and_turns(fixture_transcript):
    t = parse_transcript(fixture_transcript)
    assert t.meta["topic"] == "kv-cache"
    assert t.meta["assessed"] is False
    assert len(t.turns) == 3
    assert t.turns[0].concept == "kv-cache" and t.turns[0].level == 2
    assert t.turns[0].question.startswith("What is a KV cache")
    assert "O(n)" in t.turns[0].answer
    assert t.turns[1].follow_ups == [("Why would that shrink anything?", "")]
    assert t.turns[2].answer == ""  # "(no answer)" normalizes to empty


def test_unanswered_question_scores_zero_regardless_of_grader(fixture_transcript):
    t = parse_transcript(fixture_transcript)
    grades = grade_transcript(t, {}, FakeGrader())
    assert grades[2].score == 0  # FakeGrader said 3; no answer forces 0


def test_mark_assessed_flips_only_the_flag(fixture_transcript):
    mark_assessed(fixture_transcript)
    text = fixture_transcript.read_text(encoding="utf-8")
    assert "assessed: true" in text
    assert "assessed: false" not in text
    assert parse_transcript(fixture_transcript).meta["assessed"] is True
    assert "What is a KV cache" in text  # body untouched


def test_find_unassessed_skips_assessed(tmp_path, fixture_transcript):
    assert find_unassessed(tmp_path) == fixture_transcript
    mark_assessed(fixture_transcript)
    assert find_unassessed(tmp_path) is None


# ---------------------------------------------------------------------------
# Ratings updates
# ---------------------------------------------------------------------------

def _grades():
    return [
        Grade(concept="kv-cache", level=2, question="q1", score=4),
        Grade(concept="kv-cache", level=3, question="q2", score=1,
              misconceptions=["claimed RoPE is applied to V"], wiki_gap=True),
        Grade(concept="flash-attention", level=3, question="q3", score=0),
    ]


def test_update_ratings_fresh_concept_golden():
    ratings = {"version": 1, "concepts": {}}
    update_ratings(ratings, _grades(), "2026-06-11")
    kv = ratings["concepts"]["kv-cache"]
    # 1200 →(win vs 1200, K32)→ 1216 →(loss vs 1400, K32)→ round(1216+32·(0−E))
    step1 = 1216
    step2 = elo_update(step1, 1400, 0.0, 32)
    assert kv["rating"] == step2
    assert kv["sessions"] == 1
    assert kv["last_assessed"] == "2026-06-11"
    assert kv["trend"] == [step2]
    assert kv["wiki_page"] == "wiki/concepts/kv-cache.md"
    fa = ratings["concepts"]["flash-attention"]
    assert fa["rating"] == elo_update(1200, 1400, 0.0, 32)


def test_update_ratings_uses_low_k_after_five_sessions():
    ratings = {"version": 1, "concepts": {"kv-cache": {
        "rating": 1400, "sessions": 6, "last_assessed": "2026-06-01",
        "trend": [1400], "wiki_page": "wiki/concepts/kv-cache.md"}}}
    update_ratings(ratings, [Grade(concept="kv-cache", level=3,
                                   question="q", score=4)], "2026-06-11")
    # draw-expected vs equal opponent: 1400 + 16·(1−0.5) = 1408
    assert ratings["concepts"]["kv-cache"]["rating"] == 1408
    assert ratings["concepts"]["kv-cache"]["sessions"] == 7


# ---------------------------------------------------------------------------
# Maintenance queue
# ---------------------------------------------------------------------------

def test_queue_generates_tasks_for_weak_scores_and_wiki_gaps(tmp_path):
    qpath = tmp_path / "maintenance_queue.json"
    new = queue_follow_ups(_grades(), "2026-06-11", qpath)
    types = sorted(t["type"] for t in new)
    # kv-cache (score 1) + flash-attention (score 0) → 2 generate_qa; 1 wiki_gap
    assert types == ["generate_qa", "generate_qa", "wiki_gap"]
    saved = json.loads(qpath.read_text())
    assert len(saved["tasks"]) == 3
    ids = [t["id"] for t in saved["tasks"]]
    assert len(ids) == len(set(ids))
    assert all(t["status"] == "pending" for t in saved["tasks"])
    gap = next(t for t in saved["tasks"] if t["type"] == "wiki_gap")
    assert gap["page"] == "wiki/concepts/kv-cache.md"


def test_queue_ids_do_not_collide_with_existing(tmp_path):
    qpath = tmp_path / "maintenance_queue.json"
    qpath.write_text(json.dumps({"tasks": [
        {"id": "q-001", "type": "generate_qa", "concept": "x",
         "status": "pending"}]}), encoding="utf-8")
    new = queue_follow_ups(_grades(), "2026-06-11", qpath)
    all_ids = [t["id"] for t in json.loads(qpath.read_text())["tasks"]]
    assert len(all_ids) == len(set(all_ids)) == 4
    assert all(t["id"] != "q-001" for t in new)


def test_queue_no_tasks_for_strong_session(tmp_path):
    qpath = tmp_path / "maintenance_queue.json"
    strong = [Grade(concept="kv-cache", level=3, question="q", score=4)]
    assert queue_follow_ups(strong, "2026-06-11", qpath) == []


# ---------------------------------------------------------------------------
# JSONL + report
# ---------------------------------------------------------------------------

def test_assessment_log_append_integrity(tmp_path):
    path = tmp_path / "assessment_log.jsonl"
    append_assessment_log({"date": "2026-06-11", "overall": 2.5}, path)
    append_assessment_log({"date": "2026-06-12", "overall": 3.0}, path)
    lines = path.read_text().splitlines()
    assert len(lines) == 2
    parsed = [json.loads(line) for line in lines]
    assert parsed[0]["date"] == "2026-06-11"
    assert parsed[1]["overall"] == 3.0


def test_report_renders_required_sections(fixture_transcript):
    t = parse_transcript(fixture_transcript)
    grades = _grades()
    ratings = update_ratings({"version": 1, "concepts": {}}, grades, "2026-06-11")
    report = render_report(t, grades, ratings, "2026-06-11")
    for section in ("## Scores", "## Top weaknesses", "## Top strengths",
                    "## ⚠️ Misconceptions", "## 7-day micro study plan"):
        assert section in report, f"missing section: {section}"
    assert "claimed RoPE is applied to V" in report
    assert "[[kv-cache]]" in report  # study plan links to wiki pages
    assert len(re.findall(r"^- \*\*Day \d\*\*", report, flags=re.MULTILINE)) == 7
    assert "Wiki gaps" in report


# ---------------------------------------------------------------------------
# Done-when: full fixture assessment
# ---------------------------------------------------------------------------

def test_assess_fixture_end_to_end(tmp_path, fixture_transcript):
    state = tmp_path / "state"
    reports = tmp_path / "reports"
    result = assess_transcript(fixture_transcript, grader=FakeGrader(),
                               state_dir=state, reports_dir=reports,
                               log_to_wiki=False)

    # report written and human-readable
    assert result.report_path.exists()
    assert "7-day micro study plan" in result.report_path.read_text()

    # ratings updated
    ratings = json.loads((state / "skill_ratings.json").read_text())
    assert ratings["concepts"]["kv-cache"]["sessions"] == 1
    assert ratings["updated"]

    # ≥ 1 maintenance task queued
    queue = json.loads((state / "maintenance_queue.json").read_text())
    assert len(queue["tasks"]) >= 1

    # machine-readable log entry appended (spec §4.4 shape)
    entry = json.loads((state / "assessment_log.jsonl").read_text().splitlines()[-1])
    assert {"date", "topic", "style", "overall", "per_concept",
            "misconceptions", "report"} <= set(entry)
    assert entry["topic"] == "kv-cache"

    # transcript flipped to assessed and refuses double-grading
    assert parse_transcript(fixture_transcript).meta["assessed"] is True
    with pytest.raises(ValueError, match="already assessed"):
        assess_transcript(fixture_transcript, grader=FakeGrader(),
                          state_dir=state, reports_dir=reports,
                          log_to_wiki=False)
