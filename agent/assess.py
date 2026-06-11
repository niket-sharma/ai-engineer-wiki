"""OP-7 ASSESS — grade a completed interview session against the wiki.

Spec: interview-agent-spec.md §3.2, §4.1, §4.3, §4.4.

The concept page IS the rubric (full-page router pattern — whole pages, never
chunks). Deterministic parts (transcript parsing, Elo math, ratings/queue/log
writers, report rendering) are unit-testable offline; the LLM grader is
injected so tests can substitute a fake.

State contract: ASSESS is the ONLY writer of state/skill_ratings.json. It also
appends to state/assessment_log.jsonl, queues tasks in
state/maintenance_queue.json, writes wiki/reports/, flips `assessed: true` in
the transcript frontmatter (the single permitted raw/ mutation — one field, in
a file this system itself created), and logs to wiki/log.md.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from interview import LEVEL_RATINGS, _load_prompt, load_skill_ratings

REPO_ROOT = Path(__file__).resolve().parent.parent
WIKI_DIR = REPO_ROOT / "wiki"
STATE_DIR = REPO_ROOT / "state"
REPORTS_DIR = WIKI_DIR / "reports"
INTERVIEWS_DIR = REPO_ROOT / "raw" / "interviews"

# Elo (spec §4.1): K=32 for a concept's first 5 sessions, then 16.
K_EARLY = 32
K_LATE = 16
EARLY_SESSIONS = 5
DEFAULT_RATING = 1200


# ---------------------------------------------------------------------------
# Elo math
# ---------------------------------------------------------------------------

def expected_score(rating: float, opponent: float) -> float:
    return 1.0 / (1.0 + 10 ** ((opponent - rating) / 400.0))


def outcome_for(score: int) -> float:
    """Score 0–4 → Elo outcome: 0–1 loss, 2 draw, 3–4 win."""
    if score <= 1:
        return 0.0
    if score == 2:
        return 0.5
    return 1.0


def k_factor(sessions: int) -> int:
    return K_EARLY if sessions < EARLY_SESSIONS else K_LATE


def elo_update(rating: float, opponent: float, outcome: float, k: int) -> int:
    return round(rating + k * (outcome - expected_score(rating, opponent)))


# ---------------------------------------------------------------------------
# Transcript parsing
# ---------------------------------------------------------------------------

@dataclass
class TranscriptTurn:
    number: int
    concept: str
    level: int
    question: str
    answer: str
    follow_ups: list[tuple[str, str]] = field(default_factory=list)


@dataclass
class Transcript:
    path: Path
    meta: dict
    turns: list[TranscriptTurn]


def parse_transcript(path: Path) -> Transcript:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        raise ValueError(f"{path} has no frontmatter")
    fm_end = text.find("\n---", 4)
    meta: dict = {}
    for line in text[4:fm_end].splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip()
    meta["assessed"] = meta.get("assessed", "false").lower() == "true"

    turns: list[TranscriptTurn] = []
    current: TranscriptTurn | None = None
    pending_followup: str | None = None
    for line in text[fm_end + 4:].splitlines():
        header = re.match(
            r"^## Q(\d+) \(concept: ([a-z0-9-]+), level: (\d)", line)
        if header:
            current = TranscriptTurn(
                number=int(header.group(1)),
                concept=header.group(2),
                level=int(header.group(3)),
                question="", answer="",
            )
            turns.append(current)
            pending_followup = None
            continue
        if current is None:
            continue
        m = re.match(r"^\*\*Interviewer \(follow-up \d+\):\*\* (.*)$", line)
        if m:
            pending_followup = m.group(1)
            continue
        m = re.match(r"^\*\*Interviewer:\*\* (.*)$", line)
        if m:
            current.question = m.group(1)
            continue
        m = re.match(r"^\*\*Candidate:\*\* (.*)$", line)
        if m:
            answer = "" if m.group(1) == "(no answer)" else m.group(1)
            if pending_followup is not None:
                current.follow_ups.append((pending_followup, answer))
                pending_followup = None
            else:
                current.answer = answer
    return Transcript(path=path, meta=meta, turns=turns)


def mark_assessed(path: Path) -> None:
    """Flip `assessed: false` → `assessed: true` in transcript frontmatter."""
    text = path.read_text(encoding="utf-8")
    fm_end = text.find("\n---", 4)
    head, tail = text[:fm_end], text[fm_end:]
    head = re.sub(r"^assessed: false$", "assessed: true", head,
                  flags=re.MULTILINE)
    path.write_text(head + tail, encoding="utf-8")


def find_unassessed(interviews_dir: Path | None = None) -> Path | None:
    d = interviews_dir or INTERVIEWS_DIR
    if not d.is_dir():
        return None
    candidates = []
    for p in sorted(d.glob("*.md")):
        text = p.read_text(encoding="utf-8")
        if re.search(r"^assessed: false$", text[:600], flags=re.MULTILINE):
            candidates.append(p)
    return candidates[-1] if candidates else None


# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------

@dataclass
class Grade:
    concept: str
    level: int
    question: str
    score: int                      # 0–4
    gaps: list[str] = field(default_factory=list)
    misconceptions: list[str] = field(default_factory=list)
    wiki_gap: bool = False


def _strip_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    return re.sub(r"\s*```$", "", text).strip()


class LLMGrader:
    """Grades one Q/A pair against the full concept page (the rubric)."""

    def __init__(self, model: str | None = None):
        import openai

        self.client = openai.OpenAI()
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o")
        self.system = _load_prompt("grader_system.md")

    def grade(self, turn: TranscriptTurn, rubric_page: str) -> dict:
        follow_ups = "\n".join(
            f"Follow-up: {fq}\nCandidate: {fa or '(no answer)'}"
            for fq, fa in turn.follow_ups
        )
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=800,
            messages=[
                {"role": "system", "content": self.system},
                {"role": "user", "content": (
                    f"Question (difficulty level {turn.level}): {turn.question}\n"
                    f"Candidate answer: {turn.answer or '(no answer)'}\n"
                    f"{follow_ups}\n\n"
                    f"Rubric — the full wiki page for [{turn.concept}]:\n"
                    f"---\n{rubric_page[:16000]}\n---\n\n"
                    "Respond ONLY in JSON."
                )},
            ],
        )
        raw = _strip_fences(response.choices[0].message.content or "{}")
        data = json.loads(raw)
        return {
            "score": max(0, min(4, int(data.get("score", 0)))),
            "gaps": [str(g) for g in data.get("gaps", [])],
            "misconceptions": [str(m) for m in data.get("misconceptions", [])],
            "wiki_gap": bool(data.get("wiki_gap", False)),
        }


def grade_transcript(transcript: Transcript, concept_pages: dict[str, str],
                     grader) -> list[Grade]:
    grades = []
    for turn in transcript.turns:
        rubric = concept_pages.get(turn.concept, "")
        result = grader.grade(turn, rubric)
        if not turn.answer and not turn.follow_ups:
            result["score"] = 0  # no answer is a 0 regardless of the grader
        grades.append(Grade(concept=turn.concept, level=turn.level,
                            question=turn.question, **result))
    return grades


def load_concept_pages(transcript: Transcript) -> dict[str, str]:
    pages: dict[str, str] = {}
    for turn in transcript.turns:
        if turn.concept in pages:
            continue
        for sub in ("concepts", "system-design"):
            path = WIKI_DIR / sub / f"{turn.concept}.md"
            if path.exists():
                pages[turn.concept] = path.read_text(encoding="utf-8")
                break
    return pages


# ---------------------------------------------------------------------------
# Ratings update (sole writer of skill_ratings.json)
# ---------------------------------------------------------------------------

def update_ratings(ratings: dict, grades: list[Grade], date: str) -> dict:
    """Sequential per-question Elo updates; one session increment per concept.

    K is fixed by the concept's session count entering this assessment.
    """
    concepts = ratings.setdefault("concepts", {})
    by_concept: dict[str, list[Grade]] = {}
    for g in grades:
        by_concept.setdefault(g.concept, []).append(g)

    for concept, concept_grades in by_concept.items():
        entry = concepts.setdefault(concept, {
            "rating": DEFAULT_RATING, "sessions": 0, "last_assessed": None,
            "trend": [], "wiki_page": f"wiki/concepts/{concept}.md",
        })
        k = k_factor(entry["sessions"])
        rating = entry["rating"]
        for g in concept_grades:
            opponent = LEVEL_RATINGS.get(g.level, DEFAULT_RATING)
            rating = elo_update(rating, opponent, outcome_for(g.score), k)
        entry["rating"] = rating
        entry["sessions"] += 1
        entry["last_assessed"] = date
        entry["trend"] = (entry.get("trend", []) + [rating])[-20:]

    ratings["updated"] = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
    return ratings


def save_ratings(ratings: dict, path: Path | None = None) -> Path:
    path = path or (STATE_DIR / "skill_ratings.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ratings, indent=2) + "\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Maintenance queue (spec §4.3)
# ---------------------------------------------------------------------------

def queue_follow_ups(grades: list[Grade], date: str,
                     queue_path: Path | None = None) -> list[dict]:
    """score ≤ 2 → generate_qa task; wiki_gap → wiki_gap task. Returns the
    newly queued tasks."""
    queue_path = queue_path or (STATE_DIR / "maintenance_queue.json")
    try:
        queue = json.loads(queue_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        queue = {"tasks": []}
    tasks = queue.setdefault("tasks", [])

    existing_ids = {t.get("id", "") for t in tasks}
    counter = 1

    def next_id() -> str:
        nonlocal counter
        while f"q-{counter:03d}" in existing_ids:
            counter += 1
        new = f"q-{counter:03d}"
        existing_ids.add(new)
        return new

    new_tasks: list[dict] = []
    weak_concepts: dict[str, int] = {}
    for g in grades:
        if g.score <= 2:
            weak_concepts[g.concept] = min(
                weak_concepts.get(g.concept, 4), g.score)
    for concept, worst in sorted(weak_concepts.items()):
        new_tasks.append({
            "id": next_id(), "type": "generate_qa", "concept": concept,
            "difficulty": 4,
            "reason": f"scored {worst}/4 on {date}", "status": "pending",
        })
    seen_gap_pages = set()
    for g in grades:
        if not g.wiki_gap:
            continue
        page = f"wiki/concepts/{g.concept}.md"
        if page in seen_gap_pages:
            continue
        seen_gap_pages.add(page)
        new_tasks.append({
            "id": next_id(), "type": "wiki_gap", "page": page,
            "section": (g.gaps[0] if g.gaps else "depth insufficient to grade"),
            "reason": f"rubric too thin to grade on {date}", "status": "pending",
        })

    tasks.extend(new_tasks)
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    queue_path.write_text(json.dumps(queue, indent=2) + "\n", encoding="utf-8")
    return new_tasks


# ---------------------------------------------------------------------------
# Report + logs
# ---------------------------------------------------------------------------

def _overall(grades: list[Grade]) -> float:
    return round(sum(g.score for g in grades) / len(grades), 2) if grades else 0.0


def render_report(transcript: Transcript, grades: list[Grade],
                  ratings: dict, date: str) -> str:
    topic = transcript.meta.get("topic", "unknown")
    style = transcript.meta.get("style", "unknown")
    overall = _overall(grades)

    by_concept: dict[str, list[int]] = {}
    for g in grades:
        by_concept.setdefault(g.concept, []).append(g.score)
    avg = {c: sum(s) / len(s) for c, s in by_concept.items()}
    weakest = sorted(avg, key=avg.get)[:3]
    strongest = sorted(avg, key=avg.get, reverse=True)[:3]

    lines = [
        "---",
        f'title: "Interview Report — {topic} ({date})"',
        "tags: [report, interview]",
        f"last_updated: {date}",
        "---",
        "",
        f"# Interview Report — {topic} ({style}), {date}",
        "",
        f"**Overall: {overall} / 4** across {len(grades)} questions.",
        f"Transcript: `{transcript.path.name}`",
        "",
        "## Scores",
        "",
        "| # | Concept | Level | Score |",
        "|---|---------|-------|-------|",
    ]
    for i, g in enumerate(grades, start=1):
        lines.append(f"| {i} | [[{g.concept}]] | {g.level} | {g.score}/4 |")

    lines += ["", "## Top weaknesses", ""]
    for c in weakest:
        rating = ratings.get("concepts", {}).get(c, {}).get("rating", DEFAULT_RATING)
        lines.append(f"- [[{c}]] — avg {avg[c]:.1f}/4, rating now {rating}")
        for g in grades:
            if g.concept == c:
                for gap in g.gaps[:3]:
                    lines.append(f"  - gap: {gap}")

    lines += ["", "## Top strengths", ""]
    for c in strongest:
        lines.append(f"- [[{c}]] — avg {avg[c]:.1f}/4")

    misconceptions = [m for g in grades for m in g.misconceptions]
    lines += ["", "## ⚠️ Misconceptions (fix these first)", ""]
    lines += [f"- {m}" for m in misconceptions] or ["- None detected."]

    wiki_gaps = sorted({g.concept for g in grades if g.wiki_gap})
    if wiki_gaps:
        lines += ["", "## Wiki gaps (rubric too thin — queued for MAINTAIN)", ""]
        lines += [f"- [[{c}]]" for c in wiki_gaps]

    lines += ["", "## 7-day micro study plan", ""]
    focus = weakest or list(avg)
    plan_actions = [
        "Re-read the page top to bottom; close it and rewrite the TL;DR from memory.",
        "Redo the questions you missed; check against the page only afterward.",
        "Work the Technical Detail section; reproduce any formula with real numbers.",
        "Explain the concept aloud in 2 minutes as if to an interviewer.",
        "Attempt one level-harder question on the concept (see wiki/qa/).",
        "Write down the tradeoffs table from memory; diff against the page.",
        "Mixed review: one question per weak concept, closed book.",
    ]
    for day in range(7):
        concept = focus[day % len(focus)]
        lines.append(f"- **Day {day + 1}** — [[{concept}]]: {plan_actions[day]}")

    lines.append("")
    return "\n".join(lines)


def append_assessment_log(entry: dict, path: Path | None = None) -> Path:
    path = path or (STATE_DIR / "assessment_log.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return path


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

@dataclass
class AssessmentResult:
    transcript_path: Path
    report_path: Path
    overall: float
    grades: list[Grade]
    new_tasks: list[dict]


def assess_transcript(
    transcript_path: Path,
    grader=None,
    state_dir: Path | None = None,
    reports_dir: Path | None = None,
    log_to_wiki: bool = True,
) -> AssessmentResult:
    state_dir = state_dir or STATE_DIR
    reports_dir = reports_dir or REPORTS_DIR
    date = datetime.now().strftime("%Y-%m-%d")

    transcript = parse_transcript(transcript_path)
    if transcript.meta.get("assessed"):
        raise ValueError(f"{transcript_path.name} is already assessed.")
    if not transcript.turns:
        raise ValueError(f"{transcript_path.name} has no Q/A turns.")

    concept_pages = load_concept_pages(transcript)
    if grader is None:
        grader = LLMGrader()
    grades = grade_transcript(transcript, concept_pages, grader)

    ratings = load_skill_ratings(state_dir / "skill_ratings.json")
    ratings = update_ratings(ratings, grades, date)
    save_ratings(ratings, state_dir / "skill_ratings.json")

    new_tasks = queue_follow_ups(grades, date,
                                 state_dir / "maintenance_queue.json")

    topic = transcript.meta.get("topic", "unknown")
    report_path = reports_dir / f"{date}-{topic}.md"
    counter = 2
    while report_path.exists():
        report_path = reports_dir / f"{date}-{topic}-{counter}.md"
        counter += 1
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(transcript, grades, ratings, date),
                           encoding="utf-8")

    per_concept: dict[str, list[int]] = {}
    for g in grades:
        per_concept.setdefault(g.concept, []).append(g.score)
    rel_report = (report_path.relative_to(REPO_ROOT).as_posix()
                  if report_path.is_relative_to(REPO_ROOT) else str(report_path))
    append_assessment_log({
        "date": date,
        "topic": topic,
        "style": transcript.meta.get("style", "unknown"),
        "overall": _overall(grades),
        "per_concept": {c: round(sum(s) / len(s), 1)
                        for c, s in per_concept.items()},
        "misconceptions": [m for g in grades for m in g.misconceptions],
        "report": rel_report,
    }, state_dir / "assessment_log.jsonl")

    mark_assessed(transcript_path)

    if log_to_wiki:
        from wiki_tool import append_wiki_log

        append_wiki_log(
            f"ASSESS: graded {transcript_path.name} — overall "
            f"{_overall(grades)}/4 over {len(grades)} questions; "
            f"report {rel_report}; queued {len(new_tasks)} maintenance task(s)."
        )

    return AssessmentResult(
        transcript_path=transcript_path,
        report_path=report_path,
        overall=_overall(grades),
        grades=grades,
        new_tasks=new_tasks,
    )


def run_assess(transcript: str | None = None) -> AssessmentResult | None:
    """CLI entry: assess the given transcript, or the latest unassessed one."""
    if transcript:
        path = Path(transcript)
        if not path.is_absolute():
            path = REPO_ROOT / path
        if not path.exists():
            print(f"Transcript not found: {transcript}")
            return None
    else:
        path = find_unassessed()
        if path is None:
            print("No unassessed transcripts in raw/interviews/.")
            return None

    print(f"Assessing {path.name} …")
    try:
        result = assess_transcript(path)
    except ValueError as exc:
        print(f"Cannot assess: {exc}")
        return None

    rel = result.report_path.relative_to(REPO_ROOT)
    print(f"\nOverall: {result.overall}/4 across {len(result.grades)} questions.")
    for i, g in enumerate(result.grades, start=1):
        print(f"  Q{i} [{g.concept}] level {g.level}: {g.score}/4"
              + ("  ⚠ wiki_gap" if g.wiki_gap else ""))
    print(f"Report: {rel}")
    print(f"Queued {len(result.new_tasks)} maintenance task(s).")
    return result
