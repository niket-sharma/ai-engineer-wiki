"""OP-6 INTERVIEW — adaptive mock-interview session manager.

Spec: interview-agent-spec.md §3.1, §4.1, §4.2.

Deterministic parts (scope resolution, Elo-band difficulty picker, QA-bank
parsing, dedup, transcript writing) live here and are unit-tested without
network access. The LLM is used only for novel question generation, deep-style
follow-up probes, and the private mid-session correctness judgment that drives
escalation — all optional: with no API key the session falls back to the
wiki/qa/ question bank at a fixed difficulty.

State contract: this module READS state/skill_ratings.json but never writes it
(only ASSESS may modify ratings). It writes new files under raw/interviews/
and never overwrites an existing raw file.
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import json

REPO_ROOT = Path(__file__).resolve().parent.parent
WIKI_DIR = REPO_ROOT / "wiki"
STATE_DIR = REPO_ROOT / "state"
INTERVIEWS_DIR = REPO_ROOT / "raw" / "interviews"
PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

try:
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass

# Elo difficulty mapping (spec §4.1): level 1 = 1000 … level 5 = 1800.
LEVEL_RATINGS = {1: 1000, 2: 1200, 3: 1400, 4: 1600, 5: 1800}
DEFAULT_RATING = 1200
BAND = 150  # productive-struggle zone: |level rating - concept rating| <= 150

STYLES = ("drill", "deep", "system-design", "behavioral")

# wiki/qa/ pages group questions under "## L1/L2/L3" headings.
QA_LEVEL_MAP = {"l1": 2, "l2": 3, "l3": 4}


# ---------------------------------------------------------------------------
# Ratings (read-only here)
# ---------------------------------------------------------------------------

def load_skill_ratings(path: Path | None = None) -> dict:
    """Load state/skill_ratings.json; degrade gracefully to defaults (§7.5)."""
    path = path or (STATE_DIR / "skill_ratings.json")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {"version": 1, "updated": None, "concepts": {}}
    if not isinstance(data.get("concepts"), dict):
        data["concepts"] = {}
    return data


def concept_rating(ratings: dict, concept: str) -> int:
    entry = ratings.get("concepts", {}).get(concept, {})
    rating = entry.get("rating", DEFAULT_RATING)
    try:
        return int(rating)
    except (TypeError, ValueError):
        return DEFAULT_RATING


def eligible_levels(rating: int) -> list[int]:
    """Levels whose rating is within ±BAND of the concept rating."""
    return [lvl for lvl, r in LEVEL_RATINGS.items() if abs(r - rating) <= BAND]


def pick_level(rating: int) -> int:
    """Pick the question level for a concept rating.

    Within the ±150 band; ties break toward the harder level. If the rating
    drifts outside every band (rating < 850 or > 1950), clamp to the nearest
    level so the session can always proceed.
    """
    levels = eligible_levels(rating)
    if not levels:
        return min(LEVEL_RATINGS, key=lambda lvl: abs(LEVEL_RATINGS[lvl] - rating))
    return max(levels, key=lambda lvl: (-abs(LEVEL_RATINGS[lvl] - rating), lvl))


# ---------------------------------------------------------------------------
# Scope resolution
# ---------------------------------------------------------------------------

@dataclass
class Scope:
    topic: str
    concepts: list[str]
    concept_pages: dict[str, str]          # slug -> full page content
    qa_pages: dict[str, str]               # repo-relative path -> content
    company: str | None = None
    company_page: str | None = None


def _frontmatter(text: str) -> str:
    if not text.startswith("---\n"):
        return ""
    end = text.find("\n---", 4)
    return text[4:end] if end != -1 else ""


def _fm_list(fm: str, key: str) -> list[str]:
    m = re.search(rf"^{key}:\s*\[(.*)\]\s*$", fm, flags=re.MULTILINE)
    if not m:
        return []
    items = re.findall(r'"([^"]+)"|\'([^\']+)\'|([^,\[\]"\']+)', m.group(1))
    out = []
    for a, b, c in items:
        v = (a or b or c).strip()
        if v:
            out.append(v)
    return out


def _link_slugs(values: list[str]) -> list[str]:
    return [re.sub(r"[\[\]]", "", v).strip().lower() for v in values]


def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.strip().lower()).strip("-")


def resolve_scope(
    topic: str | None = None,
    company: str | None = None,
    weakest: bool = False,
    n_weakest: int = 3,
    ratings: dict | None = None,
    max_concepts: int = 8,
) -> Scope:
    """Resolve which wiki pages a session draws on (spec §3.1 step 1)."""
    ratings = ratings if ratings is not None else load_skill_ratings()

    if weakest:
        known = ratings.get("concepts", {})
        if not known:
            raise ValueError(
                "No skill ratings yet — run a topic interview first "
                "(e.g. 'interview me on transformers')."
            )
        ranked = sorted(known, key=lambda c: concept_rating(ratings, c))
        concepts = ranked[:n_weakest]
        topic = topic or "weakest-topics"
    elif topic:
        topic = _slugify(topic)
        concepts = _concepts_for_topic(topic, max_concepts)
        if not concepts:
            raise ValueError(
                f"No wiki coverage found for topic '{topic}'. "
                "Try 'list wiki pages' or ingest a source first."
            )
    else:
        raise ValueError("Interview needs a topic, or weakest=True.")

    concept_pages: dict[str, str] = {}
    for slug in concepts:
        path = _find_page(slug, subdirs=("concepts", "system-design"))
        if path:
            concept_pages[slug] = path.read_text(encoding="utf-8")

    qa_pages = _qa_pages_for(topic, concepts)

    company_slug = _slugify(company) if company else None
    company_page = None
    if company_slug:
        path = _find_page(company_slug, subdirs=("companies",))
        if path:
            company_page = path.read_text(encoding="utf-8")

    return Scope(
        topic=topic,
        concepts=[c for c in concepts if c in concept_pages] or concepts,
        concept_pages=concept_pages,
        qa_pages=qa_pages,
        company=company_slug,
        company_page=company_page,
    )


def _find_page(slug: str, subdirs: tuple[str, ...]) -> Path | None:
    for sub in subdirs:
        d = WIKI_DIR / sub
        if not d.is_dir():
            continue
        candidate = d / f"{slug}.md"
        if candidate.exists():
            return candidate
    return None


def _concepts_for_topic(topic: str, max_concepts: int) -> list[str]:
    """topic → the topic's own page (if any), its `related` links, and any
    concept pages tagged with the topic."""
    concepts: list[str] = []

    def add(slug: str) -> None:
        slug = slug.strip().lower()
        if slug and slug not in concepts:
            concepts.append(slug)

    own = _find_page(topic, subdirs=("concepts", "system-design"))
    if own:
        add(topic)
        fm = _frontmatter(own.read_text(encoding="utf-8"))
        for slug in _link_slugs(_fm_list(fm, "related")):
            if _find_page(slug, subdirs=("concepts", "system-design")):
                add(slug)

    for sub in ("concepts", "system-design"):
        d = WIKI_DIR / sub
        if not d.is_dir():
            continue
        for path in sorted(d.glob("*.md")):
            if len(concepts) >= max_concepts:
                break
            fm = _frontmatter(path.read_text(encoding="utf-8"))
            tags = [t.lower() for t in _fm_list(fm, "tags")]
            if topic in tags:
                add(path.stem)

    return concepts[:max_concepts]


def _qa_pages_for(topic: str, concepts: list[str]) -> dict[str, str]:
    qa_dir = WIKI_DIR / "qa"
    if not qa_dir.is_dir():
        return {}
    wanted = set(concepts) | {topic}
    out: dict[str, str] = {}
    for path in sorted(qa_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        fm = _frontmatter(text)
        tags = {t.lower() for t in _fm_list(fm, "tags")}
        related = set(_link_slugs(_fm_list(fm, "related")))
        slug_terms = set(path.stem.lower().replace("-qa", "").split("-"))
        if wanted & (tags | related) or wanted & slug_terms:
            out[f"wiki/qa/{path.name}"] = text
    return out


# ---------------------------------------------------------------------------
# Question bank
# ---------------------------------------------------------------------------

@dataclass
class Question:
    concept: str
    level: int
    text: str
    source: str            # "wiki/qa/<page>.md" or "generated"
    generated: bool = False
    explicit: bool = True  # False when attributed to the topic by fallback


def normalize_question(text: str) -> str:
    """Canonical form used for in-session dedup."""
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


def parse_qa_bank(scope: Scope) -> list[Question]:
    """Extract questions from wiki/qa/ pages (## L1/L2/L3 → ### Qn. headings).

    Each question is attributed to the first scope concept its text mentions,
    falling back to the session topic.
    """
    questions: list[Question] = []
    for rel_path, text in scope.qa_pages.items():
        level = 2
        for line in text.splitlines():
            m = re.match(r"^##\s+(L\d)\b", line)
            if m:
                level = QA_LEVEL_MAP.get(m.group(1).lower(), 2)
                continue
            q = re.match(r"^###\s+Q\d+[.):]?\s*(.+)$", line)
            if q:
                q_text = q.group(1).strip()
                concept, explicit = _attribute_concept(q_text, scope)
                questions.append(
                    Question(
                        concept=concept,
                        level=level,
                        text=q_text,
                        source=rel_path,
                        explicit=explicit,
                    )
                )
    return questions


def _attribute_concept(question_text: str, scope: Scope) -> tuple[str, bool]:
    lowered = question_text.lower()
    for slug in scope.concepts:
        if slug.replace("-", " ") in lowered or slug in lowered:
            return slug, True
    return scope.topic, False


# ---------------------------------------------------------------------------
# LLM interviewer (optional)
# ---------------------------------------------------------------------------

def _load_prompt(name: str) -> str:
    path = PROMPTS_DIR / name
    return path.read_text(encoding="utf-8") if path.exists() else ""


class LLMInterviewer:
    """Thin wrapper for the generative parts of a session.

    Used for: novel question generation when bank coverage is thin, deep-style
    follow-up probes, and the private correct/partial/wrong judgment that
    drives escalation. Never used to grade officially — that is ASSESS.
    """

    def __init__(self, model: str | None = None):
        import openai  # local import so offline sessions never need it

        self.client = openai.OpenAI()
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o")
        self.system = _load_prompt("interviewer_system.md")

    def _chat(self, prompt: str, max_tokens: int = 400) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": self.system},
                {"role": "user", "content": prompt},
            ],
        )
        return (response.choices[0].message.content or "").strip()

    def generate_question(
        self, concept: str, level: int, page: str, asked: list[str], style: str,
        company_page: str | None = None,
    ) -> str:
        company_note = (
            f"\nCompany interview notes (bias style/topics toward this):\n{company_page[:4000]}"
            if company_page
            else ""
        )
        avoid = "\n".join(f"- {q}" for q in asked[-12:]) or "- (none yet)"
        return self._chat(
            f"Generate ONE interview question.\n"
            f"Concept: {concept}\nDifficulty level: {level} (1=recall … 5=open-ended design under constraints)\n"
            f"Style: {style}\n"
            f"Ground it strictly in this wiki page (do not quote or reveal its content):\n"
            f"---\n{page[:8000]}\n---{company_note}\n"
            f"Already asked this session (do NOT repeat or rephrase these):\n{avoid}\n"
            f"Output the question text only — no preamble, no answer, no hints."
        )

    def follow_up(self, question: str, answer: str, concept: str, page: str) -> str:
        return self._chat(
            f"The candidate was asked: {question}\n"
            f"They answered: {answer}\n"
            f"Concept page for your private reference (do not reveal):\n---\n{page[:6000]}\n---\n"
            f"Ask ONE probing follow-up ('why', 'what breaks if…', 'how would you verify…'). "
            f"Do not teach, do not confirm or deny correctness. Question only."
        )

    def judge(self, question: str, answer: str, page: str) -> str:
        """Private provisional verdict for difficulty adaptation only."""
        verdict = self._chat(
            f"Question: {question}\nCandidate answer: {answer}\n"
            f"Reference page:\n---\n{page[:6000]}\n---\n"
            f"Classify the answer as exactly one word: correct, partial, or wrong.",
            max_tokens=4,
        ).lower()
        return verdict if verdict in ("correct", "partial", "wrong") else "unknown"


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------

@dataclass
class Turn:
    question: Question
    answer: str = ""
    follow_ups: list[tuple[str, str]] = field(default_factory=list)
    verdict: str = "unknown"   # private; drives escalation, never shown


class InterviewSession:
    """Stateful session: question selection, adaptive difficulty, dedup,
    transcript writing."""

    def __init__(
        self,
        scope: Scope,
        style: str = "drill",
        max_questions: int = 5,
        duration_min: int | None = None,
        ratings: dict | None = None,
        interviewer: LLMInterviewer | None = None,
        tutor: bool = False,
        start_level: int | None = None,
    ):
        if style not in STYLES:
            raise ValueError(f"style must be one of {STYLES}")
        self.scope = scope
        self.style = style
        self.max_questions = 1 if style == "system-design" else max_questions
        self.duration_min = duration_min
        self.ratings = ratings if ratings is not None else load_skill_ratings()
        self.interviewer = interviewer
        self.tutor = tutor
        self.turns: list[Turn] = []
        self._asked: set[str] = set()
        self._bank = parse_qa_bank(scope)
        self._concept_cycle = 0
        self._started = time.monotonic()
        self._start_date = datetime.now()
        # Working level per concept: seeded from the Elo band, adjusted ±1
        # in-session on correct/wrong (spec §3.1 step 3).
        self.working_levels: dict[str, int] = {
            c: (start_level or pick_level(concept_rating(self.ratings, c)))
            for c in scope.concepts
        }
        if style == "system-design":
            self.working_levels = {c: 5 for c in self.working_levels}

    # -- selection ----------------------------------------------------------

    def _next_concept(self) -> str:
        concept = self.scope.concepts[self._concept_cycle % len(self.scope.concepts)]
        self._concept_cycle += 1
        return concept

    def _bank_question(self, concept: str, level: int) -> Question | None:
        """Unasked bank question for `concept`, preferring the exact level,
        then the nearest level."""
        candidates = [
            q for q in self._bank
            if q.concept == concept and normalize_question(q.text) not in self._asked
        ]
        if not candidates:
            return None
        candidates.sort(key=lambda q: (not q.explicit, abs(q.level - level), q.level))
        return candidates[0]

    def next_question(self) -> Question | None:
        if len(self.turns) >= self.max_questions or self._time_up():
            return None

        for _ in range(len(self.scope.concepts)):
            concept = self._next_concept()
            level = self.working_levels.get(concept, 2)

            question = self._bank_question(concept, level)
            if question is None or abs(question.level - level) > 1:
                generated = self._generate(concept, level)
                question = generated or question
            if question is None:
                continue

            key = normalize_question(question.text)
            if key in self._asked:
                continue
            self._asked.add(key)
            return question
        return None

    def _generate(self, concept: str, level: int) -> Question | None:
        if self.interviewer is None:
            return None
        page = self.scope.concept_pages.get(concept, "")
        try:
            text = self.interviewer.generate_question(
                concept, level, page, [t.question.text for t in self.turns],
                self.style, self.scope.company_page,
            )
        except Exception:  # noqa: BLE001 — fall back to the bank on API errors
            return None
        if not text or normalize_question(text) in self._asked:
            return None
        return Question(concept=concept, level=level, text=text,
                        source="generated", generated=True)

    # -- answering ----------------------------------------------------------

    def record_answer(self, question: Question, answer: str) -> Turn:
        turn = Turn(question=question, answer=answer)
        if self.interviewer is not None:
            page = self.scope.concept_pages.get(question.concept, "")
            try:
                turn.verdict = self.interviewer.judge(question.text, answer, page)
            except Exception:  # noqa: BLE001
                turn.verdict = "unknown"
        self._adapt(question.concept, turn.verdict)
        self.turns.append(turn)
        return turn

    def _adapt(self, concept: str, verdict: str) -> None:
        level = self.working_levels.get(concept, 2)
        if verdict == "correct":
            self.working_levels[concept] = min(5, level + 1)
        elif verdict == "wrong":
            self.working_levels[concept] = max(1, level - 1)

    def _time_up(self) -> bool:
        if not self.duration_min:
            return False
        return (time.monotonic() - self._started) >= self.duration_min * 60

    # -- transcript ---------------------------------------------------------

    def elapsed_minutes(self) -> int:
        return max(1, round((time.monotonic() - self._started) / 60))

    def concepts_touched(self) -> list[str]:
        seen: list[str] = []
        for t in self.turns:
            if t.question.concept not in seen:
                seen.append(t.question.concept)
        return seen

    def transcript(self) -> str:
        date = self._start_date.strftime("%Y-%m-%d")
        lines = [
            "---",
            "type: interview-transcript",
            f"date: {date}",
            f"topic: {self.scope.topic}",
            f"style: {self.style}",
        ]
        if self.scope.company:
            lines.append(f"company: {self.scope.company}")
        lines += [
            f"duration_min: {self.elapsed_minutes()}",
            f"questions: {len(self.turns)}",
            f"concepts_touched: [{', '.join(self.concepts_touched())}]",
            "assessed: false",
            "---",
            "",
            f"# Interview Transcript — {self.scope.topic} ({self.style})",
            "",
        ]
        for i, turn in enumerate(self.turns, start=1):
            q = turn.question
            tag = "generated:true" if q.generated else f"source: {q.source}"
            lines.append(f"## Q{i} (concept: {q.concept}, level: {q.level}, {tag})")
            lines.append(f"**Interviewer:** {q.text}")
            lines.append("")
            lines.append(f"**Candidate:** {turn.answer or '(no answer)'}")
            lines.append("")
            for j, (fq, fa) in enumerate(turn.follow_ups, start=1):
                lines.append(f"**Interviewer (follow-up {j}):** {fq}")
                lines.append("")
                lines.append(f"**Candidate:** {fa or '(no answer)'}")
                lines.append("")
        return "\n".join(lines)

    def save_transcript(self, out_dir: Path | None = None) -> Path:
        out_dir = out_dir or INTERVIEWS_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        date = self._start_date.strftime("%Y-%m-%d")
        stem = f"{date}-{self.scope.topic}-{self.style}"
        path = out_dir / f"{stem}.md"
        counter = 2
        while path.exists():  # raw/ is immutable — never overwrite
            path = out_dir / f"{stem}-{counter}.md"
            counter += 1
        path.write_text(self.transcript(), encoding="utf-8")
        return path


# ---------------------------------------------------------------------------
# Request parsing + CLI loop
# ---------------------------------------------------------------------------

def parse_interview_request(text: str) -> dict | None:
    """Parse natural-language interview requests, e.g.
    'Interview me on transformers, 30 minutes, company=capital-one'
    'Interview me on my weakest topics'
    'Run a system design interview, hard mode'
    'drill me on kv-cache, 5 questions'
    Returns None when the text is not an interview request.
    """
    lowered = text.lower()
    if not re.search(r"\binterview\b|\bdrill me\b|\bmock interview\b", lowered):
        return None

    req: dict = {"topic": None, "style": None, "company": None,
                 "duration_min": None, "max_questions": None,
                 "weakest": False, "tutor": False, "start_level": None}

    if re.search(r"system[- ]design", lowered):
        req["style"] = "system-design"
    for s in ("behavioral", "deep", "drill"):
        if req["style"] is None and re.search(rf"\b{s}\b", lowered):
            req["style"] = s

    m = re.search(r"company\s*[=:]?\s*([a-z0-9-]+)", lowered)
    if m:
        req["company"] = m.group(1)
    m = re.search(r"(\d+)\s*min", lowered)
    if m:
        req["duration_min"] = int(m.group(1))
    m = re.search(r"(\d+)\s*questions?", lowered)
    if m:
        req["max_questions"] = int(m.group(1))
    if "weakest" in lowered:
        req["weakest"] = True
    if "tutor" in lowered:
        req["tutor"] = True
    if re.search(r"\bhard\b", lowered):
        req["start_level"] = 4
    elif re.search(r"\beasy\b", lowered):
        req["start_level"] = 1

    m = re.search(r"\bon\s+(my\s+weakest\s+topics?|[a-z0-9][a-z0-9 -]*)", lowered)
    if m and not req["weakest"]:
        candidate = m.group(1)
        candidate = re.split(r",|\bfor\b|\bcompany\b|\d+\s*min", candidate)[0]
        req["topic"] = _slugify(candidate)

    return req


def run_interview(
    topic: str | None = None,
    style: str = "drill",
    company: str | None = None,
    duration_min: int | None = None,
    max_questions: int = 5,
    weakest: bool = False,
    tutor: bool = False,
    start_level: int | None = None,
    use_llm: bool | None = None,
    out_dir: Path | None = None,
) -> Path | None:
    """Interactive CLI interview loop. Returns the transcript path."""
    scope = resolve_scope(topic=topic, company=company, weakest=weakest)

    interviewer = None
    if use_llm is None:
        use_llm = bool(os.getenv("OPENAI_API_KEY"))
    if use_llm:
        try:
            interviewer = LLMInterviewer()
        except Exception as exc:  # noqa: BLE001
            print(f"(LLM unavailable — falling back to question bank: {exc})")

    session = InterviewSession(
        scope=scope, style=style, max_questions=max_questions,
        duration_min=duration_min, interviewer=interviewer,
        tutor=tutor, start_level=start_level,
    )

    print(f"\n=== Mock interview: {scope.topic} | style={style} "
          f"| up to {session.max_questions} questions ===")
    print("Concepts in scope:", ", ".join(scope.concepts))
    print("Type your answer and press Enter. 'skip' to pass, 'end' to finish early.\n")

    n_followups = {"drill": 0, "deep": 2, "system-design": 4, "behavioral": 1}[style]

    while True:
        question = session.next_question()
        if question is None:
            break
        print(f"\nQ{len(session.turns) + 1}. {question.text}")
        try:
            answer = input("> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n(Session ended early.)")
            break
        if answer.lower() == "end":
            break
        if answer.lower() == "skip":
            answer = ""
        turn = session.record_answer(question, answer)

        if interviewer is not None and answer:
            page = scope.concept_pages.get(question.concept, "")
            for _ in range(n_followups):
                try:
                    probe = interviewer.follow_up(question.text, answer, question.concept, page)
                except Exception:  # noqa: BLE001
                    break
                if not probe:
                    break
                print(f"\n↳ {probe}")
                try:
                    f_answer = input("> ").strip()
                except (KeyboardInterrupt, EOFError):
                    f_answer = ""
                turn.follow_ups.append((probe, f_answer))
                if not f_answer or f_answer.lower() in ("skip", "end"):
                    break

    if not session.turns:
        print("No questions were asked — nothing to save.")
        return None

    path = session.save_transcript(out_dir)
    rel = path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path
    print(f"\nSession complete: {len(session.turns)} questions.")
    print(f"Transcript written to {rel}")
    print("Run ASSESS to grade it (e.g. 'assess my interview').")
    return path
