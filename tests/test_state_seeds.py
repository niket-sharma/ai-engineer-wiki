"""Phase 0 scaffolding tests: state/ seeds and interview-agent directories.

Guards the contracts from interview-agent-spec.md §4 (data schemas) and
§7.5 (deleting state/ degrades gracefully to defaults).
"""
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_interview_agent_directories_exist():
    for rel in ("state", "wiki/reports", "raw/interviews", "raw/auto"):
        assert (REPO_ROOT / rel).is_dir(), f"missing directory: {rel}"


def test_skill_ratings_seed_schema():
    data = json.loads((REPO_ROOT / "state" / "skill_ratings.json").read_text())
    assert data["version"] == 1
    assert isinstance(data["concepts"], dict)
    # Seed must be empty so unknown concepts fall back to the default rating (1200)
    for concept in data["concepts"].values():
        assert {"rating", "sessions", "last_assessed", "trend", "wiki_page"} <= set(concept)


def test_maintenance_queue_seed_schema():
    data = json.loads((REPO_ROOT / "state" / "maintenance_queue.json").read_text())
    assert isinstance(data["tasks"], list)
    for task in data["tasks"]:
        assert task["type"] in ("generate_qa", "wiki_gap")
        assert task["status"] in ("pending", "done", "skipped")


def test_assessment_log_is_valid_jsonl():
    path = REPO_ROOT / "state" / "assessment_log.jsonl"
    assert path.exists()
    for line in path.read_text().splitlines():
        if line.strip():
            json.loads(line)


def test_skill_md_defines_new_operations():
    skill = (REPO_ROOT / "skill.md").read_text(encoding="utf-8")
    for op in ("OP-6: INTERVIEW", "OP-7: ASSESS", "OP-8: MAINTAIN"):
        assert op in skill, f"skill.md missing operation spec: {op}"


def test_skill_md_new_ops_survive_system_prompt_truncation():
    """agent.py embeds only the first 24,000 chars of skill.md (agent/agent.py:85).

    If the new op specs land past that cut they are invisible to the agent.
    """
    skill = (REPO_ROOT / "skill.md").read_text(encoding="utf-8")
    for op in ("OP-6: INTERVIEW", "OP-7: ASSESS", "OP-8: MAINTAIN"):
        assert skill.index(op) < 24000, f"{op} spec falls outside the 24k prompt excerpt"
