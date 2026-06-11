# state/ — Interview Agent State

Version-controlled state files for the adaptive interview layer
(see `interview-agent-spec.md`). No database — plain files in git.

| File | Written by | Schema |
|---|---|---|
| `skill_ratings.json` | ASSESS only | Per-concept Elo ratings. Defaults: rating 1200, K=32 for first 5 sessions then 16. Difficulty levels 1–5 map to opponent ratings 1000–1800. |
| `maintenance_queue.json` | ASSESS (producer), MAINTAIN (consumer) | `{"tasks": [{id, type: generate_qa\|wiki_gap, ...}]}` |
| `assessment_log.jsonl` | ASSESS | One JSON object per assessed session (append-only). |

**Invariants**

- Only the ASSESS operation may modify `skill_ratings.json`.
- Deleting this directory must degrade gracefully: unknown concepts start at
  rating 1200, the queue starts empty (spec §7.5).

## skill_ratings.json — per-concept entry shape

```json
{
  "version": 1,
  "updated": "2026-06-10T14:00:00Z",
  "concepts": {
    "kv-cache": {
      "rating": 1340,
      "sessions": 4,
      "last_assessed": "2026-06-08",
      "trend": [1200, 1265, 1310, 1340],
      "wiki_page": "wiki/concepts/kv-cache.md"
    }
  }
}
```

## maintenance_queue.json — task shapes

```json
{"id": "q-001", "type": "generate_qa", "concept": "gqa",
 "difficulty": 4, "reason": "scored 1/4 on 2026-06-10", "status": "pending"}
{"id": "q-002", "type": "wiki_gap", "page": "wiki/concepts/rope.md",
 "section": "extrapolation behavior", "reason": "rubric too thin to grade",
 "status": "pending"}
```

## assessment_log.jsonl — entry shape

```json
{"date": "2026-06-10", "topic": "transformers", "style": "deep",
 "overall": 2.7, "per_concept": {"kv-cache": 3, "rope": 1},
 "misconceptions": ["claimed RoPE is applied to V"],
 "report": "wiki/reports/2026-06-10-transformers.md"}
```
