#!/usr/bin/env python3
"""
fetch_sources.py — Download raw source material into the wiki's raw/ directories.

Sources fetched:
  - arXiv papers (abstract + key metadata, optionally full text via arxiv2text)
  - GitHub READMEs (via GitHub raw API)
  - YouTube transcripts (via youtube-transcript-api)

Usage:
    # Install deps first:
    pip install arxiv requests youtube-transcript-api rich

    # Fetch all sources:
    python scripts/fetch_sources.py

    # Fetch a single source by key:
    python scripts/fetch_sources.py --only attention-is-all-you-need

    # Fetch and immediately trigger ingest (requires claude CLI in PATH):
    python scripts/fetch_sources.py --ingest
"""

import argparse
import json
import re
import sys
import textwrap
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Source manifest — add new sources here
# ---------------------------------------------------------------------------

SOURCES = [
    # ── Transformers ──────────────────────────────────────────────────────
    {
        "key": "attention-is-all-you-need",
        "type": "arxiv",
        "arxiv_id": "1706.03762",
        "out": "raw/transformers/attention-is-all-you-need.md",
        "tags": ["transformers", "attention"],
    },
    {
        "key": "flash-attention",
        "type": "arxiv",
        "arxiv_id": "2205.14135",
        "out": "raw/transformers/flash-attention.md",
        "tags": ["transformers", "optimization"],
    },
    {
        "key": "flash-attention-2",
        "type": "arxiv",
        "arxiv_id": "2307.08691",
        "out": "raw/transformers/flash-attention-2.md",
        "tags": ["transformers", "optimization"],
    },
    {
        "key": "rope-positional-encoding",
        "type": "arxiv",
        "arxiv_id": "2104.09864",
        "out": "raw/transformers/rope-positional-encoding.md",
        "tags": ["transformers", "positional-encoding"],
    },
    {
        "key": "gqa-grouped-query-attention",
        "type": "arxiv",
        "arxiv_id": "2305.13245",
        "out": "raw/transformers/gqa-grouped-query-attention.md",
        "tags": ["transformers", "attention", "inference"],
    },
    {
        "key": "karpathy-gpt-lecture",
        "type": "youtube",
        "video_id": "kCc8FmEb1nY",
        "out": "raw/transformers/karpathy-gpt-lecture-transcript.md",
        "tags": ["transformers", "gpt", "lecture"],
        "title": "Andrej Karpathy — Let's build GPT: from scratch, in code, spelled out",
    },
    {
        "key": "nanogpt-readme",
        "type": "github",
        "url": "https://raw.githubusercontent.com/karpathy/nanoGPT/master/README.md",
        "out": "raw/transformers/nanogpt-readme.md",
        "tags": ["transformers", "gpt", "implementation"],
    },
    # ── Fine-tuning & Alignment ───────────────────────────────────────────
    {
        "key": "lora",
        "type": "arxiv",
        "arxiv_id": "2106.09685",
        "out": "raw/rl-and-rlhf/lora-paper.md",
        "tags": ["fine-tuning", "peft"],
    },
    {
        "key": "qlora",
        "type": "arxiv",
        "arxiv_id": "2305.14314",
        "out": "raw/rl-and-rlhf/qlora-paper.md",
        "tags": ["fine-tuning", "peft", "quantization"],
    },
    {
        "key": "instructgpt-rlhf",
        "type": "arxiv",
        "arxiv_id": "2203.02155",
        "out": "raw/rl-and-rlhf/instructgpt-rlhf.md",
        "tags": ["alignment", "rlhf"],
    },
    {
        "key": "dpo",
        "type": "arxiv",
        "arxiv_id": "2305.18290",
        "out": "raw/rl-and-rlhf/dpo-paper.md",
        "tags": ["alignment", "dpo"],
    },
    {
        "key": "deepseek-r1",
        "type": "arxiv",
        "arxiv_id": "2501.12948",
        "out": "raw/rl-and-rlhf/deepseek-r1-grpo.md",
        "tags": ["alignment", "grpo", "reasoning"],
    },
    {
        "key": "ppo",
        "type": "arxiv",
        "arxiv_id": "1707.06347",
        "out": "raw/rl-and-rlhf/ppo-paper.md",
        "tags": ["rl", "ppo"],
    },
    # ── RAG & Agents ─────────────────────────────────────────────────────
    {
        "key": "rag-original",
        "type": "arxiv",
        "arxiv_id": "2005.11401",
        "out": "raw/mlops/rag-original-paper.md",
        "tags": ["rag", "retrieval"],
    },
    {
        "key": "hyde",
        "type": "arxiv",
        "arxiv_id": "2212.10496",
        "out": "raw/mlops/hyde-paper.md",
        "tags": ["rag", "retrieval"],
    },
    {
        "key": "colbert",
        "type": "arxiv",
        "arxiv_id": "2004.12832",
        "out": "raw/mlops/colbert-paper.md",
        "tags": ["retrieval", "reranking"],
    },
    {
        "key": "react-agents",
        "type": "arxiv",
        "arxiv_id": "2210.03629",
        "out": "raw/mlops/react-agents-paper.md",
        "tags": ["agents", "reasoning"],
    },
    {
        "key": "langgraph-readme",
        "type": "github",
        "url": "https://raw.githubusercontent.com/langchain-ai/langgraph/main/README.md",
        "out": "raw/mlops/langgraph-readme.md",
        "tags": ["agents", "orchestration"],
    },
    # ── Statistics & Classic ML ───────────────────────────────────────────
    {
        "key": "attention-survey",
        "type": "arxiv",
        "arxiv_id": "2106.04803",  # Efficient Transformers: A Survey
        "out": "raw/statistics-and-ml/efficient-transformers-survey.md",
        "tags": ["transformers", "survey"],
    },
]

# ---------------------------------------------------------------------------
# Fetch functions
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent


def fetch_arxiv(source: dict, verbose: bool = True) -> str:
    """Fetch arXiv paper metadata and abstract."""
    try:
        import arxiv  # type: ignore
    except ImportError:
        return _fallback_arxiv(source)

    arxiv_id = source["arxiv_id"]
    if verbose:
        print(f"  📄 Fetching arXiv:{arxiv_id} ...")

    client = arxiv.Client()
    search = arxiv.Search(id_list=[arxiv_id])
    results = list(client.results(search))
    if not results:
        return _fallback_arxiv(source)

    paper = results[0]
    authors = ", ".join(a.name for a in paper.authors[:6])
    if len(paper.authors) > 6:
        authors += " et al."

    categories = ", ".join(paper.categories)
    published = paper.published.strftime("%Y-%m-%d") if paper.published else "unknown"

    content = textwrap.dedent(f"""\
        # {paper.title}

        **Authors:** {authors}
        **Published:** {published}
        **arXiv ID:** {arxiv_id}
        **URL:** https://arxiv.org/abs/{arxiv_id}
        **Categories:** {categories}

        ---

        ## Abstract

        {paper.summary.strip()}

        ---

        ## Key Contributions (to be filled during ingest)

        - [ ] Main contribution 1
        - [ ] Main contribution 2
        - [ ] Key equations / algorithms

        ## Figures & Tables (to be filled during ingest)

        ## Interview-Relevant Insights (to be filled during ingest)

        ## Notes

        _Fetched {datetime.now().strftime('%Y-%m-%d')} by fetch_sources.py_
        _Run INGEST on this file to compile into the wiki._
    """)
    return content


def _fallback_arxiv(source: dict) -> str:
    arxiv_id = source["arxiv_id"]
    return textwrap.dedent(f"""\
        # arXiv:{arxiv_id}

        **URL:** https://arxiv.org/abs/{arxiv_id}

        > ⚠️ Could not fetch paper automatically. Install the `arxiv` package:
        > `pip install arxiv`
        > Or paste the abstract + notes here manually.

        ---

        ## Abstract

        _Paste abstract here_

        ## Key Contributions

        ## Interview-Relevant Insights

        _Fetched {datetime.now().strftime('%Y-%m-%d')}_
    """)


def fetch_github(source: dict, verbose: bool = True) -> str:
    """Fetch a raw file from GitHub."""
    try:
        import requests  # type: ignore
    except ImportError:
        print("  ⚠️  requests not installed. Run: pip install requests")
        return f"# GitHub source\n\nURL: {source['url']}\n\n_Install `requests` to fetch automatically._\n"

    url = source["url"]
    if verbose:
        print(f"  🐙 Fetching {url} ...")

    resp = requests.get(url, timeout=15)
    if resp.status_code != 200:
        return f"# GitHub source\n\nURL: {url}\n\n_HTTP {resp.status_code} — fetch failed._\n"

    header = textwrap.dedent(f"""\
        <!-- Source: {url} -->
        <!-- Fetched: {datetime.now().strftime('%Y-%m-%d')} by fetch_sources.py -->
        <!-- Run INGEST on this file to compile into the wiki. -->

    """)
    return header + resp.text


def fetch_youtube(source: dict, verbose: bool = True) -> str:
    """Fetch a YouTube transcript."""
    try:
        from youtube_transcript_api import YouTubeTranscriptApi  # type: ignore
        from youtube_transcript_api.formatters import TextFormatter  # type: ignore
    except ImportError:
        print("  ⚠️  youtube-transcript-api not installed. Run: pip install youtube-transcript-api")
        return _fallback_youtube(source)

    video_id = source["video_id"]
    title = source.get("title", f"YouTube video {video_id}")
    if verbose:
        print(f"  🎥 Fetching transcript for {video_id} ...")

    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
        formatter = TextFormatter()
        transcript_text = formatter.format_transcript(transcript_list)

        # Wrap long lines
        wrapped = "\n".join(
            textwrap.fill(line, width=100) if len(line) > 100 else line
            for line in transcript_text.splitlines()
        )

        word_count = len(transcript_text.split())

        content = textwrap.dedent(f"""\
            # {title}

            **Source:** https://www.youtube.com/watch?v={video_id}
            **Fetched:** {datetime.now().strftime('%Y-%m-%d')}
            **Word count:** ~{word_count:,}

            > Run INGEST on this file to compile into the wiki.

            ---

            ## Transcript

            {wrapped}
        """)
        return content

    except Exception as exc:
        print(f"  ⚠️  Transcript fetch failed: {exc}")
        return _fallback_youtube(source)


def _fallback_youtube(source: dict) -> str:
    video_id = source["video_id"]
    title = source.get("title", f"YouTube video {video_id}")
    return textwrap.dedent(f"""\
        # {title}

        **Source:** https://www.youtube.com/watch?v={video_id}

        > ⚠️ Transcript fetch failed. Options:
        > 1. Install `youtube-transcript-api`: `pip install youtube-transcript-api`
        > 2. Paste your own notes below

        ---

        ## Notes

        _Paste lecture notes or transcript here_
    """)


# ---------------------------------------------------------------------------
# Ingest trigger
# ---------------------------------------------------------------------------

def trigger_ingest(out_path: str, verbose: bool = True) -> None:
    """Print the claude CLI ingest command (or run it if claude is in PATH)."""
    import shutil, subprocess
    cmd = f'claude "Ingest {out_path} into the wiki. Follow OP-1 from SKILL.md exactly."'
    if shutil.which("claude"):
        if verbose:
            print(f"  🤖 Triggering ingest: {cmd}")
        subprocess.run(["claude", f"Ingest {out_path} into the wiki. Follow OP-1 from SKILL.md exactly."])
    else:
        if verbose:
            print(f"\n  ➡️  To ingest, run:\n     {cmd}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch raw source material for the AI Engineer Wiki")
    parser.add_argument("--only", metavar="KEY", help="Fetch only the source with this key")
    parser.add_argument("--ingest", action="store_true", help="Trigger claude ingest after each fetch")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be fetched, don't write")
    parser.add_argument("--list", action="store_true", help="List all source keys and exit")
    args = parser.parse_args()

    if args.list:
        print("\nAvailable source keys:\n")
        for s in SOURCES:
            print(f"  {s['key']:<40}  [{s['type']}]  →  {s['out']}")
        print()
        return

    sources_to_fetch = SOURCES
    if args.only:
        sources_to_fetch = [s for s in SOURCES if s["key"] == args.only]
        if not sources_to_fetch:
            print(f"❌ No source with key '{args.only}'. Run with --list to see all keys.")
            sys.exit(1)

    print(f"\n🚀 Fetching {len(sources_to_fetch)} source(s)...\n")
    fetched, skipped, failed = 0, 0, 0

    for source in sources_to_fetch:
        out_path = REPO_ROOT / source["out"]
        print(f"[{source['key']}]")

        if out_path.exists():
            # Don't overwrite if file already has real content (> 500 chars and not just a stub)
            existing = out_path.read_text(encoding="utf-8")
            if len(existing) > 500 and "Paste abstract here" not in existing and "fetch_sources.py" not in existing:
                print(f"  ⏭️  {source['out']} already has content, skipping.")
                skipped += 1
                print()
                continue

        if args.dry_run:
            print(f"  [dry-run] Would write to {source['out']}")
            print()
            continue

        try:
            if source["type"] == "arxiv":
                content = fetch_arxiv(source)
            elif source["type"] == "github":
                content = fetch_github(source)
            elif source["type"] == "youtube":
                content = fetch_youtube(source)
            else:
                print(f"  ⚠️  Unknown source type: {source['type']}")
                failed += 1
                continue

            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(content, encoding="utf-8")
            print(f"  ✅ Written to {source['out']} ({len(content):,} chars)")
            fetched += 1

            if args.ingest:
                trigger_ingest(source["out"])

        except Exception as exc:
            print(f"  ❌ Failed: {exc}")
            failed += 1

        print()

    print(f"Done. Fetched: {fetched}  |  Skipped: {skipped}  |  Failed: {failed}")
    if fetched > 0 and not args.ingest:
        print("\nTo ingest all fetched sources into the wiki, run:")
        for s in sources_to_fetch:
            print(f'  claude "Ingest {s[\"out\"]} into the wiki. Follow OP-1 from SKILL.md exactly."')


if __name__ == "__main__":
    main()
