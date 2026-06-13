import argparse
import re

from interview import parse_interview_request, run_interview


def parse_assess_request(text: str) -> bool:
    lowered = text.lower()
    return bool(re.search(
        r"\b(assess|grade|score)\b.*\b(interview|session|transcript)\b"
        r"|\bhow did i do\b", lowered))


def _launch_interview(req: dict) -> None:
    try:
        run_interview(
            topic=req.get("topic"),
            style=req.get("style") or "drill",
            company=req.get("company"),
            duration_min=req.get("duration_min"),
            max_questions=req.get("max_questions") or 5,
            weakest=req.get("weakest", False),
            tutor=req.get("tutor", False),
            start_level=req.get("start_level"),
        )
    except ValueError as exc:
        print(f"\nCannot start interview: {exc}")


def chat_repl() -> None:
    from agent import run_agent

    print("AI Engineer Wiki - Chat Agent")
    print("Commands: 'quit' to exit, 'clear' to reset history")
    print("Say 'interview me on <topic>' to start a mock interview.")
    print("=" * 50)

    history: list = []
    while True:
        try:
            query = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not query:
            continue
        if query.lower() == "quit":
            break
        if query.lower() == "clear":
            history = []
            print("History cleared.")
            continue

        interview_req = parse_interview_request(query)
        if interview_req:
            _launch_interview(interview_req)
            continue

        if parse_assess_request(query):
            from assess import run_assess

            run_assess()
            continue

        from maintain import parse_maintain_request

        if parse_maintain_request(query):
            from maintain import run_maintain

            run_maintain(no_pr=True)
            continue

        try:
            answer, history = run_agent(query, history)
        except Exception as exc:  # noqa: BLE001
            print(f"\nError: {exc}")
            continue

        print(f"\nWiki Agent: {answer}")


def main() -> None:
    parser = argparse.ArgumentParser(description="AI Engineer Wiki agent CLI")
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("interview", help="Run an adaptive mock interview (OP-6)")
    p.add_argument("--topic", help="Wiki topic/concept slug, e.g. kv-cache")
    p.add_argument("--style", choices=["drill", "deep", "system-design", "behavioral"],
                   default="drill")
    p.add_argument("--company", help="Company slug, e.g. capital-one")
    p.add_argument("--questions", type=int, default=5, dest="max_questions")
    p.add_argument("--duration", type=int, default=None, dest="duration_min",
                   help="Session cap in minutes")
    p.add_argument("--weakest", action="store_true",
                   help="Interview on the lowest-rated concepts")
    p.add_argument("--tutor", action="store_true",
                   help="Allow corrective nudges mid-session")
    p.add_argument("--level", type=int, choices=[1, 2, 3, 4, 5], default=None,
                   dest="start_level", help="Override starting difficulty")
    p.add_argument("--no-llm", action="store_true",
                   help="Question-bank only; no API calls")

    a = sub.add_parser("assess", help="Grade a completed interview (OP-7)")
    a.add_argument("--transcript", default=None,
                   help="Path under raw/interviews/; defaults to the latest "
                        "unassessed transcript")

    m = sub.add_parser("maintain", help="Run the weekly maintainer (OP-8)")
    m.add_argument("--dry-run", action="store_true",
                   help="Report what would change without writing")
    m.add_argument("--no-fetch", action="store_true",
                   help="Skip the watchlist fetch (queue tasks only)")
    m.add_argument("--no-pr", action="store_true",
                   help="Apply changes to the working tree without branching/PR")
    m.add_argument("--max-pages", type=int, default=12,
                   help="Cap on wiki pages touched (hard max 12)")
    m.add_argument("--no-llm", action="store_true",
                   help="Offline mode; LLM-dependent tasks stay pending")

    args = parser.parse_args()

    if args.command == "maintain":
        from maintain import run_maintain

        run_maintain(
            dry_run=args.dry_run,
            no_fetch=args.no_fetch,
            no_pr=args.no_pr,
            max_pages=args.max_pages,
            use_llm=False if args.no_llm else None,
        )
        return

    if args.command == "assess":
        from assess import run_assess

        run_assess(args.transcript)
        return

    if args.command == "interview":
        try:
            run_interview(
                topic=args.topic,
                style=args.style,
                company=args.company,
                duration_min=args.duration_min,
                max_questions=args.max_questions,
                weakest=args.weakest,
                tutor=args.tutor,
                start_level=args.start_level,
                use_llm=False if args.no_llm else None,
            )
        except ValueError as exc:
            print(f"Cannot start interview: {exc}")
        return

    chat_repl()


if __name__ == "__main__":
    main()
