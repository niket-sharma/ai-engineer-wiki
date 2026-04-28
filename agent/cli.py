from agent import run_agent


def main() -> None:
    print("AI Engineer Wiki - Chat Agent")
    print("Commands: 'quit' to exit, 'clear' to reset history")
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

        try:
            answer, history = run_agent(query, history)
        except Exception as exc:  # noqa: BLE001
            print(f"\nError: {exc}")
            continue

        print(f"\nWiki Agent: {answer}")


if __name__ == "__main__":
    main()
