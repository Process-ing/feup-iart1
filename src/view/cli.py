class Cli:
    def __init__(self):
        pass

    @staticmethod
    def __get_tokens(text: str) -> list[str]:
        return text.strip().split()

    def __print_prefix(self) -> None:
        print("[router-solver]# ", end="")

    def run(self) -> None:
        while True:
            self.__print_prefix()

            try:
                text = input()
            except EOFError:
                break

            tokens = self.__get_tokens(text)
            if not tokens:
                continue

            if tokens[0] in ["exit", "e", "quit", "q"]:
                break
            else:
                print(f"Unknown command: {tokens[0]}")