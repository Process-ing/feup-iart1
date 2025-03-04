from ..model import RouterProblem

class Cli:
    def __init__(self) -> None:
        pass

    @staticmethod
    def __get_tokens(text: str) -> list[str]:
        return text.strip().split()

    def __print_prefix(self) -> None:
        print("[router-solver]# ", end="")

    def read_input(self) -> list[str]:
        self.__print_prefix()

        try:
            text = input()
        except EOFError:
            return ['exit']

        tokens = self.__get_tokens(text)
        return tokens

    def print_problem(self, problem: RouterProblem) -> None:
        print(problem.building)

    def print_success(self, message: str) -> None:
        print(message)

    def print_error(self, message: str) -> None:
        print(message)
