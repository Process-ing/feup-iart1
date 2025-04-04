import sys
from typing import List
from ..model import RouterProblem

class Cli:
    def __init__(self) -> None:
        pass

    @staticmethod
    def __get_tokens(text: str) -> List[str]:
        return text.strip().split()

    def __print_prefix(self) -> None:
        print('[router-solver]# ', end='')

    def read_input(self) -> List[str]:
        self.__print_prefix()

        try:
            text = input()
        except EOFError:
            self.__print_prefix()
            print('quit')
            return ['quit']
        except KeyboardInterrupt:
            print('^C')
            sys.exit(1)

        tokens = self.__get_tokens(text)
        return tokens

    def print_problem(self, problem: RouterProblem) -> None:
        print(problem.building)

    def print_success(self, message: str) -> None:
        print(message)

    def print_error(self, message: str) -> None:
        print(message)

    def print_help(self) -> None:
        print('''Commands:
    load <file>  Load a problem from a file.
    show         Show the current problem.
    solve        Solve the current problem.
    exit         Exit the program.
    quit         Exit the program.
    help         Show this help.''')
