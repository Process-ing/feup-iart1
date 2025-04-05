import sys
from typing import List
from ..model import RouterProblem

class Cli:
    """
    Command-line interface for interacting with the router problem solver.
    """
    def __init__(self) -> None:
        """
        Initialize the CLI instance.
        """

    @staticmethod
    def __get_tokens(text: str) -> List[str]:
        """
        Split input text into tokens.

        Args:
            text (str): The raw input string.

        Returns:
            List[str]: List of command tokens.
        """
        return text.strip().split()

    def __print_prefix(self) -> None:
        """
        Print the command prompt prefix.
        """
        print('[router-solver]# ', end='')

    def read_input(self) -> List[str]:
        """
        Read a line of input from the user.

        Returns:
            List[str]: Parsed command tokens from input.
        """
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
        """
        Print the current router problem's building layout.

        Args:
            problem (RouterProblem): The loaded problem instance.
        """
        print(problem.building)

    def print_success(self, message: str) -> None:
        """
        Print a success message.

        Args:
            message (str): The message to display.
        """
        print(message)

    def print_error(self, message: str) -> None:
        """
        Print an error message.

        Args:
            message (str): The error message to display.
        """
        print(message)

    def print_help(self) -> None:
        """
        Print the list of available commands and usage.
        """
        print('''Commands:
    load <file>             Load a problem from a file.
    load-solution <file>    Load and show a solution from a file.
    show                    Show the current problem.
    solve                   Solve the current problem.
    exit                    Exit the program.
    quit                    Exit the program.
    help                    Show this help.''')
