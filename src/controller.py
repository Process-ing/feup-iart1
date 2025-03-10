from enum import Enum
from typing import Optional, override
from src.algorithm import Algorithm
from src.model import RouterProblem
from src.view import Cli, OptimizationView

# TODO(Process-ing): Remove this
class MockAlgorithm(Algorithm):
    def __init__(self, problem: RouterProblem) -> None:
        self.i = 0
        self.j = 0
        self.problem = problem

    @override
    def step(self) -> None:
        self.problem.building.place_router(self.j, self.i)
        self.i += 1
        if self.i == self.problem.building.columns:
            self.i = 0
            self.j += 1
            if self.j == self.problem.building.rows:
                self.j = 0

class CommandResult(Enum):
    SUCCESS = 0
    FAILURE = 1
    FILE_ERROR = 2
    EXIT = 3

class Controller:
    def __init__(self, cli: Cli, problem: Optional[RouterProblem] = None):
        self.__problem = problem
        self.__cli = cli

    def run_cli(self) -> None:
        while True:
            tokens = self.__cli.read_input()
            if not tokens:
                continue

            if self.process_command(tokens) == CommandResult.EXIT:
                break

    def load_problem(self, filename: str) -> CommandResult:
        try:
            with open(filename, 'r', encoding='utf-8') as file:
                text = file.read()
                problem = RouterProblem.from_text(text)
                if not problem:
                    self.__cli.print_error('Failed to load problem')
                    return CommandResult.FAILURE

        except FileNotFoundError:
            self.__cli.print_error(f"File '{filename}' not found")
            return CommandResult.FILE_ERROR

        except IsADirectoryError:
            self.__cli.print_error(f"'{filename}' is a directory")
            return CommandResult.FILE_ERROR

        self.__problem = problem
        self.__cli.print_success(f"Problem loaded from '{filename}'")
        return CommandResult.SUCCESS

    def process_command(self, tokens: list[str]) -> CommandResult:
        command = tokens[0]
        if command in ['exit', 'quit']:
            return CommandResult.EXIT

        if command in ['help']:
            self.__cli.print_help()
            return CommandResult.SUCCESS

        if command in ['load']:
            if len(tokens) != 2:
                self.__cli.print_error('Usage: load <file>')
                return CommandResult.FAILURE

            filename = tokens[1]
            return self.load_problem(filename)

        if command in ['show']:  # show
            if not self.__problem:
                self.__cli.print_error('No problem loaded')
                return CommandResult.FAILURE

            algorithm = MockAlgorithm(self.__problem)
            opt_view = OptimizationView(self.__problem, algorithm)
            opt_view.render()

            return CommandResult.SUCCESS

        self.__cli.print_error(f"Unknown command '{command}'")
        return CommandResult.FAILURE
