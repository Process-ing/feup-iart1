from enum import Enum
from src.model.problem import RouterProblem
from src.view.cli import Cli

class CommandResult(Enum):
    SUCCESS = 0
    FAILURE = 1
    EXIT = 2

class Controller:
    def __init__(self, cli: Cli, **kwargs):
        self.__problem = kwargs.get('problem')
        self.__cli = cli

    def run_cli(self) -> None:
        while True:
            tokens = self.__cli.read_input()
            if not tokens:
                continue

            if self.process_command(tokens) == CommandResult.EXIT:
                break

    def process_command(self, tokens: list[str]) -> bool:
        if tokens[0].startswith("e") or tokens[0].startswith("q"):  # exit, quit
            return CommandResult.EXIT
        elif tokens[0].startswith("sh"):  # show
            self.__cli.print_building(self.__problem)
            return CommandResult.SUCCESS
        else:
            self.__cli.print_error(f"Unknown command '{tokens[0]}'")
            return CommandResult.FAILURE