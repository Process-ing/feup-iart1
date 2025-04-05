from copy import deepcopy
from enum import Enum
from typing import Dict, List, Optional

# pylint: disable=wildcard-import
from src.algorithm import *
from src.model import RouterProblem
from src.view import Cli
from src.view.solve_usage import print_solve_usage
from src.view.window import OptimizationWindow, ProblemWindow

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
            self.__cli.print_error(f'File \'{filename}\' not found')
            return CommandResult.FILE_ERROR

        except IsADirectoryError:
            self.__cli.print_error(f'\'{filename}\' is a directory')
            return CommandResult.FILE_ERROR

        self.__problem = problem
        self.__cli.print_success(f'Problem loaded from \'{filename}\'')
        return CommandResult.SUCCESS

    def process_command(self, tokens: List[str]) -> CommandResult:
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

        if command in ['show']:
            if not self.__problem:
                self.__cli.print_error('No problem loaded')
                return CommandResult.FAILURE

            problem_win = ProblemWindow(self.__problem)
            problem_win.launch()

            return CommandResult.SUCCESS

        if command in ['solve']:
            if not self.__problem:
                self.__cli.print_error('No problem loaded')
                return CommandResult.FAILURE

            problem = deepcopy(self.__problem)

            algorithm = self.receive_algorithm(problem, tokens)
            if not algorithm:
                print_solve_usage()
                return CommandResult.FAILURE

            opt_win = OptimizationWindow(problem, algorithm)
            opt_win.launch()
            opt_win.cleanup()

            print(f"Best score: {opt_win.get_best_score()} ({opt_win.get_duration()} s)")
            opt_win.dump_to_file('results.txt')
            print('Score saved to results.txt')
            problem.dump_to_file('solution.txt')
            print('Output saved to solution.txt')

            return CommandResult.SUCCESS

        self.__cli.print_error(f'Unknown command \'{command}\'')
        return CommandResult.FAILURE

    @staticmethod
    def receive_algorithm(problem: RouterProblem, tokens: List[str]) -> Optional[Algorithm]:
        algorithm_name = None if len(tokens) < 2 else tokens[1]
        algorithm: Algorithm
        config: Optional[AlgorithmConfig]

        flags = Controller.parse_algorithm_flags(tokens[2:])
        if flags is None:
            return None

        if algorithm_name == 'random-walk':
            config = RandomWalkConfig.from_flags(flags)
            if not config:
                return None

            algorithm = RandomWalk(problem, config)

        elif algorithm_name == 'random-descent':
            config = RandomDescentConfig.from_flags(flags)
            if not config:
                return None

            algorithm = RandomDescent(problem, config)

        elif algorithm_name == 'simulated-annealing':
            config = SimulatedAnnealingConfig.from_flags(flags)
            if not config:
                return None

            algorithm = SimulatedAnnealing(problem, config)

        elif algorithm_name == 'tabu':
            default_tabu_tenure = TabuSearch.get_default_tenure(problem)
            config = TabuSearchConfig.from_flags(flags, default_tabu_tenure)
            if not config:
                return None

            algorithm = TabuSearch(problem, config)

        elif algorithm_name == 'genetic':
            default_init_routers = GeneticAlgorithm.get_default_init_routers(problem)
            config = GeneticAlgorithmConfig.from_flags(flags, default_init_routers)
            if not config:
                return None

            algorithm = GeneticAlgorithm(problem, config)

        else:
            return None

        return algorithm

    @staticmethod
    def parse_algorithm_flags(tokens: List[str]) -> Optional[Dict[str, str]]:
        '''
        Parse flags from tokens. Flags are expected to start with '--'.
        They can be specified in either of two formats:
            --flag=value    (an inline value)
            --flag value    (the value is the next token)
        Returns a dictionary mapping flag names to their string values.
        If a flag is provided without a value, it is set as "True".
        '''
        flags = {}
        i = 0
        while i < len(tokens):
            token = tokens[i]
            if token.startswith('--'):
                if '=' in token:
                    flag, value = token[2:].split('=', 1)
                    flags[flag] = value
                else:
                    flag = token[2:]
                    # Check if a value follows and it isn't another flag:
                    if i + 1 < len(tokens) and not tokens[i + 1].startswith('--'):
                        flags[flag] = tokens[i + 1]
                        i += 1
                    else:
                        flags[flag] = 'True'
            else:
                return None
            i += 1
        return flags
