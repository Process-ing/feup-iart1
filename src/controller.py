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
    """
    Controller class for handling CLI commands and managing problem solving and visualization.

    Attributes:
        __problem (Optional[RouterProblem]): The currently loaded router problem.
        __cli (Cli): The CLI interface for reading user commands and displaying output.
    """
    def __init__(self, cli: Cli, problem: Optional[RouterProblem] = None):
        """
        Initialize the Controller with a CLI interface and an optional router problem.

        Args:
            cli (Cli): CLI interface for input/output.
            problem (Optional[RouterProblem]): Optional problem instance to load initially.
        """
        self.__problem = problem
        self.__cli = cli

    def run_cli(self) -> None:
        """
        Run the main CLI loop, reading and processing user commands until 'exit' is received.
        """
        while True:
            tokens = self.__cli.read_input()
            if not tokens:
                continue

            if self.process_command(tokens) == CommandResult.EXIT:
                break

    def load_problem(self, filename: str) -> CommandResult:
        """
        Load a problem definition from a file.

        Args:
            filename (str): Path to the file containing the problem definition.

        Returns:
            CommandResult: SUCCESS if loaded correctly, FILE_ERROR on file issue, FAILURE otherwise.
        """
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

    def load_solution(self, filename: str) -> CommandResult:
        """
        Load a precomputed solution into the current problem from a file.

        Args:
            filename (str): Path to the file containing the solution.

        Returns:
            CommandResult: SUCCESS if successful, FILE_ERROR if file issues, FAILURE otherwise.
        """
        try:
            with open(filename, 'r', encoding='utf-8') as file:
                text = file.read()
                assert self.__problem is not None
                self.__problem.building.load_solution(text)

        except FileNotFoundError:
            self.__cli.print_error(f'File \'{filename}\' not found')
            return CommandResult.FILE_ERROR

        except IsADirectoryError:
            self.__cli.print_error(f'\'{filename}\' is a directory')
            return CommandResult.FILE_ERROR

        self.__cli.print_success(f'Problem loaded from \'{filename}\'')
        return CommandResult.SUCCESS

    def process_command(self, tokens: List[str]) -> CommandResult: # pylint: disable=too-many-branches
        """
        Process a list of command tokens input by the user.

        Args:
            tokens (List[str]): The list of command and arguments.

        Returns:
            CommandResult: Result of command execution (e.g., SUCCESS, FAILURE, EXIT).
        """
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

        if command in ['load-solution']:
            # Ensure the problem is loaded
            if not self.__problem:
                self.__cli.print_error('No problem loaded')
                return CommandResult.FAILURE

            # Load the solution
            if len(tokens) != 2:
                self.__cli.print_error('Usage: load-solution <file>')
                return CommandResult.FAILURE

            filename = tokens[1]
            result = self.load_solution(filename)
            if result != CommandResult.SUCCESS:
                self.__cli.print_error('Error loading solution')
                return result

            # Launch the visualization window
            problem_win = ProblemWindow(self.__problem)
            problem_win.launch()
            return CommandResult.SUCCESS

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
        """
        Parse the algorithm type and its configuration from CLI tokens.

        Args:
            problem (RouterProblem): Problem to solve.
            tokens (List[str]): CLI tokens including the algorithm name and flags.

        Returns:
            Optional[Algorithm]: Configured algorithm instance, or None if invalid.
        """
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
        """
        Parse algorithm configuration flags from command-line tokens.

        Flags must start with '--' and may include:
            --flag=value    (inline)
            --flag value    (space-separated)

        If a flag is specified without a value, it is set to 'True'.

        Args:
            tokens (List[str]): List of CLI tokens representing flags and their values.

        Returns:
            Optional[Dict[str, str]]: A dictionary of flag names and their string values,
                                      or None if a syntax error is found.
        """
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
