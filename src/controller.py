from copy import deepcopy
from enum import Enum
from typing import Optional

from src.algorithm import Algorithm, RandomWalk, RandomDescent, SimulatedAnnealing, TabuSearch, GeneticAlgorithm
from src.view.score_visualizer import ScoreVisualizer
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
            with open(filename, "r", encoding="utf-8") as file:
                text = file.read()
                problem = RouterProblem.from_text(text)
                if not problem:
                    self.__cli.print_error("Failed to load problem")
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
        if command in ["exit", "quit"]:
            return CommandResult.EXIT

        if command in ["help"]:
            self.__cli.print_help()
            return CommandResult.SUCCESS

        if command in ["load"]:
            if len(tokens) != 2:
                self.__cli.print_error("Usage: load <file>")
                return CommandResult.FAILURE

            filename = tokens[1]
            return self.load_problem(filename)

        if command in ["show"]:
            if not self.__problem:
                self.__cli.print_error("No problem loaded")
                return CommandResult.FAILURE

            problem_win = ProblemWindow(self.__problem)
            problem_win.launch()

            return CommandResult.SUCCESS

        if command in ["solve"]:
            if not self.__problem:
                self.__cli.print_error("No problem loaded")
                return CommandResult.FAILURE

            problem = deepcopy(self.__problem)

            algorithm_name = None if len(tokens) < 2 else tokens[1]
            algorithm: Algorithm
            if algorithm_name == "random-walk":
                max_iterations = 200 if len(tokens) < 3 else int(tokens[2])
                algorithm = RandomWalk(problem, max_iterations=max_iterations)
            elif algorithm_name == "random-descent":
                algorithm = RandomDescent(problem)
            elif algorithm_name == "simulated-annealing":
                temperature = 100000 if len(tokens) < 3 else int(tokens[2])
                cooling_schedule = 0.99 if len(tokens) < 4 else float(tokens[3])
                max_iterations = 200 if len(tokens) < 5 else int(tokens[4])
                algorithm = SimulatedAnnealing(problem, temperature=temperature, cooling_schedule=cooling_schedule, max_iterations=max_iterations)
            elif algorithm_name == "tabu":
                tabu_tenure = None if len(tokens) < 3 else int(tokens[2])
                neighborhood_len = 10 if len(tokens) < 4 else int(tokens[3])
                max_iterations = None if len(tokens) < 5 else int(tokens[4])
                algorithm = TabuSearch(problem, tabu_tenure=tabu_tenure, neighborhood_len=neighborhood_len, max_iterations=max_iterations)
            elif algorithm_name == "genetic":
                population_size = 10 if len(tokens) < 3 else int(tokens[2])
                max_generations = 1000 if len(tokens) < 4 else int(tokens[3])
                algorithm = GeneticAlgorithm(problem, population_size=population_size, max_generations=max_generations)
            else:
                print_solve_usage()
                return CommandResult.FAILURE

            visualizer = ScoreVisualizer()
            visualizer.show()
            opt_win = OptimizationWindow(problem, algorithm, visualizer)
            opt_win.launch()
            opt_win.cleanup()

            problem.dump_to_file("solution.txt")
            print("Output saved to solution.txt")

            return CommandResult.SUCCESS

        self.__cli.print_error(f"Unknown command '{command}'")
        return CommandResult.FAILURE
