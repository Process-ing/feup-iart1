from typing import override
import pygame
from src.algorithm import Algorithm
from src.model import RouterProblem
from src.view.pygameview import PygameView

class OptimizationView(PygameView):
    # def __init__(self, problem: RouterProblem) -> None:
    def __init__(self, problem: RouterProblem, algorithm: Algorithm) -> None:
        self.__problem = problem
        self.__algorithm = algorithm

    @override
    def render(self) -> None:
        pygame.init()

        monitor_width = pygame.display.Info().current_w
        monitor_height = pygame.display.Info().current_h
        problem_height, problem_width = self.__problem.building.shape
        cell_size = max(min(monitor_width // problem_width, monitor_height // problem_height), 1)
        height, width = problem_height * cell_size, problem_width * cell_size

        pygame.display.set_caption('Router Optimization')
        screen = pygame.display.set_mode((width, height))
        clock = pygame.time.Clock()

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            screen.fill((0, 0, 0))
            self.render_problem(screen, cell_size)
            pygame.display.flip()

            self.__algorithm.step()
            clock.tick(60)

        pygame.quit()

    def render_problem(self, screen: pygame.Surface, cell_size: int) -> None:
        for row, column, _cell in self.__problem.building.iter():
            color = (row + column) % 255
            pygame.draw.rect(
                screen,
                (color, color, color),
                (column * cell_size, row * cell_size, cell_size, cell_size)
            )
