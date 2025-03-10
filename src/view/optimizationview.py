from typing import override
import pygame
from src.algorithm import Algorithm
from src.model import Building, RouterProblem, CellType
from src.view.viewer import ProblemViewer

class OptimizationView:
    def __init__(self, problem: RouterProblem, algorithm: Algorithm) -> None:
        self.__problem = problem
        self.__algorithm = algorithm

    def render(self) -> None:
        pygame.init()

        problem_height, problem_width = self.__problem.building.shape
        cell_size = max(min(
            pygame.display.Info().current_w // problem_width,
            pygame.display.Info().current_h // problem_height
        ), 1)
        height, width = problem_height * cell_size, problem_width * cell_size

        pygame.display.set_caption('Router Optimization')
        screen = pygame.display.set_mode((width, height))

        clock = pygame.time.Clock()
        font = pygame.font.Font('BigBlueTerm437NerdFont-Regular.ttf', 18)
        problem_viewer = ProblemViewer()

        score = self.__problem.get_score()

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            problem_screen = problem_viewer.render(self.__problem)
            scaled_problem = pygame.transform.scale(problem_screen, (width, height))
            screen.blit(scaled_problem, (0, 0))

            text = font.render(f'Score: {score} FPS: {int(clock.get_fps())}', True, (255, 255, 255))
            screen.blit(text, (10, 10))

            pygame.display.flip()

            self.__algorithm.step()
            score = self.__problem.get_score()
            clock.tick()

        pygame.quit()
