from typing import override
import pygame
from src.algorithm import Algorithm
from src.model import Building, RouterProblem, CellType
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
        font = pygame.font.Font('BigBlueTerm437NerdFont-Regular.ttf', 36)

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            screen.fill((0, 0, 0))
            self.__render_problem(screen, cell_size)
            text = font.render(f'Score: 0', True, (255, 255, 255))
            screen.blit(text, (10, 10))

            pygame.display.flip()

            self.__algorithm.step()
            clock.tick(60)

        pygame.quit()

    def __render_problem(self, screen: pygame.Surface, cell_size: int) -> None:
        for row, column, _cell in self.__problem.building.iter():
            pygame.draw.rect(
                screen,
                self.__to_color(_cell),
                (column * cell_size, row * cell_size, cell_size, cell_size)
            )

    def __to_color(self, cell: int) -> tuple[int, int, int]:
        cell_type = cell & Building.CELL_TYPE_MASK
        if cell_type == CellType.VOID.value:
            return 0xece256 if cell & Building.BACKBONE_BIT else 0x360043

        if cell_type == CellType.TARGET.value:
            if cell & Building.BACKBONE_BIT:
                return 0xece256
            if cell & Building.CONNECTED_BIT:
                return 0x5392a4
            return 0x206071

        if cell_type == CellType.WALL.value:
            return 0xcab81c if cell & Building.BACKBONE_BIT else 0x33356c

        if cell_type == CellType.ROUTER.value:
            return 0x7fc382

        if cell_type == CellType.BACKBONE.value:
            return 0x00ffff

        raise ValueError(f'Invalid cell type {cell}')
