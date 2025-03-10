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
        problem_screen = pygame.Surface((problem_width, problem_height))
        problem_screen = problem_screen.convert(8)
        problem_screen.set_palette(self.__create_pallete())

        clock = pygame.time.Clock()
        font = pygame.font.Font('BigBlueTerm437NerdFont-Regular.ttf', 18)

        score = self.__problem.get_score()

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            self.__render_problem(problem_screen)
            scaled_problem = pygame.transform.scale(problem_screen, (width, height))

            text = font.render(f'Score: {score}; FPS: {clock.get_fps()}', True, (255, 255, 255))

            screen.blit(scaled_problem, (0, 0))
            screen.blit(text, (10, 10))

            pygame.display.flip()

            self.__algorithm.step()
            score = self.__problem.get_score()
            clock.tick()

        pygame.quit()

    def __create_pallete(self) -> list[tuple[int, int, int]]:
        def int_to_rgb(color: int) -> tuple[int, int, int]:
            return ((color >> 16) & 0xFF, (color >> 8) & 0xFF, color & 0xFF)

        palette = [(64, 64, 64)] * 256

        palette[CellType.VOID.value] = int_to_rgb(0x360043)
        palette[CellType.VOID.value | Building.BACKBONE_BIT] = int_to_rgb(0xece256)
        palette[CellType.VOID.value | Building.CONNECTED_BIT] = int_to_rgb(0x360043)
        palette[CellType.VOID.value | Building.BACKBONE_BIT | Building.CONNECTED_BIT] = int_to_rgb(0xece256)

        palette[CellType.TARGET.value] = int_to_rgb(0x206071)
        palette[CellType.TARGET.value | Building.BACKBONE_BIT] = int_to_rgb(0xece256)
        palette[CellType.TARGET.value | Building.CONNECTED_BIT] = int_to_rgb(0x5392a4)
        palette[CellType.TARGET.value | Building.BACKBONE_BIT | Building.CONNECTED_BIT] = int_to_rgb(0xece256)

        palette[CellType.WALL.value] = int_to_rgb(0x33356c)
        palette[CellType.WALL.value | Building.BACKBONE_BIT] = int_to_rgb(0xcab81c)
        palette[CellType.WALL.value | Building.CONNECTED_BIT] = int_to_rgb(0x33356c)
        palette[CellType.WALL.value | Building.BACKBONE_BIT | Building.CONNECTED_BIT] = int_to_rgb(0xcab81c)

        palette[CellType.ROUTER.value] = int_to_rgb(0x7fc382)
        palette[CellType.ROUTER.value | Building.BACKBONE_BIT] = int_to_rgb(0x7fc382)
        palette[CellType.ROUTER.value | Building.CONNECTED_BIT] = int_to_rgb(0x7fc382)
        palette[CellType.ROUTER.value | Building.BACKBONE_BIT | Building.CONNECTED_BIT] = int_to_rgb(0x7fc382)

        palette[CellType.BACKBONE.value] = int_to_rgb(0x00ffff)
        palette[CellType.BACKBONE.value | Building.BACKBONE_BIT] = int_to_rgb(0x00ffff)
        palette[CellType.BACKBONE.value | Building.CONNECTED_BIT] = int_to_rgb(0x00ffff)
        palette[CellType.BACKBONE.value | Building.BACKBONE_BIT | Building.CONNECTED_BIT] = int_to_rgb(0x00ffff)

        return palette

    def __render_problem(self, screen: pygame.Surface) -> None:
        pygame.pixelcopy.array_to_surface(screen, self.__problem.building.as_nparray().transpose())
