from typing import override
import pygame
from src.algorithm import Algorithm
from src.model import Building, RouterProblem, CellType
from src.view.pygameview import PygameView

class OptimizationView(PygameView):
    def __init__(self, problem: RouterProblem, algorithm: Algorithm) -> None:
        self.__problem = problem
        self.__algorithm = algorithm

    @override
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

            text = font.render(f'Score: {score}', True, (255, 255, 255))

            screen.blit(scaled_problem, (0, 0))
            screen.blit(text, (10, 10))

            pygame.display.flip()

            self.__algorithm.step()
            score = self.__problem.get_score()
            clock.tick()

        pygame.quit()

    def __create_pallete(self) -> list[tuple[int, int, int]]:
        palette = [(0, 0, 0)] * 256

        palette[CellType.VOID.value] = (0x36, 0x00, 0x43)
        palette[CellType.VOID.value | Building.BACKBONE_BIT] = (0xec, 0xe2, 0x56)
        palette[CellType.VOID.value | Building.CONNECTED_BIT] = (0x36, 0x00, 0x43)
        palette[CellType.VOID.value | Building.BACKBONE_BIT
                | Building.CONNECTED_BIT] = (0xec, 0xe2, 0x56)

        palette[CellType.TARGET.value] = (0x20, 0x60, 0x71)
        palette[CellType.TARGET.value | Building.BACKBONE_BIT] = (0xec, 0xe2, 0x56)
        palette[CellType.TARGET.value | Building.CONNECTED_BIT] = (0x53, 0x92, 0xa4)
        palette[CellType.TARGET.value | Building.BACKBONE_BIT
                | Building.CONNECTED_BIT] = (0xec, 0xe2, 0x56)

        palette[CellType.WALL.value] = (0x33, 0x35, 0x6c)
        palette[CellType.WALL.value | Building.BACKBONE_BIT] = (0xca, 0xb8, 0x1c)
        palette[CellType.WALL.value | Building.CONNECTED_BIT] = (0x33, 0x35, 0x6c)
        palette[CellType.WALL.value | Building.BACKBONE_BIT
                | Building.CONNECTED_BIT] = (0xca, 0xb8, 0x1c)

        palette[CellType.ROUTER.value] = (0x7f, 0xc3, 0x82)
        palette[CellType.ROUTER.value | Building.BACKBONE_BIT] = (0x7f, 0xc3, 0x82)
        palette[CellType.ROUTER.value | Building.CONNECTED_BIT] = (0x7f, 0xc3, 0x82)
        palette[CellType.ROUTER.value | Building.BACKBONE_BIT
                | Building.CONNECTED_BIT] = (0x7f, 0xc3, 0x82)

        palette[CellType.BACKBONE.value] = (0x00, 0xff, 0xff)
        palette[CellType.BACKBONE.value | Building.BACKBONE_BIT] = (0x00, 0xff, 0xff)
        palette[CellType.BACKBONE.value | Building.CONNECTED_BIT] = (0x00, 0xff, 0xff)
        palette[CellType.BACKBONE.value | Building.BACKBONE_BIT
                | Building.CONNECTED_BIT] = (0x00, 0xff, 0xff)

        return palette

    def __render_problem(self, screen: pygame.Surface) -> None:
        pygame.pixelcopy.array_to_surface(screen, self.__problem.building.as_nparray().transpose())
