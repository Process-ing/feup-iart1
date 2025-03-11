from abc import abstractmethod

import pygame

class PygameWindow:
    def __init__(self, max_framerate: float = 0) -> None:
        self._max_framerate = max_framerate

    @abstractmethod
    def get_window_size(self) -> tuple[int, int]:
        pass

    @abstractmethod
    def get_window_caption(self) -> str:
        pass

    @abstractmethod
    def on_init(self) -> None:
        pass

    @abstractmethod
    def on_display(self, screen: pygame.Surface) -> None:
        pass

    @abstractmethod
    def on_update(self) -> None:
        pass

    def launch(self) -> None:
        pygame.init()

        screen = pygame.display.set_mode(self.get_window_size())
        pygame.display.set_caption(self.get_window_caption())

        self.on_init()
        clock = pygame.time.Clock()

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            self.on_display(screen)
            print(f"\r{round(clock.get_fps())}", end="")  # TODO(Process-ing): Remove this
            pygame.display.flip()

            self.on_update()
            clock.tick(self._max_framerate)

        pygame.quit()
