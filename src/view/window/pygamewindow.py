from abc import abstractmethod
from typing import List, Tuple

import pygame

class PygameWindow:
    def __init__(self, max_framerate: float = 0) -> None:
        self._max_framerate = max_framerate

    @abstractmethod
    def get_window_size(self) -> Tuple[int, int]:
        pass

    @abstractmethod
    def get_window_caption(self) -> str:
        pass

    @abstractmethod
    def on_init(self, screen: pygame.Surface) -> None:
        pass

    @abstractmethod
    def on_update(self, event: List[pygame.event.Event], screen: pygame.Surface) -> None:
        pass

    def launch(self) -> None:
        pygame.init()

        screen = pygame.display.set_mode(self.get_window_size())
        pygame.display.set_caption(self.get_window_caption())

        self.on_init(screen)
        clock = pygame.time.Clock()

        running = True
        while running:
            events = pygame.event.get()
            for event in events:
                if event.type == pygame.QUIT:
                    running = False

            self.on_update(events, screen)
            clock.tick(self._max_framerate)

        pygame.quit()
