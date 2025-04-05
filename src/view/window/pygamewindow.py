from abc import abstractmethod
from typing import List, Tuple

import pygame

class PygameWindow:
    """
    Abstract base class for a Pygame application window.
    Provides a framework for setting up and running a Pygame loop
    with user-defined initialization and update behavior.
    """
    def __init__(self, max_framerate: float = 0) -> None:
        """
        Initializes the Pygame window base class.

        Args:
            max_framerate (float): The maximum number of frames per second.
                                   Set to 0 for uncapped framerate.
        """
        self._max_framerate = max_framerate

    @abstractmethod
    def get_window_size(self) -> Tuple[int, int]:
        """
        Returns the size of the window.

        Returns:
            Tuple[int, int]: Width and height of the window in pixels.
        """
        pass

    @abstractmethod
    def get_window_caption(self) -> str:
        """
        Returns the caption of the window.

        Returns:
            str: The title text shown in the window bar.
        """
        pass

    @abstractmethod
    def on_init(self, screen: pygame.Surface) -> None:
        """
        Called once when the window is first created.
        Used for drawing the initial frame or setup logic.

        Args:
            screen (pygame.Surface): The main drawing surface for the window.
        """
        pass

    @abstractmethod
    def on_update(self, event: List[pygame.event.Event], screen: pygame.Surface) -> None:
        """
        Called once every frame to update the window content.

        Args:
            event (List[pygame.event.Event]): The list of Pygame events since the last frame.
            screen (pygame.Surface): The main drawing surface for the window.
        """
        pass

    def launch(self) -> None:
        """
        Launches the window and runs the main Pygame loop until the window is closed.
        """
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
