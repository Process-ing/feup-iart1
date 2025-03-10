from abc import abstractmethod

class PygameView:
    @abstractmethod
    def render(self) -> None:
        pass
