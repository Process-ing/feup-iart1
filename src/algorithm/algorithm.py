from abc import abstractmethod

class Algorithm:
    @abstractmethod
    def step(self) -> None:
        pass
