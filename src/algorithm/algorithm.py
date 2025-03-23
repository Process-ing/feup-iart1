from abc import abstractmethod

class Algorithm:
    @abstractmethod
    def step(self) -> None:
        pass

    @abstractmethod
    def done(self) -> bool:
        pass