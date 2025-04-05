from abc import abstractmethod
from typing import Iterator

class AlgorithmConfig:
    pass

class Algorithm:
    @abstractmethod
    def run(self) -> Iterator[str]:
        pass
