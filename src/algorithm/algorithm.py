from abc import abstractmethod
from typing import Iterator, Optional

class AlgorithmConfig:
    pass

class Algorithm:
    @abstractmethod
    def run(self) -> Iterator[Optional[str]]:
        pass
