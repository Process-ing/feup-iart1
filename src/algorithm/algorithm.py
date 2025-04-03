from abc import abstractmethod
from typing import Iterator

class Algorithm:
    @abstractmethod
    def run(self) -> Iterator[None]:
        pass
