from abc import abstractmethod
from typing import Iterator, Optional

class AlgorithmConfig:
    """
    Represents the configuration for an algorithm. This class is currently a placeholder 
    and may be extended in the future to hold specific configuration parameters for 
    algorithmic operations.
    """

class Algorithm:
    """
    Abstract base class representing an algorithm. Specific algorithms should inherit from 
    this class and implement the `run` method.

    Methods:
        run: Executes the algorithm and yields results.
    """

    @abstractmethod
    def run(self) -> Iterator[Optional[str]]:
        """
        Executes the algorithm and returns an iterator that yields results as strings.

        The results can be `None` if no result is available at a given step.

        Returns:
            Iterator[Optional[str]]: An iterator that yields the results of the algorithm.
        """
