from abc import abstractmethod
from src.model.generic_building import GenericBuilding

class GenericRouterProblem:
    '''
    Abstract class representing a generic router problem.

    This class defines the basic structure and methods that any router
    problem implementation should follow.
    '''

    @property
    @abstractmethod
    def router_range(self) -> int:
        '''
        Get the range of the router.

        Returns:
            int: The range of the router.
        '''

    @abstractmethod
    def check_budget(self, building: GenericBuilding) -> bool:
        '''
        Check if the budget is sufficient for the given building.

        Args:
            building (GenericBuilding): The building to check the budget for.
        
        Returns:
            bool: True if the budget is sufficient, False otherwise.
        '''

    @abstractmethod
    def get_score(self, building: GenericBuilding) -> int:
        '''
        Get the score of the building.
        
        Args:
            building (GenericBuilding): The building to get the score for.
        
        Returns:
            int: The score of the building.
        '''
