from abc import abstractmethod

class GenericBuilding:
    '''
    Abstract class representing a generic building.

    This class defines the basic structure and methods that any building
    implementation should follow.
    '''
    @abstractmethod
    def get_num_routers(self) -> int:
        '''
        Get the number of routers in the building.
        
        Returns:
            int: The number of routers in the building.
        '''
        pass

    @abstractmethod
    def get_num_connected_cells(self) -> int:
        '''
        Get the number of connected cells in the building.

        Returns:
            int: The number of connected cells in the building.
        '''
        pass

    @abstractmethod
    def get_coverage(self) -> int:
        '''
        Get the coverage of the building.

        Returns:
            int: The coverage of the building.
        '''
        pass
