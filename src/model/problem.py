from typing import Tuple, cast, override
import numpy as np
from src.model.generic_building import GenericBuilding
from src.model.generic_problem import GenericRouterProblem
from src.model.building import Building

type BudgetInfo = Tuple[int, int, int]

class RouterProblem(GenericRouterProblem):
    '''
    Class representing a router problem.

    This class contains the building, router range, budget information,
    and methods to check the budget and calculate the score.
    '''

    def __init__(self, building: Building, router_range: int, budget_info: BudgetInfo) -> None:
        '''
        Initializes the RouterProblem with the given building, router range,
        and budget information.

        Args:
            building (Building): The building for the router problem.
            router_range (int): The range of the router.
            budget_info (BudgetInfo): A tuple containing the budget information.
                The tuple contains three integers: backbone price, router price,
                and total budget.
        '''
        self.__building = building
        self.__best_building = building
        self.__router_range = router_range
        self.__backbone_price = budget_info[0]
        self.__router_price = budget_info[1]
        self.__budget = budget_info[2]

    @property
    @override
    def router_range(self) -> int:
        '''
        Get the range of the router.

        Returns:
            int: The range of the router.
        '''
        return self.__router_range

    @property
    def backbone_price(self) -> int:
        '''
        Get the price of the backbone.

        Returns:
            int: The price of the backbone.
        '''
        return self.__backbone_price

    @property
    def router_price(self) -> int:
        '''
        Get the price of the router.

        Returns:
            int: The price of the router.
        '''
        return self.__router_price

    @property
    def budget(self) -> int:
        '''
        Get the total budget.

        Returns:
            int: The total budget.
        '''
        return self.__budget

    @classmethod
    def from_text(cls, text: str) -> 'RouterProblem':
        '''
        Create a RouterProblem instance from a text representation.

        The text representation should contain the building dimensions,
        backbone coordinates, and the building map.
        The first section contains the dimensions (rows, columns, router range),
        backbone coordinate, and budget information.
        The second section contains the building map.
        The format is as follows:
        rows columns router_range 
        backbone_price router_price budget
        backbone_x backbone_y
        building_map
        The building map is represented as a string of characters.
        Each character represents a cell in the building.
        The characters are as follows:
        - '.': empty cell
        - '#': wall
        - '-': void cell

        Args:
            text (str): The text representation of the RouterProblem.
        
        Returns:
            RouterProblem: A RouterProblem instance created from the text.
        '''
        parts = text.strip().split(maxsplit=8)

        initial_section = tuple(map(int, parts[0:8]))
        building_section = parts[8]

        rows, columns, router_range = initial_section[0:3]
        budget_info = cast(BudgetInfo, tuple(initial_section[3:6]))
        backbone = cast(Tuple[int, int], tuple(initial_section[6:8]))

        building = Building.from_text((rows, columns), backbone, building_section, None)
        problem = cls(building, router_range, budget_info)
        building.problem = problem

        return problem

    @property
    def building(self) -> Building:
        '''
        Get the building associated with the router problem.

        Returns:
            Building: The building associated with the router problem.
        '''
        return self.__building

    @building.setter
    def building(self, building: Building) -> None:
        '''
        Set the building associated with the router problem.

        This method updates the building and checks if the new building
        has a better score than the current best building.
        If the new building has a better score, it updates the best building.

        Args:
            building (Building): The new building to set.
        '''
        self.__building = building
        if building.score > self.__best_building.score:
            self.__best_building = building

    @property
    def best_building(self) -> Building:
        '''
        Get the best building associated with the router problem.

        This is the building with the highest score.

        Returns:
            Building: The best building associated with the router problem.
        '''
        return self.__best_building

    @override
    def check_budget(self, building: GenericBuilding) -> bool:
        '''
        Check if the budget is sufficient for the given building.

        This method calculates the total cost of the building by multiplying
        the number of routers and connected cells by their respective prices.
        If the total cost is less than or equal to the budget, it returns True,
        otherwise it returns False.

        Args:
            building (GenericBuilding): The building to check the budget for.
        
        Returns:
            bool: True if the budget is sufficient, False otherwise.
        '''
        num_routers = building.get_num_routers()
        num_connected_cells = building.get_num_connected_cells()

        return num_routers * self.__router_price + num_connected_cells \
            * self.backbone_price <= self.budget

    @override
    def get_score(self, building: GenericBuilding) -> int:
        '''
        Get the score of the building.

        The score is calculated as follows:
        score = 1000 * coverage + (budget - total_cost)
        where:
        - coverage is the coverage of the building
        - total_cost is the sum of the cost of routers and connected cells
        The total cost is calculated by multiplying the number of routers
        and connected cells by their respective prices.
        The budget is the total budget available for the router problem.

        Args:
            building (GenericBuilding): The building to get the score for.
        Returns:
            int: The score of the building.
        '''
        num_routers = building.get_num_routers()
        num_connected_cells = building.get_num_connected_cells()
        coverage = building.get_coverage()

        return 1000 * coverage + \
            (self.budget - \
                (num_routers * self.router_price) - \
                (num_connected_cells * self.backbone_price))

    def get_available_budget(self, building: GenericBuilding) -> int:
        '''
        Get the available budget for the given building.

        The available budget is calculated as:
        available_budget = budget - total_cost
        where:
        - total_cost is the sum of the cost of routers and connected cells
        The total cost is calculated by multiplying the number of routers
        and connected cells by their respective prices.
        The budget is the total budget available for the router problem.

        Args:
            building (GenericBuilding): The building to get the available budget for.
        
        Returns:
            int: The available budget for the building.
        '''
        num_routers = building.get_num_routers()
        num_connected_cells = building.get_num_connected_cells()

        return self.budget - (num_routers * self.router_price) \
            - (num_connected_cells * self.backbone_price)


    def dump_to_file(self, filename: str) -> None:
        '''
        Dump the best building to a file.

        The file will contain the number of backbone cells and their coordinates,
        followed by the number of router cells and their coordinates.
        The format is as follows:
        num_backbone_cells
        row1 col1
        row2 col2
        ...
        num_router_cells
        row1 col1
        row2 col2
        ...

        Args:
            filename (str): The name of the file to dump the building to.
        '''
        building_map = self.__best_building.as_nparray()
        backbone_cells = [(i, j) for (i, j), cell in np.ndenumerate(building_map) \
            if cell & Building.BACKBONE_BIT]
        router_cells = [(i, j) for (i, j), cell in np.ndenumerate(building_map) \
            if cell & Building.ROUTER_BIT]

        with open(filename, 'w', encoding='utf-8') as file:
            file.write(f'{len(backbone_cells)}\n')
            for (row, col) in backbone_cells:
                file.write(f'{row} {col}\n')
            file.write(f'{len(router_cells)}\n')
            for (row, col) in router_cells:
                file.write(f'{row} {col}\n')
