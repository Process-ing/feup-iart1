from abc import abstractmethod


class GenericBuilding:
    @abstractmethod
    def get_num_routers(self) -> int:
        pass

    @abstractmethod
    def get_num_connected_cells(self) -> int:
        pass

    @abstractmethod
    def get_coverage(self) -> int:
        pass
