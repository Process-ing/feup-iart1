from enum import Enum
from typing import Callable, Deque, Dict, Iterator, Optional, Set, Tuple, cast, List, override
from collections import deque
from copy import deepcopy
import random
import numpy as np

from src.model.generic_building import GenericBuilding
from src.model.generic_problem import GenericRouterProblem
from src.model.disjoint_set import DisjointSet
from src.model.error import ProblemLoadError

type CellArray = np.ndarray[Tuple[int, ...], np.dtype[np.uint8]]
type CheckBudgetCallback = Callable[['Building'], bool]
type Pos = Tuple[int, int]
type SteinerTreeQueue = Deque[Tuple[Pos, Pos, Optional[Pos], Optional[Pos]]]

class Operator:
    """
    Represents an operation to place or remove a router in a building.

    Attributes:
        place (bool): True if the operation is to place a router, False to remove.
        row (int): The row index of the router's position.
        col (int): The column index of the router's position.
    """
    def __init__(self, place: bool, row: int, col: int) -> None:
        """
        Initializes an Operator instance.

        Args:
            place (bool): Indicates if the operation is to place a router (True) or remove (False).
            row (int): Row index of the operation.
            col (int): Column index of the operation.
        """
        self.place = place
        self.row = row
        self.col = col

    def apply(self, building: 'Building') -> Optional['Building']:
        """
        Applies the operation to the given building by either placing or removing a router.

        Args:
            building (Building): The building instance to apply the operation to.

        Returns:
            Optional[Building]: A new building with the operation applied,
            or None if the operation failed.
        """
        new_building = building.copy()

        if self.place:
            success = new_building.place_router(self.row, self.col)
        else:
            success = new_building.remove_router(self.row, self.col)

        assert building.problem is not None
        return new_building if success and building.problem.check_budget(new_building) else None

    @property
    def pos(self) -> Pos:
        """
        Returns the position of the operator as a tuple (row, col).

        Returns:
            Pos: A tuple representing the position of the operator.
        """
        return (self.row, self.col)

class CellType(Enum):
    VOID = 0
    TARGET = 1
    WALL = 2

class Building(GenericBuilding):
    """
    Represents a building grid with routers, targets, walls, and a backbone.
    The building allows placing and removing routers, covering neighboring cells,
    and checking for valid configurations.

    Attributes:
        cells (CellArray): A 2D array representing the building grid.
        backbone_root (Pos): The position of the backbone in the grid.
        new_router_probability (float): The probability of placing a new router in the grid.
        problem (Optional[GenericRouterProblem]): The problem associated
        with the building (used for validation).
        score (Optional[int]): A score representing the quality of the current
        building configuration.
    """
    BACKBONE_BIT = np.uint8(1 << 7)  # Marks a cell connected to the backbone
    COVERED_BIT = np.uint8(1 << 6)  # Marks a cell covered by a router
    ROUTER_BIT = np.uint8(1 << 5)  # Marks a cell as a router
    CELL_TYPE_MASK = np.uint8(0b00011111)  # Mask for the cell type

    CHAR_CELL_MAP = {
        ord('-'): CellType.VOID,
        ord('.'): CellType.TARGET,
        ord('#'): CellType.WALL,
    }

    def __init__(self, cells: CellArray, backbone: Pos, new_router_probability: float,
                 problem: Optional[GenericRouterProblem]) -> None:
        """
        Initializes a Building instance.

        Args:
            cells (CellArray): A 2D array representing the building grid.
            backbone (Pos): The position of the backbone in the grid.
            new_router_probability (float): The probability of placing a new router in the grid.
            problem (Optional[GenericRouterProblem]): The problem associated with the building.
        """
        self.__cells: CellArray = cells
        self.__backbone_root = backbone
        self.__new_router_probability = new_router_probability
        self.problem = problem
        self.__score: Optional[int] = None

    def copy(self) -> 'Building':
        """
        Creates a copy of the current building.

        Returns:
            Building: A new building instance with the same configuration.
        """
        return Building(deepcopy(self.__cells), self.__backbone_root,
                   self.__new_router_probability, self.problem)

    @classmethod
    def from_text(cls, shape: Tuple[int, int], backbone: Pos,
                  text: str, problem: Optional[GenericRouterProblem]) -> 'Building':
        """
        Creates a Building instance from a text representation.

        Args:
            shape (Tuple[int, int]): The dimensions of the grid (rows, columns).
            backbone (Pos): The position of the backbone.
            text (str): A string representing the building's grid.
            problem (Optional[GenericRouterProblem]): The problem associated with the building.

        Returns:
            Building: A new Building instance created from the text representation.

        Raises:
            ProblemLoadError: If there is an error loading the building configuration.
        """
        rows, columns = shape
        if rows < 1 or columns < 1:
            raise ProblemLoadError(f'Invalid building size {rows}x{columns}')
        if backbone[0] < 0 or backbone[0] >= rows or backbone[1] < 0 or backbone[1] >= columns:
            raise ProblemLoadError(f'Invalid backbone position ({backbone[0], backbone[1]})')

        text = text.replace('\n', '')
        if len(text) != rows * columns:
            raise ProblemLoadError('Invalid text size')

        cells = np.frombuffer(text.encode(), dtype=np.uint8)
        cells = cells.reshape((rows, columns)).copy()

        for (row, col), cell_byte in np.ndenumerate(cells):
            cell = int(cell_byte)
            if cell in cls.CHAR_CELL_MAP:
                cells[row, col] = cls.CHAR_CELL_MAP[cell].value
            else:
                raise ProblemLoadError(f'Invalid character in text \'{chr(cell)}\'')

        cells[backbone] |= cls.BACKBONE_BIT

        return cls(cells, backbone, 0.8, problem)

    def load_solution(self, text: str) -> None:
        """
        Loads a solution for the building from a text representation.

        Args:
            text (str): A string representing the solution.
        """
        lines = text.split('\n')
        num_backbones = int(lines[0])
        for i in range(1, num_backbones + 1):
            row, col = map(int, lines[i].split())
            self.__cells[row, col] |= self.BACKBONE_BIT
        num_routers = int(lines[num_backbones + 1])
        router_cells = []
        for i in range(num_backbones + 2, num_backbones + 2 + num_routers):
            row, col = map(int, lines[i].split())
            router_cells.append((row, col))
            self.__cells[row, col] |= self.ROUTER_BIT
            self.cover_neighbors(row, col)

    @property
    def rows(self) -> int:
        """
        Returns the number of rows in the building grid.

        Returns:
            int: The number of rows.
        """
        return self.__cells.shape[0]

    @property
    def columns(self) -> int:
        """
        Returns the number of columns in the building grid.

        Returns:
            int: The number of columns.
        """
        return self.__cells.shape[1]

    @property
    def shape(self) -> Tuple[int, int]:
        """
        Returns the shape of the building grid as a tuple (rows, columns).

        Returns:
            Tuple[int, int]: The shape of the grid.
        """
        return cast(Tuple[int, int], self.__cells.shape)

    @property
    def backbone(self) -> Pos:
        """
        Returns the position of the backbone.

        Returns:
            Pos: The position of the backbone in the grid.
        """
        return self.__backbone_root

    def get_routers(self) -> List[Tuple[int, int]]:
        """
        Returns a list of positions of all routers in the building.

        Returns:
            List[Tuple[int, int]]: A list of positions of routers.
        """
        return list(zip(*np.where(self.__cells & self.ROUTER_BIT)))

    def get_target_cells(self) -> List[Tuple[int, int]]:
        """
        Returns a list of positions of all target cells in the building.

        Returns:
            List[Tuple[int, int]]: A list of positions of target cells.
        """
        return list(zip(*np.where(self.__cells & self.CELL_TYPE_MASK == CellType.TARGET.value)))

    def __str__(self) -> str:
        """
        Returns a string representation of the building grid.

        Returns:
            str: A string representation of the grid.
        """
        return '\n'.join(''.join(map(chr, row)) for row in self.__cells)

    def as_nparray(self) -> CellArray:
        """
        Returns a copy of the building grid as a NumPy array.

        Returns:
            CellArray: A copy of the grid as a NumPy array.
        """
        return self.__cells.copy()

    def as_nparray_transposed(self) -> CellArray:
        """
        Returns the transposed version of the cell array.

        This method transposes the current cell array and returns the result.
        
        Returns:
            CellArray: The transposed cell array.
        """
        return self.__cells.transpose()

    def iter(self) -> Iterator[Tuple[int, int, int]]:
        """
        Iterates over the cells in the grid and yields their row, column, and value.

        This method uses numpy's ndenumerate to iterate over the cells of the grid,
        yielding the row, column, and cell value (converted to an integer).
        
        Yields:
            Tuple[int, int, int]: A tuple of the row index, column index,
            and cell value as an integer.
        """
        return ((row, column, int(cell)) for (row, column), cell in np.ndenumerate(self.__cells))

    def get_connected_routers(self, root: Pos) -> Set[Pos]:
        """
        Finds all routers connected to the given root position.

        Uses a breadth-first search (BFS) to traverse through the
        grid starting from the root position
        and collects all connected routers.

        Args:
            root (Pos): The starting position of the root router.

        Returns:
            Set[Pos]: A set of positions representing all routers connected to the root.
        """
        routers = set()
        backbones = set()
        queue = deque([root])
        backbones.add(root)
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

        while queue:
            row, col = queue.popleft()
            if (self.__cells[row, col] & self.ROUTER_BIT) != 0:
                routers.add((row,col))
            for dr, dc in directions:
                nr, nc = row + dr, col + dc
                if 0 <= nr < self.rows and 0 <= nc < self.columns and \
                        (nr, nc) not in backbones and \
                        self.__cells[nr, nc] & self.BACKBONE_BIT:
                    queue.append((nr, nc))
                    backbones.add((nr, nc))
        return routers

    def cover_neighbors(self, row: int, col: int) -> None:
        """
        Updates the coverage of neighbors around the specified cell.

        This method calculates the coverage area of the neighbors around the given cell 
        and marks them as covered if they are not blocked by walls.
        
        Args:
            row (int): The row index of the target cell.
            col (int): The column index of the target cell.
        """
        assert self.problem is not None
        rrange = self.problem.router_range

        # Define the boundaries for the neighborhood based on router's range
        row_start = max(0, row - rrange)
        row_len = min(self.__cells.shape[0] - row_start, 2 * rrange + 1)
        col_start = max(0, col - rrange)
        col_len = min(self.__cells.shape[1] - col_start, 2 * rrange + 1)

        # Calculate center position of the router within the neighborhood
        ctr_row = row - row_start
        ctr_col = col - col_start

        # Initialize a neighborhood matrix to store the coverage status
        neighborhood = np.zeros((row_len, col_len), dtype=np.uint8)
        neighborhood |= self.__cells[row_start:row_start + row_len, \
            col_start:col_start + col_len] & Building.CELL_TYPE_MASK

        # Mark the router itself as covered
        neighborhood[ctr_row, ctr_col] |= self.COVERED_BIT

        # Iterate over the four cardinal directions (left, right, up, down)
        # to mark neighbors as covered
        line_iters = [
            ((ctr_row, ncol) for ncol in range(ctr_col + 1, col_len)),
            ((ctr_row, ncol) for ncol in range(ctr_col - 1, -1, -1)),
            ((nrow, ctr_col) for nrow in range(ctr_row + 1, row_len)),
            ((nrow, ctr_col) for nrow in range(ctr_row - 1, -1, -1))
        ]

        # For each line direction, mark neighbors until a wall is encountered
        for line_iter in line_iters:
            for nrow, ncol in line_iter:
                if neighborhood[nrow, ncol] & self.CELL_TYPE_MASK == CellType.WALL.value:
                    break # Stop if a wall is encountered
                neighborhood[nrow, ncol] |= self.COVERED_BIT

        # Iterate over the four diagonal directions and mark neighbors
        # if they are covered by both row and column
        square_iters = [
            (((nrow, ncol) for nrow in range(ctr_row - 1, -1, -1)
                for ncol in range(ctr_col - 1, -1, -1)), 1, 1),
            (((nrow, ncol) for nrow in range(ctr_row - 1, -1, -1)
                for ncol in range(ctr_col + 1, col_len)), 1, -1),
            (((nrow, ncol) for nrow in range(ctr_row + 1, row_len)
                for ncol in range(ctr_col - 1, -1, -1)), -1, 1),
            (((nrow, ncol) for nrow in range(ctr_row + 1, row_len)
                for ncol in range(ctr_col + 1, col_len)), -1, -1),
        ]

        # For each diagonal direction, check the coverage based on neighbors
        # in both row and column direction
        for square_iter, rstep, cstep in square_iters:
            for nrow, ncol in square_iter:
                if neighborhood[nrow, ncol] & self.CELL_TYPE_MASK != CellType.WALL.value \
                    and neighborhood[nrow + rstep, ncol] & self.COVERED_BIT \
                    and neighborhood[nrow, ncol + cstep] & self.COVERED_BIT:
                    neighborhood[nrow, ncol] |= self.COVERED_BIT

        # Update the cells with the new coverage information
        self.__cells[row_start:row_start + row_len, col_start:col_start + col_len] |= neighborhood

    def place_router(self, row: int, column: int) -> bool:
        """
        Places a router at the specified position and connects it to the backbone.

        This method checks if the position is valid, places a router if possible, 
        and then connects it to the nearest backbone using a breadth-first search (BFS).
        
        Args:
            row (int): The row index of the target cell.
            column (int): The column index of the target cell.
        
        Returns:
            bool: True if the router was successfully placed, False otherwise.
        """

        current_cell = self.__cells[row, column]

        # Check if position is valid (routers cannot be placed inside walls)
        if current_cell & self.CELL_TYPE_MASK == CellType.WALL.value \
            or current_cell & self.CELL_TYPE_MASK == CellType.VOID.value:
            return False

        # Check if router is already placed
        if (current_cell & self.ROUTER_BIT) != 0:
            return False

        # Place the router
        self.__cells[row, column] |= self.ROUTER_BIT

        # Connect the router to the backbone (BFS)
        queue = deque([(row, column)])
        visited = np.zeros((self.rows, self.columns), dtype=bool)
        visited[(row, column)] = True
        parent: Dict[Tuple[int, int], Tuple[int, int]] = {}

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

        while queue:
            r, c = queue.popleft()

            if self.__cells[r, c] & self.BACKBONE_BIT:
                while (r, c) != (row, column):
                    self.__cells[r, c] |= self.BACKBONE_BIT
                    r, c = parent[(r, c)]
                self.__cells[r, c] |= self.BACKBONE_BIT
                break

            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.rows and 0 <= nc < self.columns and not visited[nr, nc]:
                    queue.append((nr, nc))
                    visited[(nr, nc)] = True
                    parent[(nr, nc)] = (r, c)


        # Update router coverage
        self.cover_neighbors(row, column)

        return True

    def update_neighbor_coverage(self, row: int, column: int) -> None:
        """
        Updates the coverage of neighbors around the specified cell
        after a router placement or removal.

        This method recalculates the coverage area of the neighbors surrounding the given router. It 
        marks the neighbors as covered if they are within the router's range
        and are not blocked by walls. 
        The coverage is updated after a router has been added or removed from a position.

        Args:
            row (int): The row index of the target router's position.
            column (int): The column index of the target router's position.
        """

        assert self.problem is not None
        router_range = self.problem.router_range

        # Define the bounds for the area surrounding the router that needs coverage updates
        cell_row_start = max(0, row - router_range)
        cell_row_end = min(self.__cells.shape[0], row + router_range + 1)
        cell_col_start = max(0, column - router_range)
        cell_col_end = min(self.__cells.shape[1], column + router_range + 1)

        # Remove coverage from the previously covered area around the router
        self.__cells[cell_row_start:cell_row_end, cell_col_start:cell_col_end] &= ~self.COVERED_BIT

        # Define the bounds for the router's extended coverage area (double the router range)
        router_row_start = max(0, row - 2 * router_range)
        router_row_end = min(self.__cells.shape[0], row + 2 * router_range + 1)
        router_col_start = max(0, column - 2 * router_range)
        router_col_end = min(self.__cells.shape[1], column + 2 * router_range + 1)

        # Update coverage for each cell in the extended range
        for rrow in range(router_row_start, router_row_end):
            for rcol in range(router_col_start, router_col_end):
                if self.__cells[rrow, rcol] & self.ROUTER_BIT:
                    self.__cells[rrow, rcol] |= self.COVERED_BIT
                    self.cover_neighbors(rrow, rcol)

    def remove_router(self, row: int, column: int) -> bool:
        """
        Removes a router from the specified position and disconnects it from the network.

        This method removes the router at the given position and clears its 
        coverage and backbone connections.
        It then attempts to maintain the integrity of the network by ensuring 
        the remaining routers are still 
        connected to the backbone. The router is removed only if it exists at
        the specified position.

        Args:
            row (int): The row index of the router's position to be removed.
            column (int): The column index of the router's position to be removed.

        Returns:
            bool: True if the router was successfully removed, False
            otherwise (e.g., no router at the position).
        """
        if (self.__cells[row, column] & self.ROUTER_BIT) == 0:
            return False

        # Initialize grid boundaries and movement directions
        max_row, max_col = self.rows, self.columns
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        queue: Deque[Tuple[int, int]] = deque()
        queue.append((row, column))

        # Start the process of removing the router and clearing its connections
        while queue:
            r, c = queue.popleft()

            # Remove the router and its backbone connection (except for the backbone root)
            self.__cells[r, c] &= ~(self.ROUTER_BIT | self.BACKBONE_BIT) \
                if (r, c) != self.__backbone_root \
                else ~self.ROUTER_BIT

            # Check the neighboring cells to propagate the router removal
            for dr, dc in directions:
                new_r, new_c = r + dr, c + dc
                if 0 <= new_r < max_row and 0 <= new_c < max_col and \
                    (self.__cells[new_r, new_c] & self.BACKBONE_BIT) != 0 and \
                    (self.__cells[new_r, new_c] & self.ROUTER_BIT) == 0 and \
                    (new_r, new_c) != self.__backbone_root:
                    queue.append((new_r, new_c))

        # Reconnect the routers to maintain network integrity
        self.reconnect_routers()
        self.update_neighbor_coverage(row, column)
        return True

    def reconnect_routers(self) -> None:
        routers = list(zip(*np.where(self.__cells & self.BACKBONE_BIT)))
        if not routers:
            return

        def reconstruct_path(pred: Dict[Tuple[int, int], Optional[Tuple[int, int]]],
                             p: Optional[Tuple[int, int]]) -> Set[Tuple[int, int]]:
            res = set()
            while p:
                res.add(p)
                p = pred.get(p, None)
            return res

        def steiner_tree(grid: CellArray, terminals: List[Tuple[int, int]]) -> Set[Tuple[int, int]]:
            rows, cols = len(grid), len(grid[0])
            directions = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)]

            terminal_indices = {t: i for i, t in enumerate(terminals)}
            dsu = DisjointSet(len(terminals))
            queue: SteinerTreeQueue = deque()
            source = {}
            pred = {}
            res = set(terminals)

            for (x, y) in terminals:
                queue.append(((x, y), (x, y), None, None))

            while queue and dsu.forests > 1:
                (x, y), src, p1, p2 = queue.popleft()

                if (x,y) not in source:
                    source[(x, y)] = src
                    pred[(x, y)] = p1

                    for dx, dy in directions:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < rows and 0 <= ny < cols and (nx,ny) not in source:
                            queue.append(((nx, ny), src, (x, y), None))

                    continue

                if dsu.connected(terminal_indices[source[(x,y)]], terminal_indices[src]):
                    continue

                if (x, y) in terminals:
                    dsu.union(terminal_indices[source[(x, y)]], terminal_indices[src])
                    res |= reconstruct_path(pred, p1)
                    res |= reconstruct_path(pred, p2)
                else:
                    queue.append((source[(x, y)], src, p1, (x, y)))

            return res

        tree_cells = steiner_tree(self.__cells, [self.__backbone_root] + routers)

        for row, col in tree_cells:
            self.__cells[row, col] |= self.BACKBONE_BIT

    def get_neighborhood(self) -> Iterator[Operator]:
        """
        Generates neighboring building configurations by placing or removing routers.

        This method shuffles the list of routers and target cells, then iteratively
        creates new building configurations by either placing a router in a target cell
        or removing a router from its current position. The decision to place or remove
        a router is based on a predefined probability.

        Yields:
            Building: A new building configuration with a router placed or removed.
        """
        routers = self.get_routers()
        targets = self.get_target_cells()
        random.shuffle(routers)
        random.shuffle(targets)

        while routers or targets:
            rand_num = random.random()

            if (rand_num < self.__new_router_probability and targets) or not routers:
                row, col = targets.pop()
                yield Operator(True, row, col)
            else:
                row, col = routers.pop()
                yield Operator(False, row, col)

    def get_placement_neighborhood(self) -> Iterator[Operator]:
        """
        Yields a sequence of `Operator` objects representing potential placement positions 
        for routers in the grid. The targets are shuffled to introduce randomness.

        Yields:
            Operator: The operator corresponding to a potential placement at a target cell.
        """
        targets = self.get_target_cells()
        random.shuffle(targets)

        for row, col in targets:
            yield Operator(True, row, col)

    def crossover(self, other: 'Building') -> Optional[Tuple['Building', 'Building']]:
        """
        Performs a crossover between the current building and another building, 
        generating two offspring. The crossover is done by swapping subgrids between 
        the two buildings while ensuring they remain valid.

        Args:
            other (Building): The other building to perform the crossover with.

        Returns:
            Optional[Tuple['Building', 'Building']]: A tuple of two new `Building` objects
            created after the crossover. Returns `None` if the resulting buildings are invalid.
        """
        max_row, max_col = self.rows, self.columns

        lower_row = random.randint(0, max_row - 2)
        upper_row = random.randint(lower_row + 1, max_row - 1)
        lower_col = random.randint(0, max_col - 2)
        upper_col = random.randint(lower_col + 1, max_col - 1)

        def clear_edges(grid: CellArray) -> None:
            """
            Clears the edges of a given subgrid, ensuring no backbone cells remain 
            on the edges.

            Args:
                grid (CellArray): The grid to modify.
            """
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
            queue: Deque[Tuple[int, int]] = deque()

            for r in range(lower_row, upper_row):
                if (grid[r, lower_col] & self.BACKBONE_BIT) != 0:
                    queue.append((r, lower_col))
                if (grid[r, upper_col] & self.BACKBONE_BIT) != 0:
                    queue.append((r, upper_col))
            for c in range(lower_col, upper_col):
                if (grid[lower_row, c] & self.BACKBONE_BIT) != 0:
                    queue.append((lower_row, c))
                if (grid[upper_row, c] & self.BACKBONE_BIT) != 0:
                    queue.append((upper_row, c))

            while queue:
                r, c = queue.popleft()

                grid[r, c] &= ~(self.ROUTER_BIT | self.BACKBONE_BIT) \
                    if (r, c) != self.__backbone_root \
                    else ~self.ROUTER_BIT

                for dr, dc in directions:
                    new_r, new_c = r + dr, c + dc
                    if 0 <= new_r < max_row and 0 <= new_c < max_col and \
                        (grid[new_r, new_c] & self.BACKBONE_BIT) != 0 and \
                        (grid[new_r, new_c] & self.ROUTER_BIT) == 0 and \
                        (new_r, new_c) != self.__backbone_root:
                        queue.append((new_r, new_c))

        stripped_self = self.__cells & ~self.COVERED_BIT
        clear_edges(stripped_self)

        # pylint: disable=protected-access
        stripped_other = other.__cells & ~self.COVERED_BIT
        clear_edges(stripped_other)

        temp_rect = stripped_self[lower_row:upper_row, lower_col:upper_col].copy()
        stripped_self[lower_row:upper_row, lower_col:upper_col] = \
            stripped_other[lower_row:upper_row, lower_col:upper_col]
        stripped_other[lower_row:upper_row, lower_col:upper_col] = temp_rect

        assert self.problem is not None

        child1 = Building(stripped_self, self.__backbone_root,
                          self.__new_router_probability, self.problem)
        child1.reconnect_routers()
        if not self.problem.check_budget(child1):
            return None

        # pylint: disable=protected-access
        child2 = Building(stripped_other, self.__backbone_root,
                          self.__new_router_probability, self.problem)
        child2.reconnect_routers()
        if not self.problem.check_budget(child2):
            return None

        for router in child1.get_routers():
            child1.cover_neighbors(*router)
        for router in child2.get_routers():
            child2.cover_neighbors(*router)

        return child1, child2

    def is_same(self, other: 'Building') -> bool:
        """
        Compares the current building with another building to check if they are identical.

        Args:
            other (Building): The other building to compare with.

        Returns:
            bool: `True` if the buildings are the same, `False` otherwise.
        """
        # pylint: disable=protected-access
        return np.array_equal(self.__cells, other.__cells)

    def get_num_targets(self) -> int:
        """
        Returns the number of target cells in the building grid.

        Returns:
            int: The number of target cells.
        """
        return np.count_nonzero(self.__cells & self.CELL_TYPE_MASK == CellType.TARGET.value)

    @override
    def get_num_routers(self) -> int:
        """
        Returns the number of router cells in the building grid.

        Returns:
            int: The number of router cells.
        """
        return np.count_nonzero(self.__cells & self.ROUTER_BIT)

    @override
    def get_num_connected_cells(self) -> int:
        """
        Returns the number of connected backbone cells in the building grid, excluding the root.

        Returns:
            int: The number of connected backbone cells.
        """
        return np.count_nonzero(self.__cells & self.BACKBONE_BIT) - 1

    @override
    def get_coverage(self) -> int:
        """
        Returns the number of covered target cells in the building grid.

        Returns:
            int: The number of covered target cells.
        """
        return np.count_nonzero(self.__cells & (self.CELL_TYPE_MASK | self.COVERED_BIT)
                                == CellType.TARGET.value | self.COVERED_BIT)

    def get_num_uncovered_targets(self) -> int:
        """
        Returns the number of uncovered target cells in the building grid.

        Returns:
            int: The number of uncovered target cells.
        """
        return np.count_nonzero(self.__cells & (self.CELL_TYPE_MASK | self.COVERED_BIT)
                                == CellType.TARGET.value)

    @property
    def score(self) -> int:
        """
        Computes and returns the score of the building, if it hasn't been computed yet.

        Returns:
            int: The score of the building.
        """
        if self.__score is None:
            assert self.problem is not None
            self.__score = self.problem.get_score(self)
        return self.__score

    def check_is_valid(self) -> bool:
        """
        Checks if the current building configuration is valid according
        to the problem's constraints.

        Returns:
            bool: `True` if the configuration is valid, `False` otherwise.
        """

        # Check budget
        assert self.problem is not None
        if not self.problem.check_budget(self):
            print('Budget exceeded')
            return False

        # Check if every router is placed on a backbone cell
        routers_without_backbone_count = np.count_nonzero(
            ((self.__cells & self.ROUTER_BIT) != 0) &
            ((self.__cells & self.BACKBONE_BIT) == 0)
        )


        if routers_without_backbone_count > 0:
            print('Routers without backbone')
            return False

        # Check if no routers are placed on walls
        routers_in_walls_count = np.count_nonzero(
            ((self.__cells & self.ROUTER_BIT) != 0) &
            ((self.__cells & self.CELL_TYPE_MASK) == CellType.WALL.value)
        )
        if routers_in_walls_count > 0:
            print('Routers in walls')
            return False

        # Check if the root backbone is connected
        if (self.__cells[self.__backbone_root] & self.BACKBONE_BIT) == 0:
            print('Root backbone not connected')
            return False

        # Check if every router is connected to the original backbone
        connected_routers = self.get_connected_routers(self.__backbone_root)
        routers = set(self.get_routers())


        if len(connected_routers) != len(routers):
            print('Routers not connected (length mismatch)')
            return False

        if routers - connected_routers:
            print('Routers not connected (set mismatch)')
            return False

        return True
