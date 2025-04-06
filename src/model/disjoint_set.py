class DisjointSet:
    '''
    A class to represent a disjoint set (or union-find) data structure.
    
    It supports union and find operations, along with path compression
    and union by rank optimizations.
    '''

    def __init__(self, n: int):
        '''
        Initializes the disjoint set with n elements.
        Each element is its own parent, and the rank is initialized to 0.

        Args:
            n (int): The number of elements in the disjoint set.
        '''
        self.parent = list(range(n))
        self.rank = [0] * n
        self.forests = n

    def find(self, u: int) -> int:
        '''
        Finds the representative of the set containing u.
        Implements path compression to optimize future queries.
        
        Args:
            u (int): The element to find the representative for.
        '''
        if self.parent[u] != u:
            self.parent[u] = self.find(self.parent[u])
        return self.parent[u]

    def union(self, u: int, v: int) -> bool:
        '''
        Unites the sets containing u and v.
        Implements union by rank to optimize the union operation.
        
        Args:
            u (int): The first element to unite.
            v (int): The second element to unite.
        
        Returns:
            bool: True if the sets were united, False if they were already connected.
        '''
        root_u, root_v = self.find(u), self.find(v)
        if root_u != root_v:
            if self.rank[root_u] > self.rank[root_v]:
                self.parent[root_v] = root_u
            elif self.rank[root_u] < self.rank[root_v]:
                self.parent[root_u] = root_v
            else:
                self.parent[root_v] = root_u
                self.rank[root_u] += 1
            self.forests -= 1
            return True
        return False

    def connected(self, u: int, v: int) -> bool:
        '''
        Checks if u and v are in the same set.  

        Args:
            u (int): The first element to check.
            v (int): The second element to check.
        
        Returns:
            bool: True if u and v are in the same set, False otherwise.
        '''        
        return self.find(u) == self.find(v)
