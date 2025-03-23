class DisjointSet:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.forests = n

    def find(self, u: int) -> int:
        if self.parent[u] != u:
            self.parent[u] = self.find(self.parent[u])
        return self.parent[u]

    def union(self, u: int, v: int) -> bool:
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
        return self.find(u) == self.find(v)
