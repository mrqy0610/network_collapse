class DisjointSet:
    def __init__(self, graph_size: int):
        self.union_set = []
        self.rank_count = []
        for i in range(graph_size):
            self.union_set.append(i)
            self.rank_count.append(1)
        self.max_rank_count = 1

    def find_root(self, node: int) -> int:
        if node != self.union_set[node]:
            root_node = self.find_root(self.union_set[node])
            self.union_set[node] = root_node
            return root_node
        else:
            return node

    def merge(self, node1: int, node2: int):
        node1_root = self.find_root(node1)
        node2_root = self.find_root(node2)
        if node1_root != node2_root:
            if self.rank_count[node2_root] > self.rank_count[node1_root]:
                self.union_set[node1_root] = node2_root
                self.rank_count[node2_root] += self.rank_count[node1_root]

                if self.rank_count[node2_root] > self.max_rank_count:
                    self.max_rank_count = self.rank_count[node2_root]
            else:
                self.union_set[node2_root] = node1_root
                self.rank_count[node1_root] += self.rank_count[node2_root]

                if self.rank_count[node1_root] > self.max_rank_count:
                    self.max_rank_count = self.rank_count[node1_root]

    def get_biggest_component_current_ratio(self) -> float:
        return self.max_rank_count / len(self.rank_count)

    def get_rank(self, root_node: int) -> int:
        return self.rank_count[root_node]
