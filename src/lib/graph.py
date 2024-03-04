import random

from utils.pair import Pair


class Graph:

    def __init__(self, num_nodes: int = 0, num_edges: int = 0, edges_from: list = [], edges_to: list = []):
        self.adj_list = []
        self.edge_list = []

        self.num_nodes = num_nodes
        self.num_edges = num_edges

        for i in range(num_nodes):
            self.adj_list.append([])

        for i in range(num_edges):
            x = edges_from[i]
            y = edges_to[i]
            self.adj_list[x].append(y)
            self.adj_list[y].append(x)
            self.edge_list.append(Pair(edges_from[i], edges_to[i]))

    def get_two_rank_neighbors_ratio(self, covered: list) -> float:
        temp_set = set()
        for i in covered:
            temp_set.add(i)
        sum = 0
        for i in range(self.num_nodes):
            if i not in temp_set:
                for j in range(i + 1, self.num_nodes):
                    if j not in temp_set:
                        v3 = [element for element in self.adj_list[i] if element in self.adj_list[j]]
                        if len(v3) > 0:
                            sum += 1.0
        return sum


class GSet:
    def __init__(self):
        self.graph_pool = {}

    def insert_graph(self, gid: int, graph: Graph):
        assert gid not in self.graph_pool
        self.graph_pool[gid] = graph

    def sample(self):
        assert len(self.graph_pool)
        gid = random.randint(0, len(self.graph_pool) - 1)
        assert self.graph_pool[gid]
        return self.graph_pool[gid]

    def get(self, gid: int):
        assert gid in self.graph_pool
        return self.graph_pool[gid]

    def clear(self):
        self.graph_pool.clear()
