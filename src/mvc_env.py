import random
from src.lib.disjoint_set import DisjointSet


class MvcEnv:

    def __init__(self, norm: float):
        self.CcNum = 1.0
        self.norm = norm
        self.graph = None
        self.num_covered_edges = 0

        self.state_seq = []
        self.act_seq = []
        self.action_list = []
        self.reward_seq = []
        self.sum_rewards = []
        self.covered_set = set()
        self.avail_list = []
        self.node_degrees = []
        self.total_degrees = 0

    def s0(self, g):
        self.graph = g

    def step(self, a: int) -> float:
        assert self.graph
        assert a not in self.covered_set
        self.act_seq.append(a)
        self.covered_set.add(a)
        self.action_list.append(a)

        for neigh in self.graph.adj_list[a]:
            if neigh not in self.covered_set:
                self.num_covered_edges += 1

        r_t = self.get_reward()
        self.reward_seq.append(r_t)
        self.sum_rewards.append(r_t)

        return r_t

    def step_without_reward(self, a: int):
        assert self.graph
        assert a not in self.covered_set
        self.covered_set.add(a)
        self.action_list.append(a)

        for neigh in self.graph.adj_list[a]:
            if a not in neigh:
                self.num_covered_edges += 1

    # random
    def random_action(self) -> int:
        assert self.graph
        self.avail_list.clear()
        for i in range(self.graph.num_nodes):
            if i not in self.covered_set:
                useful = False
                for neigh in self.graph.adj_list[i]:
                    if neigh not in self.covered_set:
                        useful = True
                        break
                if useful:
                    self.avail_list.append(i)

        assert len(self.avail_list)
        idx = random.randint(0, len(self.avail_list) - 1)
        return self.avail_list[idx]

    # def between_action(self) -> int:
    #     assert self.graph
    #     id2node = {}
    #     node2id = {}
    #     adj_dic_origin = {}
    #     adj_list_reID = []
    #
    #     for i in range(self.graph.num_nodes):
    #         if i not in self.covered_set:
    #             for neigh in self.graph.adj_list[i]:
    #                 if neigh in self.covered_set:
    #                     if i != adj_dic_origin[-1]:
    #                         adj_dic_origin[i]
    #
    #     return

    def is_terminal(self) -> bool:
        assert self.graph
        return self.graph.num_nodes == self.num_covered_edges

    def get_reward(self) -> float:
        return - self.get_max_connected_nodes_num() / (self.graph.num_nodes * self.graph.num_nodes)

    # 源代码中注释掉的reward
    # def get_reward(self, oldCcNum) -> float:
    #     return (self.CcNum - oldCcNum) / self.CcNum * self.graph.num_nodes

    def print_graph(self):
        print("edge_list:")
        print("[", end="")
        for i in range(len(self.graph.edge_list)):
            print("[{}, {}]".format(self.graph.edge_list[i][0], self.graph.edge_list[i][1]), end="")
        print("]")

    def get_num_of_connected_components(self) -> float:
        assert self.graph
        disjoint_set = DisjointSet(self.graph.num_nodes)

        for i in range(self.graph.num_nodes):
            if i not in self.covered_set:
                for neigh in self.graph.adj_list[i]:
                    if neigh not in self.covered_set:
                        disjoint_set.merge(i, neigh)

        lccIDs = set()
        for i in range(self.graph.num_nodes):
            lccIDs.add(disjoint_set.union_set[i])
        return len(lccIDs)

    def get_max_connected_nodes_num(self) -> float:
        assert self.graph
        disjoint_set = DisjointSet(self.graph.num_nodes)
        for i in range(self.graph.num_nodes):
            if i not in self.covered_set:
                for neigh in self.graph.adj_list[i]:
                    if neigh not in self.covered_set:
                        disjoint_set.merge(i, neigh)

        return disjoint_set.max_rank_count
