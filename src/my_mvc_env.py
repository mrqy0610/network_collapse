import copy
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from src.lib.disjoint_set import DisjointSet
from utils.utils import gen_graph, gen_graph_alltypes


class MvcEnv:
    def __init__(self, config):
        self.config = config
        # my property for env
        # init graph
        self.init_graph()

        # my property for env
        self.connectivity_method = config.connectivity_method
        self.init_connectivity = self.get_connectivity()
        self.feature_num = self.get_features_num()

    def get_features_num(self):
        assert self.graph
        assert len(self.graph.nodes) > 0
        assert 0 in self.graph.nodes

        # return self.graph.nodes[0]['weights'].shape[0]
        return 1 + self.config.use_weights_features

    def reset(self, graph=None):
        self.reset_graph(graph)
        # # render graph if necessary
        if self.config.render_graph:
            self.render_graph()

        return self.get_obs()

    def step(self, action: int):
        # print('action', action)
        # print('self.graph.nodes()', self.graph.nodes())
        ######################### pre-check
        assert self.graph
        assert action in self.graph.nodes()
        # get node weights to calculate reward
        weights = self.graph.nodes[action]['weights'][0]
        # print('weights', weights.shape)
        ######################### execute action
        # remove node from graph
        if self.config.eliminate_action:
            self.graph.remove_node(action)
        # turn target node into orphan node
        else:
            neighbors = list(self.graph.neighbors(action))
            for neighbor in neighbors:
                self.graph.remove_edge(action, neighbor)
        # remove node from  available nodes list
        self.available_nodes.remove(action)
        ######################### get next_obs, reward, done and info
        # calculate reward
        r_t = self.get_reward(weights)
        # get obs
        obs = self.get_obs()
        # get terminal
        done = self.is_terminal()
        # # render graph if necessary
        if self.config.render_graph:
            self.render_graph()
        return obs, r_t, done, None

    def multi_step(self, action):
        assert len(action) > 0
        obs, done, info, now_gamma = None, False, None, 1.0
        r_cum, r_all, r_list = 0.0, 0.0, []
        all_connectivity = []
        for a in action:
            obs, r_t, done, info = self.step(a)
            all_connectivity.append(obs['connectivity'])
            r_cum = r_cum + r_t * now_gamma
            r_all = r_all + r_t
            r_list.append(r_t)
            now_gamma = now_gamma * self.config.reward_gamma
            # early stop step if done(unreachable condition)
            if done:
                break
        obs['all_connectivity'] = all_connectivity

        return obs, {'discount': r_cum, 'sumup': r_all, 'all': r_list}, done, info

    def get_reward(self, weights) -> float:
        now_connectivity = self.get_connectivity()

        return - now_connectivity / self.init_connectivity * weights

    def get_relative_connectivity(self):
        now_connectivity = self.get_connectivity()

        return now_connectivity / self.init_connectivity

    def get_connectivity(self):
        all_components = nx.connected_components(self.graph)
        if self.connectivity_method == 'pairwise':
            connectivity = 0.
            for comp in all_components:
                connectivity += len(comp) * (len(comp) - 1) / 2
        elif self.connectivity_method == 'gcc':
            connectivity = 0.
            for comp in all_components:
                connectivity = max(len(comp), connectivity)
        else:
            raise NotImplementedError()
        return connectivity

    def get_obs(self):
        # get attributes matrix
        nodes = np.array([i for i in self.graph.nodes()])
        # get adjacency matrix
        adjacency_matrix = nx.adjacency_matrix(self.graph).toarray() if len(self.graph.nodes()) > 0 else np.array([])
        # get adjacency lists
        adjacency_list = nx.to_dict_of_lists(self.graph)
        adjacency_list = {key: set(adjacency_list[key]) for key in adjacency_list}

        # features = np.array([self.graph.nodes[i]['weights'] for i in self.graph.nodes()])
        # for i in self.graph.nodes:
        #     print('features', self.graph.nodes[i]['weights'])
        #     print('adjacency_list', adjacency_list[i])
        #     print('adjacency_list', np.array([len(adjacency_list[i])]))
        #     print('cat_feature', np.concatenate([np.array([len(adjacency_list[i])]), self.graph.nodes[i]['weights']], axis=0))
        # features = np.array([
        #     self.graph.nodes[i]['weights'] if i in self.graph.nodes else np.zeros(self.feature_num)
        #     for i in range(self.init_nodes_num)
        # ])
        if self.config.use_weights_features:
            features = np.array([
                np.concatenate([np.array([len(adjacency_list[i])]), self.graph.nodes[i]['weights']], axis=0)
                if i in self.graph.nodes else np.zeros(self.feature_num)
                for i in range(self.init_nodes_num)
            ])
        else:
            features = np.array([
                np.array([len(adjacency_list[i])]) if i in self.graph.nodes else np.zeros(self.feature_num)
                for i in range(self.init_nodes_num)
            ])
        # available_nodes: nodes have not been removed
        available_nodes = np.array(self.available_nodes)
        # relative connectivity
        connectivity = self.get_relative_connectivity()

        return {
            'origin_num_nodes': self.init_nodes_num,
            'nodes': nodes,
            'features': features,
            'adjacency_matrix': adjacency_matrix,
            'adjacency_list': adjacency_list,
            'available_nodes': available_nodes,
            'connectivity': connectivity,
        }

    def is_terminal(self) -> bool:
        """
        :return:
            1. all nodes are orphan node
            2. nodes num equal or less than 1
        """
        if self.config.early_stop_env:  # env get stop when all nodes become orphan node
            adjacency_list = nx.to_dict_of_lists(self.graph)
            # end episode if all nodes are orphan node
            not_orphan_graph = False
            for node in adjacency_list:
                not_orphan_graph = (len(adjacency_list[node]) > 0)
                if not_orphan_graph:
                    break
            orphan_graph = not not_orphan_graph

            return orphan_graph or len(self.graph.nodes) <= 1
        else:  # env get stop when all nodes have been chosen
            return len(self.available_nodes) <= 1

    def init_graph(self):
        # generate graph for env
        self.graph = gen_graph(
            self.config.min_nodes, self.config.max_nodes, self.config.graph_type,
        ) if self.config.fix_graph or self.config.fix_graph_type else gen_graph_alltypes(
            self.config.min_nodes, self.config.max_nodes,
        )
        # config attributes matrix for graph
        attributes = np.array([1 / len(self.graph.nodes()) for _ in range(len(self.graph.nodes()))]) \
            if not self.config.diff_nodes_weights else np.random.rand(len(self.graph.nodes()))
        attributes = (attributes / np.sum(attributes, axis=-1)).reshape(-1, 1)
        attributes = np.concatenate([attributes], axis=1)
        nx.set_node_attributes(self.graph, {i: attributes[i] for i in range(len(self.graph.nodes))}, 'weights')
        # record init grap to reset env
        self.origin_graph = copy.deepcopy(self.graph)
        # record init nodes num
        self.init_nodes_num = len(self.graph.nodes())
        # record available nodes that are not removed
        self.available_nodes = [i for i in self.graph.nodes()]

    def reset_graph(self, graph=None):
        # use pre-load graph
        if graph is not None:
            self.graph = graph
        # use same graph for training
        elif self.config.fix_graph:
            self.graph = copy.deepcopy(self.origin_graph)
        # use same graph type for trainings
        elif self.config.fix_graph_type:
            self.graph = gen_graph(
                self.config.min_nodes, self.config.max_nodes, self.config.graph_type,
            )
        # use multi graph type for trainings
        else:
            self.graph = gen_graph_alltypes(
                self.config.min_nodes, self.config.max_nodes,
            )
        # update init_connectivity
        self.init_connectivity = self.get_connectivity()
        # save graph pos for rendering
        self.graph_pos = None
        # config attributes matrix for graph
        attributes = np.array([1 / len(self.graph.nodes()) for _ in range(len(self.graph.nodes()))]) \
            if not self.config.diff_nodes_weights else np.random.rand(len(self.graph.nodes()))
        attributes = (attributes / np.sum(attributes, axis=-1)).reshape(-1, 1)
        attributes = np.concatenate([attributes], axis=1)
        nx.set_node_attributes(self.graph, {i: attributes[i] for i in range(len(self.graph.nodes))}, 'weights')
        # record init grap to reset env
        self.origin_graph = copy.deepcopy(self.graph)
        # record init nodes num
        self.init_nodes_num = len(self.graph.nodes())
        # record available nodes that are not removed
        self.available_nodes = [i for i in self.graph.nodes()]

    def render_graph(self):
        render = nx.spring_layout(self.graph) if self.graph_pos is None else self.graph_pos
        self.graph_pos = render if self.graph_pos is None else self.graph_pos
        nx.draw(self.graph, render, font_size=4, node_color='black')
        nx.draw_networkx_edges(self.graph, render, width=1, edge_color='b')
        plt.show()

    ############################################ not use
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
