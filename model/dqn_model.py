import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from model.graphsage.graphsage import GraphSage


class Q_Network_Graph(nn.Module):
    def __init__(self, config):
        """
        Initialize a deep Q-learning network for testing algorithm
            in_features: number of features of input.
            num_actions: number of action-value to output, one-to-one correspondence to action in game.
        """
        super(Q_Network_Graph, self).__init__()
        self.config = config
        self.feature_num = config.feature_num
        self.device = 'cuda' if config.cuda else 'cpu'
        self.graphsage = GraphSage(
            config=config, feature_num=self.feature_num,
        )
        # self.action_embeddings = nn.Embedding(config.max_nodes, config.action_embed_dim)
        self.fc1 = nn.Linear(config.graphsage_output_dim + config.action_embed_dim, config.qnetwork_inner_dim)
        self.fc2 = nn.Linear(config.qnetwork_inner_dim, config.qnetwork_inner_dim)
        self.fc3 = nn.Linear(config.qnetwork_inner_dim, 1)
        # copy model to cuda
        self.to(self.device)

    def forward(self, actions, features, adj_lists, nodes):
        """
        embeds torch.Size([node_num, embed_dim])
        action_embed torch.Size([1, embed_dim])
        """
        all_embeds = []
        for action, feature, adj_list, node in zip(actions, features, adj_lists, nodes):
            embeds = self.graphsage(feature, adj_list, node)
            # get all nodes embed
            index = torch.nonzero(torch.eq(node, action.expand_as(node))).reshape(-1)
            # get action embed
            action_embed = embeds[index]
            all_embeds.append(torch.cat([embeds.mean(dim=0).unsqueeze(0), action_embed], dim=-1))
        x = torch.cat(all_embeds, )
        # actions = self.action_embeddings(actions)
        # actions = actions.unsqueeze(0) if len(actions.shape) == 1 else actions
        # x = torch.cat([x, actions], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

    def get_q_value(self, action, features, adj_lists, nodes, tensor_data=False):
        if not tensor_data:
            features = torch.from_numpy(features).to(dtype=torch.float32, device=self.device).unsqueeze(0)
            adj_lists = [adj_lists]
            nodes = torch.from_numpy(nodes).to(dtype=torch.int32).unsqueeze(0)
            action = torch.from_numpy(action).to(dtype=torch.long, device=self.device)

        return self.forward(action, features, adj_lists, nodes)

    def get_actions(self, features, adj_lists, nodes, tensor_data=False):
        max_a, max_q = -1, -math.inf
        for a in nodes:
            q = self.get_q_value(np.array([a]), features, adj_lists, nodes, tensor_data=tensor_data).item()
            max_a = a if q > max_q else max_a
            max_q = max(q, max_q)
        if len(nodes) == 0:
            max_q = 0.

        return max_a, max_q

    def evaluate_action(self, actions, features, adj_lists, nodes):
        q = self.get_q_value(actions.squeeze(-1), features, adj_lists, nodes, tensor_data=True)

        return q

    def get_max_q(self, features, adj_lists, nodes, tensor_data=False):
        all_max_q = []
        for feature, adj_list, node in zip(features, adj_lists, nodes):
            max_a, max_q = -1, -math.inf
            feature = feature.unsqueeze(0)
            adj_list = [adj_list]
            node = node.unsqueeze(0)
            for a in node[0]:
                q = self.get_q_value(
                    torch.from_numpy(np.array(a)).to(dtype=torch.long, device=self.device).unsqueeze(0),
                    feature, adj_list, node, tensor_data=tensor_data).item()
                max_a = a if q > max_q else max_a
                max_q = max(q, max_q)
            if len(node[0]) == 0:
                max_q = 0.
            all_max_q.append(max_q)

        return torch.from_numpy(np.array(all_max_q)).unsqueeze(-1).to(device=self.config.device, dtype=torch.float32)



