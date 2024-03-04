import math
import torch
import functools
import numpy as np
import torch.nn.functional as F

from torch import nn
from model.graphsage.graphsage import GraphSage
from model.attention_model.attention_model import AttentionModel
from model.attention_model.attention_utils import set_decode_type, calculate_reconstruct_loss


class Lower_Actor:
    def __init__(self, config):
        self.config = config
        self.device = config.device
        self.feature_num = config.feature_num
        self.lower_agent_type = config.lower_agent_type

    def get_sequence_actions(self, actions, features, adj_lists, nodes):
        if self.lower_agent_type == 'degree_greedy':
            all_sequence_actions = []
            for action, adj_list in zip(actions, adj_lists):
                candidate_a = []
                for a in action.cpu().numpy():
                    candidate_a.append({'key': a, 'degree': len(adj_list[a])})
                candidate_a = sorted(candidate_a, key=functools.cmp_to_key(
                    lambda a, b: 1 if a['degree'] <= b['degree'] else -1))
                all_sequence_actions.append(torch.tensor([a['key'] for a in candidate_a]).unsqueeze(0))
            return torch.cat(all_sequence_actions, dim=0).to(self.device)
        else:
            raise NotImplementedError()


class Hierarchy_Actor(nn.Module):
    def __init__(self, config):
        super(Hierarchy_Actor, self).__init__()
        self.config = config
        self.feature_num = config.feature_num
        self.device = config.device

        #################################################
        self.upper_agent = AttentionModel(
            config=config, embedding_dim=config.attention_embedding_dim,
            n_encode_layers=config.n_encode_layers, tanh_clipping=10.,
            mask_inner=True, mask_logits=True, normalization='batch',
        )
        self.lower_agent = Lower_Actor(config=config)
        #################################################

    def get_multi_actions(self, features, adj_lists, nodes, adj_matrixs, decode_type='sampling'):
        # set decode type to sampling for acting
        set_decode_type(self.upper_agent, decode_type)
        # change data to multi batch form
        if not self.config.multi_env:
            features = torch.from_numpy(features).to(dtype=torch.float32, device=self.device).unsqueeze(0)
            adj_lists = [adj_lists]
            nodes = torch.from_numpy(nodes).to(dtype=torch.int32, device=self.device).unsqueeze(0)
            adj_matrixs = torch.from_numpy(adj_matrixs).to(dtype=torch.int32, device=self.device).unsqueeze(0)
        # upper agent act to get actions set
        logprobs, upper_actions, recons_loss = self.upper_agent.get_multi_actions(features, adj_lists, nodes, adj_matrixs)
        """
        logprobs torch.Size([batch_size])
        upper_actions torch.Size([batch_size, seq_len])
        lower_actions torch.Size([batch_size, seq_len])
        """
        # lower agent act to get actions sequence
        lower_actions = self.lower_agent.get_sequence_actions(upper_actions, features, adj_lists, nodes)

        return lower_actions, logprobs, recons_loss


class Critic(nn.Module):
    def __init__(self, config):
        super(Critic, self).__init__()
        self.config = config
        self.feature_num = config.feature_num
        self.device = 'cuda' if config.cuda else 'cpu'
        self.graphsage = GraphSage(
            config=config, feature_num=self.feature_num,
        )
        self.fc1 = nn.Linear(config.graphsage_output_dim, config.qnetwork_inner_dim)
        self.fc2 = nn.Linear(config.qnetwork_inner_dim, config.qnetwork_inner_dim)
        self.fc3 = nn.Linear(config.qnetwork_inner_dim, 1)

    def forward(self, features, adj_lists, nodes, adj_matrixs):
        """
        all_embeds: torch.Size([batch_size, embed_dim])
        """
        all_embeds, all_recons_loss = [], []
        for feature, adj_list, node, adj_matrix in zip(features, adj_lists, nodes, adj_matrixs):
            embed = self.graphsage(feature, adj_list, node)
            recons_loss = calculate_reconstruct_loss(embed, adj_matrix).unsqueeze(0) \
                if self.config.add_graph_reconstruction_loss else None
            embed = embed.mean(0).unsqueeze(0)
            all_embeds.append(embed)
            all_recons_loss.append(recons_loss)
        all_embeds = torch.cat(all_embeds, dim=0)
        all_recons_loss = torch.cat(all_recons_loss, dim=0).mean() if self.config.add_graph_reconstruction_loss else None
        x = F.relu(self.fc1(all_embeds))
        x = F.relu(self.fc2(x))
        v = self.fc3(x)

        return v, all_recons_loss

