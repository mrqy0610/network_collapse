import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from model.mcts_model.gcn import GCN


def set_learning_rate(optimizer, lr):
    """Sets the learning rate to the given value"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class PolicyValueNet(nn.Module):
    def __init__(self, config):
        super(PolicyValueNet, self).__init__()
        self.config = config
        self.device = config.device
        self.graph_node_num = config.preload_graph_node_num
        # set config for hyper parameters
        self.l2_const = 1e-4  # coef of l2 penalty
        # use gcn network to process fixed nodes num graph
        self.gcn = GCN(
            nfeat=self.config.feature_num,
            nhid=self.config.gcn_hidden_size,
            nclass=self.config.gcn_hidden_size,
            dropout=self.config.gcn_dropout,
        )
        # action policy layers
        self.act_fc1 = nn.Linear(config.gcn_hidden_size, config.mcts_network_inner_dim)
        self.act_fc2 = nn.Linear(config.mcts_network_inner_dim, config.mcts_network_inner_dim)
        self.act_fc3 = nn.Linear(config.mcts_network_inner_dim, self.graph_node_num)
        # state value layers
        self.val_fc1 = nn.Linear(config.gcn_hidden_size, config.mcts_network_inner_dim)
        self.val_fc2 = nn.Linear(config.mcts_network_inner_dim, config.mcts_network_inner_dim)
        self.val_fc3 = nn.Linear(config.mcts_network_inner_dim, 1)

    def forward(self, features, adj_matrixs):
        graph_embed = self.gcn(features, adj_matrixs)
        # action policy layers
        x_act = F.relu(self.act_fc1(graph_embed))
        x_act = F.relu(self.act_fc2(x_act))
        x_act = F.log_softmax(self.act_fc3(x_act), dim=-1)
        # state value layers
        x_val = F.relu(self.val_fc1(graph_embed))
        x_val = F.relu(self.val_fc2(x_val))
        x_val = F.relu(self.val_fc3(x_val)) \
            if not self.config.reward_reflect_neg else \
            F.tanh(self.val_fc3(x_val))

        return x_act, x_val

    def policy_value_fn(self, state):
        """
        input: state{'features', 'adj_matrixs', 'nodes'}
        output: a list of (action, probability) tuples for each available
        action and the score of the board state
        """
        available_nodes = state['available_nodes']
        log_act_probs, value = self.forward(
            features=torch.from_numpy(state['features']).to(device=self.device, dtype=torch.float32).unsqueeze(0),
            adj_matrixs=torch.from_numpy(state['adjacency_matrix']).to(device=self.device, dtype=torch.float32).unsqueeze(0),
        )
        act_probs = np.exp(log_act_probs.data.cpu().numpy().flatten())
        act_probs = zip(available_nodes, act_probs[available_nodes])
        value = value.item()

        return act_probs, value

    def policy_value(self, features, adjacency_matrixs):
        """
        input: a batch of features, adjacency_matrixs
        output: a batch of action probabilities and state values
        """
        log_act_probs, value = self.forward(
            features=features,
            adj_matrixs=adjacency_matrixs,
        )
        act_probs = np.exp(log_act_probs.data.cpu().numpy())
        value = value.data.cpu().numpy()

        return act_probs, value

    def train_step(self, batch_features, batch_adjacency_matrixs, batch_actions_probs, batch_rewards, optimizer, lr):
        """perform a training step"""
        # zero the parameter gradients
        optimizer.zero_grad()
        # set learning rate
        set_learning_rate(optimizer, lr)
        # forward
        log_act_probs, value = self.forward(batch_features, batch_adjacency_matrixs)
        # define the loss = (z - v)^2 - pi^T * log(p) + c||theta||^2
        # Note: the L2 penalty is incorporated in optimizer
        value_loss = F.mse_loss(value, batch_rewards)
        policy_loss = -torch.mean(torch.sum(batch_actions_probs * log_act_probs, 1))
        loss = value_loss + policy_loss
        # backward and optimize
        loss.backward()
        optimizer.step()
        # calc policy entropy, for monitoring only
        entropy = -torch.mean(torch.sum(torch.exp(log_act_probs) * log_act_probs, 1))
        # print('----------------------------')
        # print('batch_features', batch_features.shape)
        # print('batch_adjacency_matrixs', batch_adjacency_matrixs.shape)
        # print('batch_actions_probs', batch_actions_probs.shape)
        # print('batch_rewards', batch_rewards.shape)
        # print('log_act_probs', log_act_probs.shape)
        # print('value', value.shape)

        return value_loss.item(), policy_loss.item(), entropy.item()

    def get_weights(self):
        return {k: v.cpu() for k, v in self.state_dict().items()}

    def set_weights(self, weights):
        self.load_state_dict(weights)



