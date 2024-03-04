import copy
import torch

from torch.nn import DataParallel
# from typing import NamedTuple
# from utils.boolmask import mask_long2bool, mask_long_scatter


def set_decode_type(model, decode_type):
    if isinstance(model, DataParallel):
        model = model.module
    model.set_decode_type(decode_type)


def calculate_reconstruct_loss(embed, adjacency_matrix):
    # calculate L1 loss
    H = embed
    D = torch.diag(torch.sum(adjacency_matrix, dim=1))
    L = (D - adjacency_matrix).to(dtype=torch.float32)
    recons_loss = 2 * torch.trace(torch.mm(torch.mm(torch.transpose(H, dim0=0, dim1=1), L), H)) \
        if adjacency_matrix.shape[0] > 0 else torch.zeros_like(embed).to(embed.device).sum()

    return recons_loss


def huber_loss(pred, target, d):
    a = (abs(pred - target) <= d).float()
    b = (abs(pred - target) > d).float()
    return a * abs(pred - target) ** 2 / 2 + b * d * (abs(pred - target) - d / 2)


class MultiStepsRecoder:
    # only take into account single batch acting
    def __init__(self, config, nodes):
        """
        nodes torch.Size([batch_size, graph_size])
        visited torch.Size([batch_size, 1, graph_size + 1])
        current_node torch.Size([batch_size, 1])
        first_node torch.Size([batch_size, 1])
        """
        self.config = config
        self.nodes = nodes
        # record steps
        self.steps = 0
        self.num_nodes = self.nodes.shape[1]
        self.visited = torch.zeros(
            self.nodes.shape[0], 1, self.nodes.shape[1] + 1).to(dtype=torch.bool, device=config.device)
        self.current_node = torch.zeros(self.nodes.shape[0], 1).to(config.device)
        self.first_node = torch.zeros(self.nodes.shape[0], 1).to(config.device)

    def get_current_node(self):
        return self.current_node

    def get_first_node(self):
        return self.first_node

    def get_steps(self):
        return self.steps

    def all_finished(self):
        # print('!!!!!!!!!!!!!!!')
        # print('self.current_node.item()', self.current_node.item())
        # print('self.num_nodes', self.num_nodes)
        # print('self.steps', self.steps)
        # finish decoding if remaining one node or reach end of token

        # return self.steps >= self.num_nodes - 1 or \
        #        self.steps >= self.config.max_upper_actions_len or \
        #        self.current_node.item() == self.num_nodes
        # if self.steps >= self.num_nodes - 1:
        #     print('self.steps >= self.num_nodes - 1')
        return self.steps >= self.config.max_upper_actions_len or \
               self.current_node.item() == self.num_nodes

    def reach_end_of_token(self):
        return self.current_node.item() == self.num_nodes

    def get_mask(self):
        mask = copy.deepcopy(self.visited)
        # not allow first step meet end of token
        if self.steps == 0:
            mask[:, :, -1] = True
        # force to meet end of token if only one node
        if self.steps >= self.num_nodes - 1:
            mask[:, :, :] = True
            mask[:, :, -1] = False
        # force to meet end of token if reach max node per action limit
        if self.steps >= self.config.max_upper_actions_len:
            mask[:, :, :] = True
            mask[:, :, -1] = False

        return mask

    def update(self, selected):
        """
        :param selected: torch.size([batch_size])
        :return:
        """
        # update finished nodes
        row = torch.arange(selected.shape[0])
        # check visit unvisited nodes and visit end node
        assert not torch.any(self.visited[row, 0, selected])
        assert not torch.any(self.visited[row, 0, -1])
        finish_nodes = torch.ones_like(self.visited[row, 0, selected]).to(dtype=torch.bool)
        self.visited[row, 0, selected] = finish_nodes
        # update current node and first node
        self.current_node = selected[:, None]  # Add dimension for step
        self.first_node = copy.deepcopy(self.current_node) if self.steps == 0 else self.first_node
        self.steps = self.steps + 1
