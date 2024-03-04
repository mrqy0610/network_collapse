import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

import random

"""
Set of modules for aggregating embeddings of neighbors.
"""


class MeanAggregatorHead(nn.Module):
    """
    Aggregates a node's embeddings using mean of neighbors' embeddings
    """

    def __init__(self, cuda=False, gcn=False):
        """
        Initializes the aggregator for a specific graph.
        cuda -- whether to use GPU
        gcn --- whether to perform concatenation GraphSAGE-style, or add self-loops GCN-style
        """
        super(MeanAggregatorHead, self).__init__()
        self.cuda = cuda
        self.gcn = gcn
        self.first_agg = True

    def forward(self, map_func, nodes, to_neighs, num_sample=10):
        """
        nodes --- list of nodes in a batch
        to_neighs --- list of sets, each set is the set of neighbors for node in batch
        num_sample --- number of neighbors to sample. No sampling if None.
        """
        # Local pointers to functions (speed hack)
        _set = set
        if not num_sample is None:
            _sample = random.sample
            samp_neighs = [_set(_sample(
                to_neigh, num_sample, )) if len(to_neigh) >= num_sample else to_neigh for to_neigh in to_neighs]
        else:
            samp_neighs = to_neighs

        if self.gcn:
            samp_neighs = [samp_neigh.union(set(nodes[i].unsqueeze(0).tolist())) for i, samp_neigh in enumerate(samp_neighs)]
        unique_nodes_list = list(set.union(*samp_neighs))
        unique_nodes = {n: i for i, n in enumerate(unique_nodes_list)}
        mask = Variable(torch.zeros(len(samp_neighs), len(unique_nodes)))
        column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]
        row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]
        mask[row_indices, column_indices] = 1
        if self.cuda:
            mask = mask.cuda()
        num_neigh = mask.sum(1, keepdim=True)
        mask = mask.div(num_neigh)
        if self.cuda:
            embed_matrix = map_func(torch.LongTensor(unique_nodes_list).cuda())
        else:
            embed_matrix = map_func(torch.LongTensor(unique_nodes_list))
        to_feats = mask.mm(embed_matrix)

        torch.set_printoptions(threshold=np.inf)

        return to_feats


class MeanAggregator(nn.Module):
    """
    Aggregates a node's embeddings using mean of neighbors' embeddings
    """

    def __init__(self, cuda=False, gcn=False):
        """
        Initializes the aggregator for a specific graph.
        cuda -- whether to use GPU
        gcn --- whether to perform concatenation GraphSAGE-style, or add self-loops GCN-style
        """
        super(MeanAggregator, self).__init__()
        self.cuda = cuda
        self.gcn = gcn
        self.first_agg = False

    def forward(self, map_func, features, adj_lists, nodes, to_neighs, num_sample=10):
        """
        nodes --- list of nodes in a batch
        to_neighs --- list of sets, each set is the set of neighbors for node in batch
        num_sample --- number of neighbors to sample. No sampling if None.
        """
        # Local pointers to functions (speed hack)
        _set = set
        if not num_sample is None:
            _sample = random.sample
            samp_neighs = [_set(_sample(
                to_neigh, num_sample, )) if len(to_neigh) >= num_sample else to_neigh for to_neigh in to_neighs]
        else:
            samp_neighs = to_neighs
        # add self node to neighbor nodes
        if self.gcn:
            # for i, samp_neigh in enumerate(samp_neighs):
            #     print('test_type', nodes[i].unsqueeze(0))
            samp_neighs = [samp_neigh.union(set(nodes[i].unsqueeze(0).tolist())) for i, samp_neigh in enumerate(samp_neighs)]
        unique_nodes_list = list(set.union(*samp_neighs))
        unique_nodes = {n: i for i, n in enumerate(unique_nodes_list)}
        mask = Variable(torch.zeros(len(samp_neighs), len(unique_nodes)))
        column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]
        row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]
        mask[row_indices, column_indices] = 1
        if self.cuda:
            mask = mask.cuda()
        num_neigh = mask.sum(1, keepdim=True)
        mask = mask.div(num_neigh)
        if self.cuda:
            embed_matrix = map_func(
                features, adj_lists,
                torch.LongTensor(unique_nodes_list).cuda()
            )
        else:
            embed_matrix = map_func(
                features, adj_lists,
                torch.LongTensor(unique_nodes_list)
            )
        to_feats = mask.mm(embed_matrix)

        return to_feats
