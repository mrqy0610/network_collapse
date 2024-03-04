import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable

import time
import random
import numpy as np
from collections import defaultdict
from sklearn.metrics import f1_score

from model.graphsage.encoders import Encoder
from model.graphsage.aggregators import MeanAggregator, MeanAggregatorHead

"""
Simple supervised GraphSAGE model as well as examples running the model
on the Cora and Pubmed datasets.
"""


class GraphAggregator(nn.Module):
    def __init__(self, feature_dim, embed_dim, num_sample=10, cuda=False):
        super(GraphAggregator, self).__init__()

        self.feat_dim = feature_dim
        self.num_sample = num_sample
        self.embed_dim = embed_dim
        self.cuda = cuda
        # set aggregator and encoder
        # lambda nodes: self.features[nodes]
        self.agg1 = MeanAggregatorHead(
            cuda=self.cuda,  gcn=True
        )
        self.enc1 = Encoder(
            feature_dim=feature_dim,
            embed_dim=self.embed_dim, aggregator=self.agg1,
            num_sample=self.num_sample,
            gcn=True, cuda=self.cuda
        )
        # lambda nodes: self.enc1(features, adj_lists, nodes).t()
        self.agg2 = MeanAggregator(
            cuda=self.cuda,  gcn=True
        )
        self.enc2 = Encoder(
            feature_dim=self.enc1.embed_dim,
            embed_dim=self.embed_dim, aggregator=self.agg2,
            num_sample=self.num_sample,
            base_model=self.enc1,
            gcn=True, cuda=self.cuda
        )

    def forward(self, features, adj_lists, nodes):

        return self.enc2(features, adj_lists, nodes)


class GraphSage(nn.Module):

    def __init__(self, config, feature_num):
        super(GraphSage, self).__init__()
        self.config = config
        self.feature_num = feature_num
        # set aggregator and encoder
        self.enc = GraphAggregator(
            feature_dim=feature_num, embed_dim=config.graphsage_inner_dim,
            num_sample=config.graphsage_adj_num_samples, cuda=config.cuda,
        )
        # self.xent = nn.CrossEntropyLoss()
        self.weight = nn.Parameter(torch.FloatTensor(config.graphsage_output_dim, self.enc.embed_dim))
        self.empty_embedding = nn.Parameter(torch.tensor(np.zeros((1, config.graphsage_output_dim)), dtype=torch.float32), requires_grad=True)
        init.xavier_uniform_(self.weight)

    def forward(self, features, adj_lists, nodes):
        """
        :param nodes: torch.Size([nodes_num)
        :return: torch.Size([nodes_num, embed_dim])
        """
        # return empty_embedding if meet empty graph
        if nodes.shape[0] == 0:
            return self.empty_embedding
        embeds = self.enc(features, adj_lists, nodes)
        scores = self.weight.mm(embeds)

        return scores.t()
    # def loss(self, features, adj_lists, nodes, labels):
    #     scores = self.forward(features, adj_lists, nodes)
    #     return self.xent(scores, labels.squeeze())

    # def update_graph_data(self, features, adj_lists):
    #     # update now feature and adj_lists
    #     self.features = features
    #     self.enc1.features = features
    #     self.adj_lists = adj_lists
    #     self.enc1.adj_lists = adj_lists
    #     self.enc2.adj_lists = adj_lists
