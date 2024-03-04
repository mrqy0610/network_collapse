import networkx as nx
import numpy as np

from src.lib.graph import Graph
from src.mvc_env import MvcEnv

GAMMA = 1  # decay rate of past observations
UPDATE_TIME = 1000
EMBEDDING_SIZE = 64
MAX_ITERATION = 1000000
LEARNING_RATE = 0.0001  # dai
MEMORY_SIZE = 500000
Alpha = 0.0001  ## weight of reconstruction loss
########################### hyperparameters for priority(start)#########################################
epsilon = 0.0000001  # small amount to avoid zero priority
alpha = 0.6  # [0~1] convert the importance of TD error to priority
beta = 0.4  # importance-sampling, from initial value increasing to 1
beta_increment_per_sampling = 0.001
TD_err_upper = 1.  # clipped abs error
########################## hyperparameters for priority(end)#########################################
N_STEP = 5
NUM_MIN = 30
NUM_MAX = 50
REG_HIDDEN = 32
BATCH_SIZE = 64
initialization_stddev = 0.01  # 权重初始化的方差
n_valid = 200
aux_dim = 4
num_env = 1
inf = 2147483647 / 2
#########################  embedding method ##########################################################
max_bp_iter = 3
aggregatorID = 0  # 0:sum; 1:mean; 2:GCN
embeddingMethod = 1


def gen_graph(num_min, num_max, g_type):
    cur_n = np.random.randint(num_max - num_min + 1) + num_min
    if g_type == 'erdos_renyi':
        g_graph = nx.erdos_renyi_graph(n=cur_n, p=0.15)
    elif g_type == 'powerlaw':
        g_graph = nx.powerlaw_cluster_graph(n=cur_n, m=4, p=0.05)
    elif g_type == 'small-world':
        g_graph = nx.connected_watts_strogatz_graph(n=cur_n, k=8, p=0.1)
    elif g_type == 'barabasi_albert':
        g_graph = nx.barabasi_albert_graph(n=cur_n, m=4)
    return g_graph


def gen_network(graph):
    edges = graph.edges()
    if len(edges) > 0:
        a, b = zip(*edges)
        A = np.array(a)
        B = np.array(b)
    else:
        A = np.array([0])
        B = np.array([0])
    return Graph(len(graph.nodes()), len(edges), A, B)


if __name__ == "__main__":
    g = gen_network(gen_graph(30, 50, 'barabasi_albert'))
    env = MvcEnv(50)
    env.s0(g)
    print("is_terminal: ", env.is_terminal())
    print("max_connected_nodes_num:", env.get_max_connected_nodes_num())
    print("reward: ", env.get_reward())
    env.step(1)
    print("is_terminal: ", env.is_terminal())
    print("max_connected_nodes_num:", env.get_max_connected_nodes_num())
    print("reward: ", env.get_reward())
    env.step(5)
    print("is_terminal: ", env.is_terminal())
    print("max_connected_nodes_num:", env.get_max_connected_nodes_num())
    print("reward: ", env.get_reward())
