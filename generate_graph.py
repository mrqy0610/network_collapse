import os
import networkx as nx
from utils.utils import gen_graph, render_graph

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
"""
"erdos_renyi", "powerlaw", "small-world", "barabasi_albert"
"""


def generate_graph():
    path = './data/'
    graph = gen_graph(num_min=20, num_max=20, g_type='erdos_renyi')
    render_graph(graph, path=path + '/data_set_single_node20_2/graph_20_2.png')
    if not os.path.exists(path):
        os.mkdir(path)
    nx.write_graphml(graph, path + '/data_set_single_node20_2/graph_20_2.graphml')


def load_graph():
    path = './data/'
    graph = nx.read_graphml(path + '/crime/Crime.graphml')
    pos = render_graph(graph)
    render_graph(graph, pos, path=path + '/crime/Crime.png')


def transfer_graph_data():
    path = './data/'
    data_test = './data/Crime.txt'
    graph = nx.read_edgelist(data_test)
    graph = nx.convert_node_labels_to_integers(graph, first_label=0, ordering='default')
    # get adjacency lists
    adjacency_list = nx.to_dict_of_lists(graph)
    adjacency_list = {key: set(adjacency_list[key]) for key in adjacency_list}
    for key in adjacency_list:
        print(key, adjacency_list[key])
    nx.write_graphml(graph, path + '/Crime.graphml')


# generate_graph()
# generate_graph()
# generate_graph()
load_graph()
