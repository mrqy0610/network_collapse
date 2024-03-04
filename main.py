import os
import torch
import argparse
import networkx as nx
import matplotlib.pyplot as plt

from algorithm.greedy import GreedyAgent
from algorithm.dqn import DQNAgent
from algorithm.a2c import A2CAgent
from algorithm.mcts import MCTSAgent
from algorithm.mcts_multi import MCTSMultiAgent
from src.my_mvc_env import MvcEnv
from utils.utils import set_seed, get_config_from_env

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
parser = argparse.ArgumentParser(description='onpolicy', formatter_class=argparse.RawDescriptionHelpFormatter)
############################# algorithm config
parser.add_argument("--algorithm_name", type=str, default='mcts-multi', choices=["greedy", "dqn", "hie-policy", "mcts", "mcts-multi"])
parser.add_argument("--connectivity_method", type=str, default='pairwise', choices=["pairwise", "gcc"])
parser.add_argument("--lower_agent_type", type=str, default='degree_greedy', choices=["degree_greedy", ])

############################# env config
parser.add_argument("--fix_graph", action='store_true', default=False)
parser.add_argument("--fix_graph_type", action='store_true', default=False)
parser.add_argument("--graph_type", type=str, default='barabasi_albert',
                    choices=["erdos_renyi", "powerlaw", "small-world", "barabasi_albert"])
parser.add_argument("--multi_env", action='store_true', default=False)
parser.add_argument("--min_nodes", type=int, default=10)
parser.add_argument("--max_nodes", type=int, default=50)
parser.add_argument("--train_with_preload_graph", action='store_true', default=False)
parser.add_argument("--train_graph_path", type=str, default='./data/data_set_multi_node12/')
parser.add_argument("--preload_graph_node_num", type=int, default=50)
parser.add_argument("--diff_nodes_weights", action='store_true', default=False)
parser.add_argument("--use_weights_features", action='store_true', default=False)
parser.add_argument("--eliminate_action", action='store_true', default=False)
parser.add_argument("--early_stop_env", action='store_true', default=False)
parser.add_argument("--render_graph", action='store_true', default=False)

############################# process trick
parser.add_argument("--reward_reflect", action='store_true', default=False)
parser.add_argument("--reward_reflect_neg", action='store_true', default=False)
parser.add_argument("--reward_scaling", action='store_true', default=False)
parser.add_argument("--reward_normalization", action='store_true', default=False)
parser.add_argument("--critic_use_huber_loss", action='store_true', default=False)
parser.add_argument("--huber_loss_delta", type=float, default=10.0)
parser.add_argument("--eliminate_orphan_node", action='store_true', default=False)

############################# model config
parser.add_argument("--graphsage_inner_dim", type=int, default=128)
parser.add_argument("--graphsage_output_dim", type=int, default=128)
parser.add_argument("--graphsage_adj_num_samples", type=int, default=10)
parser.add_argument("--action_embed_dim", type=int, default=128)
parser.add_argument("--qnetwork_inner_dim", type=int, default=128)

############################# hie-policy config
parser.add_argument("--attention_embedding_dim", type=int, default=128)
parser.add_argument("--n_encode_layers", type=int, default=2)
parser.add_argument("--max_upper_actions_len", type=int, default=114514)
parser.add_argument("--random_stop_actions", action='store_true', default=False)
parser.add_argument("--random_stop_eps", type=float, default=0.2)
parser.add_argument("--stop_eps_decay", action='store_true', default=False)
parser.add_argument("--stop_eps_decay_rate", type=float, default=0.01)
parser.add_argument("--stop_eps_decay_freq", type=float, default=100)
parser.add_argument("--stop_eps_decay_min", type=float, default=0.01)

############################# mcts config
parser.add_argument("--total_training_steps", type=int, default=20000)
parser.add_argument('--gcn_hidden_size', default=64, type=int, help='Number of features for each node')
parser.add_argument('--gcn_dropout', default=0.1, type=float, help='Drop out rate for gcn layers')
parser.add_argument("--mcts_network_inner_dim", type=int, default=64)
parser.add_argument("--mcts_lr_multiplier", type=float, default=1.0)
parser.add_argument("--mcts_temperature", type=float, default=1.0)
parser.add_argument("--mcts_n_playout", type=int, default=400)
parser.add_argument("--mcts_c_puct", type=float, default=5.0)
parser.add_argument("--mcts_update_epochs", type=int, default=5)
parser.add_argument("--mcts_kl_targ", type=float, default=0.01)
parser.add_argument("--mcts_discount_tree_value", action='store_true', default=True)
parser.add_argument("--mcts_l2_const", type=float, default=1e-4)
parser.add_argument("--mcts_node_value_normal", action='store_true', default=False)
parser.add_argument("--mcts_dynamic_c_puct", action='store_true', default=False)
parser.add_argument("--mcts_c_puct_base", type=float, default=19652)
parser.add_argument("--mcts_c_puct_init", type=float, default=1.25)
parser.add_argument("--mcts_ucb_add_reward", action='store_true', default=False)

############################# mcts-multi config
parser.add_argument("--mcts_num_actors", type=int, default=1)
parser.add_argument("--mcts_para_update_freq", type=int, default=20)

############################# hyper parameters config
parser.add_argument("--learning_rate", type=float, default=5e-3)
parser.add_argument("--total_episodes", type=int, default=10000)
parser.add_argument("--memory_capacity", type=int, default=200)
parser.add_argument("--model_update_freq", type=int, default=50)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--epsilon_greedy_rate", type=float, default=0.9)
parser.add_argument("--target_q_update_freq", type=int, default=100)
parser.add_argument("--reward_gamma", type=float, default=0.95)
parser.add_argument("--actor_learning_rate", type=float, default=5e-3)
parser.add_argument("--critic_learning_rate", type=float, default=5e-3)
parser.add_argument("--add_graph_reconstruction_loss", action='store_true', default=False)
parser.add_argument("--graph_reconstruction_loss_alpha", type=float, default=0.001)

############################# eval config
parser.add_argument("--eval_mode", action='store_true', default=False)
parser.add_argument("--eval_episodes", type=int, default=1)
parser.add_argument("--eval_with_preload_graph", action='store_true', default=False)
parser.add_argument("--eval_graph_path", type=str, default='./data/graph_250.graphml')
parser.add_argument("--add_eval_stage", action='store_true', default=False)
parser.add_argument("--eval_freq_in_train", type=int, default=10)
parser.add_argument("--eval_with_collapse_graph", action='store_true', default=False)

############################# res record config
parser.add_argument("--save_model", action='store_true', default=False)
parser.add_argument("--output_res", action='store_true', default=False)
parser.add_argument("--load_model", action='store_true', default=False)
parser.add_argument("--save_model_freq", type=int, default=100)
parser.add_argument("--output_res_dir", type=str, default='./output_res/')
parser.add_argument("--load_model_path", type=str, default='./output_res/model/model_492.pth')

if __name__ == "__main__":
    # set program config
    config = parser.parse_args()
    #################### train config for mcts train
    config.algorithm_name = 'greedy'
    config.eval_with_collapse_graph = True
    config.eval_mode = False
    config.preload_graph_node_num = 50
    config.add_eval_stage = True
    config.eval_freq_in_train = 1
    config.train_with_preload_graph = True
    config.train_graph_path = './data/data_set_single_node50_2/'
    config.reward_reflect = True
    config.mcts_node_value_normal = True
    config.mcts_dynamic_c_puct = True
    config.mcts_ucb_add_reward = True
    #################### test mcts load model
    config.load_model = True
    config.load_model_path = './res/mcts/cmp_mcts_multi/graph2/mcts_multi_combineTrick_singleGraphNode50_2_capacity2k_notTrainDelay_fix/model/model_13114.pth'
    #################### test mcts load model
    config.output_res = True
    config.output_res_dir = './res/mcts_multi/data_set_single_node50_2_greedy/'
    #################### test for output res and save mode
    # config.output_res = True
    # config.save_model = True
    # config.output_res_dir = './res/local_test/'
    #################### train config for greedy
    # config.algorithm_name = 'greedy'
    # config.train_with_preload_graph = True
    # config.train_graph_path = './data/data_set_single_node50_1'
    # config.output_res = True
    # config.output_res_dir = './res/greedy/cmp_graph_node50/single_graph_1/'
    # config.eval_episodes = 15
    #################### test config with graph
    # config.train_with_preload_graph = True
    # config.output_res = True
    # config.output_res_dir = './res/hie-policy/cmp_lr/greedy/'
    # config.load_model = True
    # config.load_model_path = './res/hie-policy/cmp_graph_node50/single_graph_2/model/model_3130.pth'
    # config.train_graph_path = './data/crime/'
    # # config.render_graph = True
    #################### train config for hie-policy train
    # config.algorithm_name = 'hie-policy'
    # config.train_with_preload_graph = True
    # config.reward_normalization = True
    # config.critic_use_huber_loss = True
    # config.eliminate_orphan_node = True
    # config.add_graph_reconstruction_loss = True
    # config.add_eval_stage = True
    # config.eval_freq_in_train = 1
    # # config.train_graph_path = './data/data_set_single_node20_2/'
    #################### train config for hie-policy eval
    # config.eval_mode = True
    # config.load_model = True
    # config.render_graph = True
    # config.eval_graph_path = './data/Crime.graphml'
    # config.load_model_path = './res/a2c_test_multiGraph_sameWeights_reflectRewardNew_omitOrphan_useDegFeature_noLimit_epsInit0.1_testEval/model/model_23.pth'
    # config.load_model_path = './res/a2c_test_multiGraph_sameWeights_reflectRewardNew_omitOrphan_useDegFeature_noLimit_eps0.1Stop/model/model_9133.pth'

    # set seed for training
    set_seed(123)
    # set device for training
    config.cuda = torch.cuda.is_available()
    config.device = 'cpu' if not torch.cuda.is_available() else 'cuda'
    # sey env for training
    env = MvcEnv(config=config)
    # set config from env
    get_config_from_env(config, env)
    # set agent
    if config.algorithm_name == 'hie-policy':
        config.early_stop_env = True
        config.eliminate_orphan_node = True
        agent = A2CAgent(config=config, env=env)
    elif config.algorithm_name == 'dqn':
        agent = DQNAgent(config=config, env=env)
    elif config.algorithm_name == 'mcts':
        agent = MCTSAgent(config=config, env=env)
    elif config.algorithm_name == 'mcts-multi':
        agent = MCTSMultiAgent(config=config, env=env)
    elif config.algorithm_name == 'greedy':
        agent = GreedyAgent(config=config, env=env)
    else:
        raise NotImplementedError()
    if config.eval_mode:
        agent.eval()
    elif config.eval_with_collapse_graph:
        agent.eval_graph()
    else:
        agent.train()

