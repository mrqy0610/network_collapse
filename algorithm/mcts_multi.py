import os
import gym
import ray
import copy
import math
import time
import random
import numpy as np
import networkx as nx
import multiprocessing

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from tensorboardX import SummaryWriter
from utils.replay_buffer import ShareBuffer
from model.mcts_model.policy_value_net import PolicyValueNet
from model.attention_model.hierarchy_actor_critic import Hierarchy_Actor, Critic
from utils.utils import eliminate_orphan_node, load_graph, load_all_graphs, softmax, process_step_reward, choose_graph_from_list, compute_reward_scale


class MinMaxStats:
    """A class that holds the min-max values of the tree."""

    def __init__(self, min_value_bound=None, max_value_bound=None):
        self.maximum = min_value_bound if min_value_bound else -float('inf')
        self.minimum = max_value_bound if max_value_bound else float('inf')

    def update(self, value: float):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value: float) -> float:
        if self.maximum > self.minimum:
            # We normalize only when we have set the maximum and minimum values.
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value


class TreeNode(object):
    """A node in the MCTS tree.

    Each node keeps track of its own value Q, prior probability P, and
    its visit-count-adjusted prior score u.
    """

    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {}  # a map from action to TreeNode
        self._n_visits = 0
        self._V = 0
        self._U = 0
        self._P = prior_p
        self._R = 0

    def expand(self, action_priors, reward, done):
        """Expand tree by creating new children.
        action_priors: a list of tuples of actions and their prior probability
            according to the policy function.
        """
        # expnad node if not at termial state
        if not done:
            for action, prob in action_priors:
                if action not in self._children:
                    self._children[action] = TreeNode(self, prob)
        # record node reward either at terminal or not
        self._R = reward

    def select(self, c_puct):
        """Select action among children that gives maximum action value Q
        plus bonus u(P).
        Return: A tuple of (action, next_node)
        """
        return max(self._children.items(),
                   key=lambda act_node: act_node[1].get_value(c_puct))

    def update(self, leaf_value):
        """Update node values from leaf evaluation.
        leaf_value: the value of subtree evaluation from the current player's
            perspective.
        """
        # Count visit.
        self._n_visits += 1
        # Update Q, a running average of values for all visits.
        self._V += 1.0 * (leaf_value - self._V) / self._n_visits

    def update_recursive(self, leaf_value):
        """Like a call to update(), but applied recursively for all ancestors.
        """
        # If it is not root, this node's parent should be updated first.
        if self._parent:
            self._parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def update_dense_recursive(self, now_value, gamma, min_max_stats):
        """Like a call to update(), but applied recursively for all ancestors.
        """
        # If it is not root, this node's parent should be updated first.
        if self._parent:
            self._parent.update_dense_recursive(self._R + gamma * now_value, gamma, min_max_stats)
        self.update(now_value)
        min_max_stats.update(self._V)

    def get_value(self, c_puct):
        """Calculate and return the value for this node.
        It is a combination of leaf evaluations Q, and this node's prior
        adjusted for its visit count, u.
        c_puct: a number in (0, inf) controlling the relative impact of
            value Q, and prior probability P, on this node's score.
        """
        self._U = (c_puct * self._P *
                   np.sqrt(self._parent._n_visits) / (1 + self._n_visits))

        return self._V + self._U

    def is_leaf(self):
        """Check if leaf node (i.e. no nodes below this have been expanded)."""
        return self._children == {}

    def is_root(self):
        return self._parent is None

    def add_exploration_noise(self, dirichlet_alpha, exploration_fraction):
        actions = list(self._children.keys())
        noise = np.random.dirichlet([dirichlet_alpha] * len(actions))
        frac = exploration_fraction
        for a, n in zip(actions, noise):
            self._children[a]._P = self._children[a]._P * (1 - frac) + n * frac


class MCTS(object):
    """An implementation of Monte Carlo Tree Search."""

    def __init__(self, config, policy_value_fn, c_puct=5, n_playout=10000, gamma=0.95, discount=True):
        """
        policy_value_fn: a function that takes in a board state and outputs
            a list of (action, probability) tuples and also a score in [-1, 1]
            (i.e. the expected value of the end game score from the current
            player's perspective) for the current player.
        c_puct: a number in (0, inf) that controls how quickly exploration
            converges to the maximum-value policy. A higher value means
            relying on the prior more.
        """
        self.config = config
        self._root = TreeNode(None, 1.0)
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout
        ################# add config for dense reward mcts
        self.gamma = gamma
        self.discount = discount

    def _playout(self, state, reward, done, env, min_max_stats):
        """Run a single playout from the root to the leaf, getting a value at
        the leaf and propagating it back through its parents.
        State is modified in-place, so a copy must be provided.
        """
        node = self._root
        ################ test #################
        playout_action = []
        ################ test #################
        while True:
            if node.is_leaf():
                ################ test #################
                # print('playout_action', playout_action)
                ################ test #################
                break
            # Greedily select next move.
            # action, node = node.select(self._c_puct)
            action, node = self.my_select_child(node, min_max_stats)
            ################ test #################
            playout_action.append(action)
            ################ test #################
            state, reward, done, info = env.step(action)
        # Evaluate the leaf using a network which outputs a list of
        # (action, probability) tuples p and also a score v
        # for the current player.
        action_probs, leaf_value = self._policy(state)
        # change last node value to value predict if not reach the leaf node
        leaf_value = 0.0 if done else leaf_value
        # and then expand child tree
        node.expand(action_priors=action_probs, reward=process_step_reward(
            reward, state['origin_num_nodes'], reward_reflect=True, reward_normalization=False,
            multi_step=False, reward_reflect_neg=self.config.reward_reflect_neg,
            reward_scaling=self.config.reward_scaling, reward_scale=self.config.reward_scale), done=done)
        # Update value and visit count of nodes in this traversal.
        node.update_dense_recursive(leaf_value, self.gamma, min_max_stats)

    def get_move_probs(self, state, reward, done, env, temp=1e-3):
        """Run all playouts sequentially and return the available actions and
        their corresponding probabilities.
        state: the current game state
        temp: temperature parameter in (0, 1] controls the level of exploration
        """
        # instantiate MinMaxStats every simulation for a move
        min_max_stats = MinMaxStats()
        for n in range(self._n_playout):
            env_copy = copy.deepcopy(env)
            # print('play_time', n)
            self._playout(state, reward, done, env_copy, min_max_stats)
        # calc the move probabilities based on visit counts at the root node
        act_visits = [(act, node._n_visits) for act, node in self._root._children.items()]
        acts, visits = zip(*act_visits)
        act_probs = softmax(1.0 / temp * np.log(np.array(visits) + 1e-10))
        # print('visits', visits)
        # print('acts', acts)
        # act_probs = [visit_i ** (1 / temp) for visit_i in visits]
        # total_count = sum(act_probs)
        # act_probs = [x / total_count for x in act_probs]

        return acts, act_probs

    def update_with_move(self, last_move):
        """Step forward in the tree, keeping everything we already know
        about the subtree.
        """
        #################### move to next child node
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        #################### reset tree and get a new root
        else:
            self._root = TreeNode(None, 1.0)

    def my_select_child(self, node, min_max_stats):
        def my_get_ucb(sub_node):
            ############## calculate value
            if sub_node._n_visits > 0:
                sub_node._Q = sub_node._R * self.config.mcts_ucb_add_reward + self.gamma * sub_node._V \
                    if not self.config.mcts_node_value_normal else \
                    sub_node._R * self.config.mcts_ucb_add_reward + self.gamma * min_max_stats.normalize(sub_node._V)
            else:
                sub_node._Q = 0.0
            ############## calculate visit
            _c_puct = math.log(
                (sub_node._parent._n_visits + self.config.mcts_c_puct_base + 1) / self.config.mcts_c_puct_base
            ) + self.config.mcts_c_puct_init
            _c_puct *= math.sqrt(sub_node._parent._n_visits) / (sub_node._n_visits + 1)
            # _c_puct = _c_puct * sub_node._P if self.config.mcts_dynamic_c_puct else self._c_puct
            _c_puct = _c_puct if self.config.mcts_dynamic_c_puct else self._c_puct
            sub_node._U = (_c_puct * sub_node._P *
                           np.sqrt(sub_node._parent._n_visits) / (1 + sub_node._n_visits))
            return sub_node._Q + sub_node._U

        return max(node._children.items(), key=lambda act_node: my_get_ucb(act_node[1]))

    def expand_root(self, action_probs, state, reward, done):
        # if not add visit count, will always choose first node
        self._root._n_visits += 1
        self._root.expand(action_priors=action_probs, reward=process_step_reward(
            reward, state['origin_num_nodes'], reward_reflect=True, reward_normalization=False,
            multi_step=False, reward_reflect_neg=self.config.reward_reflect_neg,
            reward_scaling=self.config.reward_scaling, reward_scale=self.config.reward_scale), done=done)

    def root_add_exploration_noise(self, dirichlet_alpha, exploration_fraction):
        self._root.add_exploration_noise(dirichlet_alpha, exploration_fraction)

    def __str__(self):
        return "MCTS"


##################### method for multi process
@ray.remote
class DataWorker(object):
    def __init__(self, rank, config, shared_storage, replay_buffer, env, graphs):
        self.rank = rank
        self.config = config
        self.shared_storage = shared_storage
        self.replay_buffer = replay_buffer
        # add env for network collapse
        self.env = env
        self.graphs = graphs
        self.model = PolicyValueNet(config=self.config)
        # config mcts process
        self.mcts = MCTS(
            config, self.model.policy_value_fn, c_puct=config.mcts_c_puct,
            n_playout=config.mcts_n_playout, gamma=config.reward_gamma,
            discount=config.mcts_discount_tree_value
        )

    ##################### method for play game
    def get_action_old(self, state, reward, done, test_mode=False):
        available_nodes = state['available_nodes']
        # the pi vector returned by MCTS as in the alphaGo Zero paper
        move_probs = np.zeros(self.config.preload_graph_node_num)
        if len(available_nodes) > 0:
            acts, probs = self.mcts.get_move_probs(
                state, reward, done, env=self.env,
                temp=self.config.mcts_temperature if not test_mode else 1e-3,
            )  # choose temp 1e-3 for test mode to get argmax action
            move_probs[list(acts)] = probs
            if not test_mode:
                # add Dirichlet Noise for exploration (needed for training stage)
                move = np.random.choice(
                    acts,
                    p=0.75 * probs + 0.25 * np.random.dirichlet(0.3 * np.ones(len(probs)))
                )
                # update the root node and reuse the search tree
                self.mcts.update_with_move(move)
                return move, move_probs
            else:
                # with the default temp=1e-3, it is almost equivalent
                # to choosing the move with the highest prob
                move = np.random.choice(acts, p=probs)
                # reset the root node
                self.mcts.update_with_move(-1)
                return move
        else:
            print("WARNING: not available nodes to choose")

    def get_action(self, state, reward, done):
        available_nodes = state['available_nodes']
        # prob for actions not avaiable is zero
        move_probs = np.zeros(self.config.preload_graph_node_num)
        if len(available_nodes) > 0:
            ###################### reset root for a new tree expand
            self.mcts.update_with_move(-1)
            ###################### use network to get root pior
            action_probs, _ = self.model.policy_value_fn(state)
            # print('action_probs', [prob for _, prob in action_probs])
            ###################### expand root previous
            self.mcts.expand_root(action_probs, state, reward, done)
            ###################### add Dirichlet Noise for exploration (only for root)
            self.mcts.root_add_exploration_noise(dirichlet_alpha=0.3, exploration_fraction=0.25)
            ###################### expand tree from now root (get all actions and probs)
            acts, probs = self.mcts.get_move_probs(
                state, reward, done, env=self.env,
                temp=self.config.mcts_temperature,
            )  # choose temp 1e-3 for test mode to get argmax action
            ###################### update move prob
            move_probs[list(acts)] = probs
            ###################### select an action to move
            move = np.random.choice(acts, p=probs)
            # self.mcts.update_with_move(move)
            return move, move_probs
        else:
            print("WARNING: not available nodes to choose")

    def reset_player(self):
        self.mcts.update_with_move(-1)
    ##################### method for play game

    def run(self):
        with torch.no_grad():
            while ray.get(self.shared_storage.get_counter.remote()) < self.config.total_training_steps:
                ###################### update model para from share space
                self.model.set_weights(ray.get(self.shared_storage.get_weights.remote()))
                self.model.eval()
                ###################### init env
                state, reward, done = self.env.reset(choose_graph_from_list(self.graphs)), 0.0, False
                all_states, actions_probs, all_rewards = [], [], []
                train_episode_reward = 0.0
                while True:
                    ###################### tree search to get an action
                    action, action_probs = self.get_action(state, reward, done)
                    ###################### store step obs, action
                    all_states.append(state)
                    actions_probs.append(action_probs)
                    ###################### perform remove action
                    state, reward, done, info = self.env.step(action)
                    ###################### store step reward
                    all_rewards.append(process_step_reward(
                        reward, state['origin_num_nodes'], reward_reflect=True, reward_normalization=False,
                        multi_step=False, reward_reflect_neg=self.config.reward_reflect_neg,
                        reward_scaling=self.config.reward_scaling, reward_scale=self.config.reward_scale))
                    train_episode_reward += reward
                    ###################### reach end of an episode
                    if done:
                        ###################### reset MCTS root node
                        self.reset_player()
                        ###################### calculate discount reward
                        all_discount_rewards, now_discount_value = [], 0.0
                        for node_r in reversed(all_rewards):
                            now_discount_value *= self.config.reward_gamma if self.config.mcts_discount_tree_value else 1.0
                            now_discount_value += node_r
                            all_discount_rewards.append(now_discount_value)
                        all_discount_rewards.reverse()
                        # print('process', self.rank, 'all_discount_rewards', all_discount_rewards)
                        print('process', self.rank, 'train_episode_reward', train_episode_reward)
                        break
                ###################### save episode data to share buffer
                self.replay_buffer.insert_all_data.remote(all_states, actions_probs, all_discount_rewards)
                ###################### save train_episode_reward to share space
                self.shared_storage.add_sampler_logs.remote(train_episode_reward)


##################### method for evaluation
def get_eval_action(config, model, env, state, reward, done):
    # config mcts process
    mcts = MCTS(
        config, model.policy_value_fn, c_puct=config.mcts_c_puct,
        n_playout=config.mcts_n_playout, gamma=config.reward_gamma,
        discount=config.mcts_discount_tree_value
    )
    available_nodes = state['available_nodes']
    # prob for actions not avaiable is zero
    move_probs = np.zeros(config.preload_graph_node_num)
    if len(available_nodes) > 0:
        ###################### reset root for a new tree expand
        mcts.update_with_move(-1)
        ###################### use network to get root pior
        action_probs, _ = model.policy_value_fn(state)
        ###################### expand root previous
        mcts.expand_root(action_probs, state, reward, done)
        ###################### expnad tree from now root (get all actions and probs)
        acts, probs = mcts.get_move_probs(
            state, reward, done, env=env, temp=1e-3,
        )  # choose temp 1e-3 for test mode to get argmax action
        ###################### update move prob
        move_probs[list(acts)] = probs
        ###################### select an action to move
        # with the default temp=1e-3, it is almost equivalent
        # to choosing the move with the highest prob
        move = np.random.choice(acts, p=probs)
        # reset the root node
        # mcts.update_with_move(-1)
        return move
    else:
        print("WARNING: not available nodes to choose")


def play_game(config, model, env, graphs, ep_i, ep_data):
    """start a game for eval"""
    state, reward, done = env.reset(choose_graph_from_list(graphs)), 0.0, False
    episode_reward = 0.
    while True:
        action = get_eval_action(
            config=config, model=model, env=env, state=state,
            reward=reward, done=done,
        )
        # perform a remove action
        state, reward, done, info = env.step(action)
        episode_reward = episode_reward + reward
        if done:
            break
    ep_data[ep_i] = episode_reward


def play_game_eval(config, model, env, graphs):
    """start a game for eval"""
    state, reward, done = env.reset(choose_graph_from_list(graphs)), 0.0, False
    episode_reward = 0.
    all_figures = []
    while True:
        all_figures.append(state['connectivity'])
        action = get_eval_action(
            config=config, model=model, env=env, state=state,
            reward=reward, done=done,
        )
        print('action', action)
        # perform a remove action
        state, reward, done, info = env.step(action)
        episode_reward = episode_reward + reward
        if done:
            break
    return all_figures, episode_reward


@ray.remote
class EvalWorker(object):
    def __init__(self, config, shared_storage, env, graphs, model_dir):
        self.config = config
        self.shared_storage = shared_storage
        self.model = PolicyValueNet(config=self.config)
        self.env = env
        self.graphs = graphs
        # info for model saving
        self.best_test_score = float('-inf')
        self.model_step_counter = None
        self.model_dir = model_dir

    def run(self):
        with torch.no_grad():
            while ray.get(self.shared_storage.get_counter.remote()) < self.config.total_training_steps:
                self.model_step_counter = ray.get(self.shared_storage.get_counter.remote())
                self.model.set_weights(ray.get(self.shared_storage.get_weights.remote()))
                self.model.eval()
                manager = multiprocessing.Manager()
                ep_data = manager.dict()
                jobs = []
                for ep_i in range(self.config.eval_episodes):
                    p = multiprocessing.Process(target=play_game, args=(
                        self.config, self.model, copy.deepcopy(self.env), self.graphs, ep_i, ep_data,
                    ))
                    jobs.append(p)
                    p.start()
                for proc in jobs:
                    proc.join()
                eval_score = sum(ep_data.values()) / self.config.eval_episodes
                print('eval_score', eval_score)
                ####################### update max score and save model
                if eval_score >= self.best_test_score and self.config.save_model:
                    self.best_test_score = eval_score
                    torch.save({
                        'policy_value_net': self.model.state_dict(),
                    }, self.model_dir + '/model_' + str(self.model_step_counter) + '.pth')
                ####################### upload eval score for logging
                self.shared_storage.add_eval_log.remote(eval_score, self.model_step_counter)
                time.sleep(100)

    def eval(self):
        print('--------------eval worker start--------------')
        with torch.no_grad():
            self.model.set_weights(ray.get(self.shared_storage.get_weights.remote()))
            self.model.eval()
            manager = multiprocessing.Manager()
            ep_data = manager.dict()
            jobs = []
            for ep_i in range(self.config.eval_episodes):
                p = multiprocessing.Process(target=play_game, args=(
                    self.config, self.model, copy.deepcopy(self.env), self.graphs, ep_i, ep_data,
                ))
                jobs.append(p)
                p.start()
            for proc in jobs:
                proc.join()
            eval_score = sum(ep_data.values()) / self.config.eval_episodes
            print('eval_score', eval_score)
            time.sleep(100)


class EvalSingleWorker(object):
    def __init__(self, config, env, graphs, model_load_para):
        self.config = config
        self.model = PolicyValueNet(config=self.config)
        self.env = env
        self.graphs = graphs
        self.model.set_weights(model_load_para)
        print('-------------- load model path --------------')

    def eval(self):
        print('-------------- eval worker start -------------')
        with torch.no_grad():
            self.model.eval()
            all_figures, episode_reward = play_game_eval(
                self.config, self.model, copy.deepcopy(self.env), self.graphs,
            )
            print('all_figures', all_figures)
            print('episode_reward', episode_reward)
        return all_figures
##################### method for evaluation


@ray.remote
class SharedStorage(object):
    def __init__(self, config):
        # global update step counter
        self.step_counter = 0
        # global network (parameters)
        self.model = PolicyValueNet(config=config)
        # global info
        self.train_episode_rewards = []
        self.eval_episode_rewards = []

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, weights):
        return self.model.set_weights(weights)

    def incr_counter(self):
        self.step_counter += 1

    def get_counter(self):
        return self.step_counter

    def add_sampler_logs(self, eps_reward):
        self.train_episode_rewards.append(eps_reward)

    def get_sampler_logs(self):
        if len(self.train_episode_rewards) > 0:
            train_episode_rewards = sum(self.train_episode_rewards) / len(self.train_episode_rewards)
            self.train_episode_rewards = []
        else:
            train_episode_rewards = None

        return train_episode_rewards

    def add_eval_log(self, eval_score, eval_step_counter):
        self.eval_episode_rewards.append({
            'eval_score': eval_score, 'step_counter': eval_step_counter,
        })

    def get_eval_logs(self):
        eval_episode_rewards = copy.deepcopy(self.eval_episode_rewards)
        self.eval_episode_rewards = []

        return eval_episode_rewards
##################### method for multi process


class MCTSMultiAgent():
    def __init__(self, config, env):
        self.config = config
        self.env = env
        # load graph from saved data
        # self.graph = load_graph(self.config.train_graph_path) if self.config.train_with_preload_graph else None
        self.graphs = load_all_graphs(self.config.train_graph_path) if self.config.train_with_preload_graph else None
        self.config.preload_graph_node_num = len(self.graphs[0].nodes()) \
            if self.config.train_with_preload_graph else self.config.preload_graph_node_num
        self.config.reward_scale = compute_reward_scale(self.config.preload_graph_node_num, self.config.reward_gamma)
        # config data buffer in training stage
        # self.data_buffer = None
        # config for record
        self.num_simulations = 1
        # config model / res dir
        self.save_model_dir = config.output_res_dir + '/model/'
        self.save_res_dir = config.output_res_dir + '/res/'
        if not os.path.exists(self.save_model_dir):
            os.makedirs(self.save_model_dir)
        if not os.path.exists(self.save_res_dir):
            os.makedirs(self.save_res_dir)
        if self.config.output_res:
            self.writter = SummaryWriter(self.save_res_dir)
        # config policy value network and optim
        self.policy_value_net = PolicyValueNet(config=config)
        self.optimizer = optim.Adam(self.policy_value_net.parameters(), weight_decay=config.mcts_l2_const)
        # config for lr decay
        self.lr_multiplier = config.mcts_lr_multiplier
        self.kl_targ = config.mcts_kl_targ

    def save_model(self, training_episode):
        torch.save({
            'policy_value_net': self.policy_value_net.state_dict(),
            'lr_multiplier': self.lr_multiplier,
        }, self.save_model_dir + '/model_' + str(training_episode) + '.pth')

    def load_model(self):
        save_model_dict = torch.load(self.config.load_model_path) if torch.cuda.is_available() \
            else torch.load(self.config.load_model_path, map_location=torch.device('cpu'))
        self.policy_value_net.load_state_dict(save_model_dict['policy_value_net'])
        # self.lr_multiplier = save_model_dict['lr_multiplier']
        # self.actor_optimizer.load_state_dict(save_model_dict['actor_optimizer'])
        # self.critic.load_state_dict(save_model_dict['critic'])
        # self.critic_optimizer.load_state_dict(save_model_dict['critic_optimizer'])

    def policy_update(self, replay_buffer):
        """update the policy-value net"""
        self.policy_value_net.train()
        batch_features, batch_adjacency_matrixs, batch_actions_probs, batch_rewards = ray.get(
            replay_buffer.sample.remote(self.config.batch_size))
        # print('----------------------------')
        # print('batch_features', batch_features.shape)
        # print('batch_adjacency_matrixs', batch_adjacency_matrixs.shape)
        old_probs, old_v = self.policy_value_net.policy_value(batch_features, batch_adjacency_matrixs)
        all_value_loss, all_policy_loss, all_entropy = [], [], []
        for i in range(self.config.mcts_update_epochs):
            value_loss, policy_loss, entropy = self.policy_value_net.train_step(
                batch_features=batch_features, batch_adjacency_matrixs=batch_adjacency_matrixs,
                batch_actions_probs=batch_actions_probs, batch_rewards=batch_rewards,
                optimizer=self.optimizer, lr=self.config.learning_rate * self.lr_multiplier)
            all_value_loss.append(value_loss)
            all_policy_loss.append(policy_loss)
            all_entropy.append(entropy)
            new_probs, new_v = self.policy_value_net.policy_value(batch_features, batch_adjacency_matrixs)
            kl = np.mean(np.sum(old_probs * (
                    np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)), axis=1))
            if kl > self.kl_targ * 4:  # early stopping if D_KL diverges badly
                break
        # adaptively adjust the learning rate
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5
        # explained_var_old = (1 - np.var(np.array(batch_rewards) - old_v.flatten()) /
        #                      np.var(np.array(batch_rewards)))
        # explained_var_new = (1 - np.var(np.array(batch_rewards) - new_v.flatten()) /
        #                      np.var(np.array(batch_rewards)))
        # print(("kl:{:.5f},"
        #        "lr_multiplier:{:.3f},"
        #        "value_loss:{},"
        #        "policy_loss:{},"
        #        "entropy:{},"
        #        "explained_var_old:{:.3f},"
        #        "explained_var_new:{:.3f}"
        #        ).format(kl, self.lr_multiplier, value_loss, policy_loss, entropy, explained_var_old, explained_var_new))

        return np.mean(all_value_loss), np.mean(all_policy_loss), np.mean(all_entropy)

    def train(self):
        ray.init()
        ###################### load model parameters
        if self.config.load_model:
            self.load_model()
        ###################### config share space
        storage = SharedStorage.remote(config=self.config)
        replay_buffer = ShareBuffer.remote(config=self.config)
        ###################### config data collect process (multi)
        workers = [DataWorker.remote(rank, self.config, storage, replay_buffer, copy.deepcopy(self.env), self.graphs)
                   for rank in range(0, self.config.mcts_num_actors)]
        for worker in workers:
            worker.run.remote()
            time.sleep(5)
        ###################### config eval process
        eval_worker = EvalWorker.remote(config=self.config, shared_storage=storage, env=copy.deepcopy(self.env),
                                        graphs=self.graphs, model_dir=self.save_model_dir)
        eval_worker.run.remote()
        workers += [eval_worker]
        ###################### wait for enough buffer data
        while ray.get(replay_buffer.get_buffer_size.remote()) < self.config.batch_size:
            time.sleep(10)
        ###################### run training pipeline
        for step_count in range(self.config.total_training_steps):
            ###################### update network
            all_value_loss, all_policy_loss, _ = self.policy_update(replay_buffer)
            ####################### log update & sample info
            log_dict = {
                'all_value_loss': all_value_loss, 'all_policy_loss': all_policy_loss,
            }
            train_episode_reward = ray.get(storage.get_sampler_logs.remote())
            if train_episode_reward is not None:
                log_dict['train_episode_reward'] = train_episode_reward
            self.output_res(log_dict, step_count)
            eval_res = ray.get(storage.get_eval_logs.remote())
            for res in eval_res:
                self.output_res({'eval_episode_reward': res['eval_score']}, res['step_counter'])
            ####################### update share space (step_counter, model_para)
            storage.incr_counter.remote()
            if step_count % self.config.mcts_para_update_freq == 0:
                print('update global para with value_loss', all_value_loss, 'policy_loss', all_policy_loss)
                storage.set_weights.remote(self.policy_value_net.get_weights())
            # time.sleep(10)
        ####################### last update share model_para
        storage.set_weights.remote(self.policy_value_net.get_weights())
        ####################### process sync
        ray.wait(workers, len(workers))
        ray.shutdown()

    def output_res(self, train_infos, total_num_steps):
        if not self.config.output_res:
            return
        print('train_infos', train_infos)
        for k, v in train_infos.items():
            self.writter.add_scalars(k, {k: v}, total_num_steps)

    def eval(self):
        ray.init()
        ###################### load model parameters
        if self.config.load_model:
            self.load_model()
        print('-------------------- eval_mode --------------------')
        ###################### config share space
        storage = SharedStorage.remote(config=self.config)
        ###################### config eval process
        eval_worker = EvalSingleWorker(config=self.config, env=copy.deepcopy(self.env),
                                        graphs=self.graphs, model_load_para=self.policy_value_net.state_dict())
        all_figures = eval_worker.eval()
        print("all_figures", all_figures)
        for (step, figure) in enumerate(all_figures):
            print(step, figure)
            self.output_res({'connectivity': figure, }, step)
