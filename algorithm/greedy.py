import os
import copy
import functools
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.distributions import Categorical
from tensorboardX import SummaryWriter
from utils.replay_buffer import RolloutBuffer
from model.attention_model.attention_utils import huber_loss
from model.attention_model.hierarchy_actor_critic import Hierarchy_Actor, Critic
from utils.utils import eliminate_orphan_node, load_graph, load_all_graphs, softmax, process_step_reward, choose_graph_from_list


class GreedyAgent():
    def __init__(self, config, env):
        self.config = config
        self.env = env
        # load graph from saved data
        self.graphs = load_all_graphs(self.config.train_graph_path) if self.config.train_with_preload_graph else None
        self.save_res_dir = config.output_res_dir + '/res/'
        if self.config.output_res:
            if not os.path.exists(self.save_res_dir):
                os.makedirs(self.save_res_dir)
            print('self.save_res_dir', self.save_res_dir)
            self.writter = SummaryWriter(self.save_res_dir)

    def choose_action(self, obs):
        nodes, adj_list = obs['available_nodes'], obs['adjacency_list']
        if self.config.lower_agent_type == 'degree_greedy':
            all_sequence_actions = [{'key': n, 'degree': len(adj_list[n])} for n in nodes]
            all_sequence_actions = sorted(all_sequence_actions, key=functools.cmp_to_key(
                lambda a, b: 1 if a['degree'] <= b['degree'] else -1))
            # print('all_sequence_actions', all_sequence_actions)
            return all_sequence_actions[0]['key']
        else:
            raise NotImplementedError()

    def train(self):
        all_episode_rewards = []
        # load graph from saved data
        # graph = load_graph(self.config.eval_graph_path) if self.config.eval_with_preload_graph else None
        for episode in range(self.config.eval_episodes):
            for graph in self.graphs:
                episode_reward = 0
                obs = self.env.reset(copy.deepcopy(graph))
                # # remove orphan node from obs if necessary
                # if self.config.eliminate_orphan_node:
                #     eliminate_orphan_node(obs)
                while True:
                    action = self.choose_action(obs)
                    next_obs, reward, done, info = self.env.step(action)
                    # # remove orphan node from obs if necessary
                    # if self.config.eliminate_orphan_node:
                    #     eliminate_orphan_node(next_obs)
                    # print('reward', reward)
                    # print('action', len(action))
                    episode_reward += reward
                    if done:
                        all_episode_rewards.append(episode_reward)
                        # self.output_res({'episode_reward': episode_reward}, self.episode_counter)
                        break
                    obs = next_obs
                # output mean episode_reward
                self.output_res({
                    'eval_episode_reward': np.mean(episode_reward),
                }, episode * 1000)
                print('eval_episode_reward', np.mean(episode_reward))

    def output_res(self, train_infos, total_num_steps):
        if not self.config.output_res:
            return
        for k, v in train_infos.items():
            self.writter.add_scalars(k, {k: v}, total_num_steps)

    def eval_graph(self):
        all_episode_rewards = []
        # load graph from saved data
        # graph = load_graph(self.config.eval_graph_path) if self.config.eval_with_preload_graph else None
        graph, episode_reward, step = self.graphs[0], 0, 0
        obs = self.env.reset(copy.deepcopy(graph))
        # # remove orphan node from obs if necessary
        # if self.config.eliminate_orphan_node:
        #     eliminate_orphan_node(obs)
        while True:
            self.output_res({'connectivity': obs['connectivity'], }, step)
            step += 1
            action = self.choose_action(obs)
            next_obs, reward, done, info = self.env.step(action)
            # # remove orphan node from obs if necessary
            # if self.config.eliminate_orphan_node:
            #     eliminate_orphan_node(next_obs)
            # print('reward', reward)
            # print('action', len(action))
            episode_reward += reward
            if done:
                all_episode_rewards.append(episode_reward)
                # self.output_res({'episode_reward': episode_reward}, self.episode_counter)
                break
            obs = next_obs
        # # output mean episode_reward
        # self.output_res({
        #     'eval_episode_reward': np.mean(episode_reward),
        # }, episode * 10)
        print('eval_episode_reward', np.mean(episode_reward))
