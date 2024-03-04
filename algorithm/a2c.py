import os
import gym
import copy
import math
import random
import numpy as np
import networkx as nx

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.distributions import Categorical
from tensorboardX import SummaryWriter
from utils.replay_buffer import RolloutBuffer
from model.attention_model.attention_utils import huber_loss
from model.attention_model.hierarchy_actor_critic import Hierarchy_Actor, Critic
from utils.utils import eliminate_orphan_node, load_graph, load_all_graphs, softmax, process_step_reward, choose_graph_from_list


class A2CAgent():
    def __init__(self, config, env):
        self.config = config
        self.env = env
        # load graph from saved data
        self.graphs = load_all_graphs(self.config.train_graph_path) if self.config.train_with_preload_graph else None
        # data rollout buffer
        self.buffer = RolloutBuffer(config=config)
        # config model and optimizer
        self.actor = Hierarchy_Actor(config=config).to(config.device)
        self.critic = Critic(config=config).to(config.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), config.actor_learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), config.critic_learning_rate)
        self.step_counter = 0
        self.learn_counter = 0
        self.episode_counter = 0
        self.save_model_dir = config.output_res_dir + '/model/'
        self.save_res_dir = config.output_res_dir + '/res/'
        if not os.path.exists(self.save_model_dir):
            os.makedirs(self.save_model_dir)
        if not os.path.exists(self.save_res_dir):
            os.makedirs(self.save_res_dir)
        if self.config.output_res:
            self.writter = SummaryWriter(self.save_res_dir)

    def insert_step_data(self, obs, action, logprob, reward, next_obs, recons_loss, done):
        self.buffer.insert_step_data(obs, action, logprob, reward, next_obs, recons_loss, done)

    def choose_action(self, obs, decode_type='sampling'):
        features, adj_lists, nodes, adj_matrixs = obs['features'], obs['adjacency_list'], obs['nodes'], obs['adjacency_matrix']
        # epsilon greedy for choosing action
        actions, logprobs, recons_loss = self.actor.get_multi_actions(features, adj_lists, nodes, adj_matrixs, decode_type)
        # only teke into account single batch act
        actions = actions.squeeze(0).cpu().numpy()

        return actions, logprobs, recons_loss

    def save_model(self, training_episode):
        torch.save({
            'actor': self.actor.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
        }, self.save_model_dir + '/model_' + str(training_episode) + '.pth')

    def load_model(self):
        save_model_dict = torch.load(self.config.load_model_path) if torch.cuda.is_available() \
            else torch.load(self.config.load_model_path, map_location=torch.device('cpu'))
        self.actor.load_state_dict(save_model_dict['actor'])
        self.actor_optimizer.load_state_dict(save_model_dict['actor_optimizer'])
        self.critic.load_state_dict(save_model_dict['critic'])
        self.critic_optimizer.load_state_dict(save_model_dict['critic_optimizer'])

    def learn(self):
        # # learn 100 times then the target network update
        # if self.learn_counter % self.config.target_q_update_freq == 0:
        #     self.target_net.load_state_dict(self.eval_net.state_dict())
        # update learner_counter every update
        self.learn_counter += 1

        batch_features, batch_adj_lists, batch_nodes, batch_adj_matrixs, \
        batch_actions, batch_logprobs, batch_rewards, batch_gammas, batch_dones, batch_recons_loss, \
        batch_next_features, batch_next_adj_lists, batch_next_nodes, batch_next_adj_matrixs = self.buffer.get_all_data()
        V, critic_recons_loss = self.critic(batch_features, batch_adj_lists, batch_nodes, batch_adj_matrixs)
        gamma = batch_gammas if self.config.algorithm_name == 'hie-policy' else self.config.reward_gamma
        next_V, _ = self.critic(batch_next_features, batch_next_adj_lists, batch_next_nodes, batch_next_adj_matrixs)
        Q = batch_rewards + gamma * next_V.detach() * (1 - batch_dones)
        ################ Actor Update ####################
        actor_loss = -((Q - V.detach()) * batch_logprobs).mean()
        origin_actor_loss = actor_loss
        if self.config.add_graph_reconstruction_loss:
            actor_loss = actor_loss + self.config.graph_reconstruction_loss_alpha * batch_recons_loss.mean()
        # use loss to confirm balance
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        ################ Critic Update ###################
        # use huber to calculate critic loss if necessary
        critic_loss = F.mse_loss(Q, V) if not self.config.critic_use_huber_loss else \
            huber_loss(Q, V, self.config.huber_loss_delta).mean()
        origin_critic_loss = critic_loss
        if self.config.add_graph_reconstruction_loss:
            critic_loss = critic_loss + self.config.graph_reconstruction_loss_alpha * critic_recons_loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        # clear data buffer
        self.buffer.clear()
        # output res
        print('actor loss', origin_actor_loss.item())
        print('critic loss', origin_critic_loss.item())

        return actor_loss.item(), critic_loss.item()

    def train(self):
        self.episode_counter = 0
        self.step_counter = 0
        if self.config.load_model:
            self.load_model()

        for episode in range(self.config.total_episodes):
            episode_reward = 0
            obs = self.env.reset(choose_graph_from_list(self.graphs))
            # remove orphan node from obs if necessary
            if self.config.eliminate_orphan_node:
                eliminate_orphan_node(obs)
            while True:
                self.step_counter += 1
                action, logprob, recons_loss = self.choose_action(obs, decode_type='sampling')
                next_obs, reward, done, info = self.env.multi_step(action) \
                    if self.config.algorithm_name == 'hie-policy' else self.env.step(action)
                # remove orphan node from obs if necessary
                if self.config.eliminate_orphan_node:
                    eliminate_orphan_node(next_obs)
                # print('reward', reward)
                episode_reward += reward['sumup'] if self.config.algorithm_name == 'hie-policy' else reward
                # save step data into buffer
                self.insert_step_data(
                    obs, action, logprob, reward if self.config.algorithm_name == 'hie-policy' else reward, next_obs, recons_loss, done)
                if self.buffer.get_buffer_size() >= self.config.batch_size:
                    # save model at fixed freq
                    if self.learn_counter % self.config.save_model_freq == 0 and self.config.save_model:
                        self.save_model(self.episode_counter)
                    # update model base on a2c algorithm
                    actor_loss, critic_loss = self.learn()
                    # output learner res
                    self.output_res({
                        'actor_loss': actor_loss,
                        'critic_loss': critic_loss,
                    }, self.learn_counter)
                    # eval model if necessary
                    if self.config.add_eval_stage and self.learn_counter % self.config.eval_freq_in_train == 0:
                        self.eval()
                if done:
                    print('episode_reward', episode_reward)
                    print('episode_length', self.step_counter)
                    self.output_res({
                        'episode_reward': episode_reward,
                        'episode_length': self.step_counter,
                    }, self.episode_counter)
                    # reset step counter and update episode counter
                    self.step_counter = 0
                    self.episode_counter += 1
                    # decay stop_eps if necessary
                    if self.config.stop_eps_decay and self.episode_counter % self.config.stop_eps_decay_freq == 0:
                        self.config.random_stop_eps = max(
                            self.config.stop_eps_decay_min, self.config.random_stop_eps - self.config.stop_eps_decay_rate)
                    break
                obs = next_obs

    def output_res(self, train_infos, total_num_steps):
        if not self.config.output_res:
            return
        for k, v in train_infos.items():
            self.writter.add_scalars(k, {k: v}, total_num_steps)

    @torch.no_grad()
    def eval(self):
        self.eval_step_counter = 0
        self.eval_episode_counter = 0
        all_episode_rewards = []
        all_episode_lengths = []
        if self.config.load_model:
            self.load_model()
        # load graph from saved data
        # graph = load_graph(self.config.eval_graph_path) if self.config.eval_with_preload_graph else None
        env = copy.deepcopy(self.env)
        for episode in range(self.config.eval_episodes):
            for graph in self.graphs:
                episode_reward = 0
                obs = env.reset(copy.deepcopy(graph))
                # remove orphan node from obs if necessary
                if self.config.eliminate_orphan_node:
                    eliminate_orphan_node(obs)
                while True:
                    self.eval_step_counter += 1
                    action, _, _ = self.choose_action(obs, decode_type='greedy')
                    next_obs, reward, done, info = env.multi_step(action) \
                        if self.config.algorithm_name == 'hie-policy' else env.step(action)
                    # remove orphan node from obs if necessary
                    if self.config.eliminate_orphan_node:
                        eliminate_orphan_node(next_obs)
                    # print('reward', reward)
                    # print('action', len(action))
                    episode_reward += reward['sumup'] if self.config.algorithm_name == 'hie-policy' else reward
                    if done:
                        all_episode_rewards.append(episode_reward)
                        all_episode_lengths.append(self.eval_step_counter)
                        # self.output_res({'episode_reward': episode_reward}, self.episode_counter)
                        self.eval_step_counter = 0
                        self.eval_episode_counter += 1
                        break
                    obs = next_obs
        # output mean episode_reward
        self.output_res({
            'eval_episode_reward': np.mean(all_episode_rewards),
            'eval_episode_length': np.mean(all_episode_lengths),
        }, self.episode_counter)
        print('eval_episode_reward', np.mean(all_episode_rewards))
        print('eval_episode_length', np.mean(all_episode_lengths))

    @torch.no_grad()
    def eval_graph(self):
        all_episode_rewards = []
        if self.config.load_model:
            self.load_model()
        # load graph from saved data
        # graph = load_graph(self.config.eval_graph_path) if self.config.eval_with_preload_graph else None
        env = copy.deepcopy(self.env)
        graph, episode_reward, step = self.graphs[0], 0, 0
        obs = env.reset(copy.deepcopy(graph))
        # remove orphan node from obs if necessary
        if self.config.eliminate_orphan_node:
            eliminate_orphan_node(obs)
        while True:
            if 'all_connectivity' in obs:
                for connectivity in obs['all_connectivity']:
                    self.output_res({'connectivity': connectivity, }, step)
                    step += 1
            else:
                self.output_res({'connectivity': obs['connectivity'], }, step)
                step += 1
            action, _, _ = self.choose_action(obs, decode_type='greedy')
            next_obs, reward, done, info = env.multi_step(action) \
                if self.config.algorithm_name == 'hie-policy' else env.step(action)
            # remove orphan node from obs if necessary
            if self.config.eliminate_orphan_node:
                eliminate_orphan_node(next_obs)
            # print('reward', reward)
            # print('action', len(action))
            episode_reward += reward['sumup'] if self.config.algorithm_name == 'hie-policy' else reward
            if done:
                all_episode_rewards.append(episode_reward)
                # self.output_res({'episode_reward': episode_reward}, self.episode_counter)
                break
            obs = next_obs
        # output mean episode_reward
        # self.output_res({
        #     'eval_episode_reward': np.mean(all_episode_rewards),
        #     'eval_episode_length': np.mean(all_episode_lengths),
        # }, self.episode_counter)
        print('eval_episode_reward', np.mean(all_episode_rewards))


#################################################################


class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size, std=0.0):
        super(ActorCritic, self).__init__()

        self.critic = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

        self.actor = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_outputs),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        value = self.critic(x)
        probs = self.actor(x)
        dist = Categorical(probs)
        return dist, value

#######################################################################
