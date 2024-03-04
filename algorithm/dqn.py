import os
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch import optim
from tensorboardX import SummaryWriter
from model.dqn_model import Q_Network_Graph
from utils.replay_buffer import ReplayBuffer
from utils.utils import eliminate_orphan_node


class DQNAgent():
    def __init__(self, config, env):
        self.config = config
        self.env = env
        # data replay buffer
        self.buffer = ReplayBuffer(config=config)
        # config model and optimizer
        self.eval_net, self.target_net = Q_Network_Graph(config=config), Q_Network_Graph(config=config)
        self.optimizer = optim.Adam(self.eval_net.parameters(), config.learning_rate)
        self.loss = nn.MSELoss()
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

    def insert_step_data(self, obs, action, reward, next_obs, done):
        self.buffer.insert_step_data(obs, action, reward, next_obs, done)

    def choose_action(self, obs):
        features, adj_lists, nodes = obs['features'], obs['adjacency_list'], obs['nodes'],
        # epsilon greedy for choosing action
        if np.random.rand() <= self.config.epsilon_greedy_rate:
            action, _ = self.eval_net.get_actions(features, adj_lists, nodes)
        else:
            action = np.random.choice(nodes, size=1)[0]
        return action

    def save_model(self, training_episode):
        torch.save({
            'model': self.eval_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, self.save_model_dir + '/model_' + str(training_episode) + '.pth')

    def load_model(self):
        save_model_dict = torch.load(self.config.load_model_path)
        self.eval_net.load_state_dict(save_model_dict['model'])
        self.target_net.load_state_dict(save_model_dict['model'])
        self.optimizer.load_state_dict(save_model_dict['optimizer'])

    def output_res(self, train_infos, total_num_steps):
        if not self.config.output_res:
            return
        for k, v in train_infos.items():
            self.writter.add_scalars(k, {k: v}, total_num_steps)

    def learn(self):
        # learn 100 times then the target network update
        if self.learn_counter % self.config.target_q_update_freq == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_counter += 1

        batch_features, batch_adj_lists, batch_nodes, \
        batch_actions, batch_rewards, batch_dones, \
        batch_next_features, batch_next_adj_lists, batch_next_nodes = self.buffer.sample(self.config.batch_size)
        q_eval = self.eval_net.evaluate_action(batch_actions, batch_features, batch_adj_lists, batch_nodes)
        q_next = self.target_net.get_max_q(batch_next_features, batch_next_adj_lists, batch_next_nodes, tensor_data=True).detach()
        q_target = batch_rewards + self.config.reward_gamma * q_next
        loss = self.loss(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        print('loss', loss.item())

        return loss.item()

    def train(self):
        self.step_counter = 0
        self.episode_counter = 0
        for episode in range(self.config.total_episodes):
            episode_reward = 0
            obs = self.env.reset()
            # remove orphan node from obs if necessary
            if self.config.eliminate_orphan_node:
                eliminate_orphan_node(obs)
            while True:
                self.step_counter += 1
                action = self.choose_action(obs)
                next_obs, reward, done, info = self.env.step(action)
                # remove orphan node from obs if necessary
                if self.config.eliminate_orphan_node:
                    eliminate_orphan_node(next_obs)
                episode_reward += reward
                # save step data into buffer
                self.insert_step_data(obs, action, reward, next_obs, done)
                if self.buffer.get_buffer_size() >= self.config.memory_capacity \
                        and self.step_counter % self.config.model_update_freq == 0:
                    if self.learn_counter % self.config.save_model_freq == 0 and self.config.save_model:
                        self.save_model(self.episode_counter)
                    loss = self.learn()
                    self.output_res({'loss': loss}, self.learn_counter)
                if done:
                    print('episode_reward', episode_reward)
                    self.output_res({'episode_reward': episode_reward}, self.episode_counter)
                    self.episode_counter += 1
                    break
                obs = next_obs


if __name__ == '__main__':
    main()
