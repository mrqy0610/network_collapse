"""
    This file is copied/apdated from https://github.com/berkeleydeeprlcourse/homework/tree/master/hw3
"""
import ray
import torch
import random
import numpy as np

from utils.utils import process_step_reward


class ReplayBuffer:
    def __init__(self, config):
        self.config = config
        self.size = config.memory_capacity
        self.device = config.device
        # config all training data
        self.all_obs = []
        self.all_actions = []
        self.all_rewards = []
        self.all_next_obs = []
        self.all_dones = []
        # pointer
        self.step = 0

    def get_buffer_size(self):
        return len(self.all_obs)

    def insert_step_data(self, obs, action, reward, next_obs, done):
        reward = process_step_reward(
            reward, obs['origin_num_nodes'],
            reward_reflect=self.config.reward_reflect,
            reward_normalization=self.config.reward_normalization,
            multi_step=False
        )
        if len(self.all_obs) >= self.size:
            self.all_obs[self.step] = obs
            self.all_actions[self.step] = action
            self.all_rewards[self.step] = reward
            self.all_next_obs[self.step] = next_obs
            self.all_dones[self.step] = done
        else:
            self.all_obs.append(obs)
            self.all_actions.append(action)
            self.all_rewards.append(reward)
            self.all_next_obs.append(next_obs)
            self.all_dones.append(done)
        self.step = (self.step + 1) % self.size

    def sample(self, batch_size):
        idxes = np.random.choice(len(self.all_obs), batch_size, replace=False)
        # split all obs into different types
        batch_features = [torch.from_numpy(self.all_obs[i]['features']).to(dtype=torch.float32, device=self.device) for i in idxes]
        batch_adj_lists = [self.all_obs[i]['adjacency_list'] for i in idxes]
        batch_nodes = [torch.from_numpy(self.all_obs[i]['nodes']).to(dtype=torch.int32) for i in idxes]
        # actions, rewards, dones
        batch_actions = torch.cat([
            torch.from_numpy(np.array(self.all_actions[i])).unsqueeze(0) for i in idxes
        ], dim=0).to(device=self.device).unsqueeze(-1)
        batch_rewards = torch.cat([
            torch.from_numpy(np.array(self.all_rewards[i])).unsqueeze(0) for i in idxes
        ], dim=0).to(dtype=torch.float32, device=self.device).unsqueeze(-1)
        if self.config.reward_normalization:
            print('batch_rewards', batch_rewards.shape)
        batch_dones = torch.cat([
            torch.from_numpy(np.array(self.all_dones[i])).unsqueeze(0) for i in idxes
        ], dim=0).to(device=self.device).unsqueeze(-1)
        # split all next_obs into different types
        batch_next_features = [torch.from_numpy(self.all_next_obs[i]['features']).to(dtype=torch.float32, device=self.device) for i in idxes]
        batch_next_adj_lists = [self.all_next_obs[i]['adjacency_list'] for i in idxes]
        batch_next_nodes = [torch.from_numpy(self.all_next_obs[i]['nodes']).to(dtype=torch.int32) for i in idxes]

        return batch_features, batch_adj_lists, batch_nodes,\
               batch_actions, batch_rewards, batch_dones, \
               batch_next_features, batch_next_adj_lists, batch_next_nodes


class RolloutBuffer:
    def __init__(self, config):
        self.config = config
        self.size = config.memory_capacity
        self.device = config.device
        # config all training data
        self.all_obs = []
        self.all_actions = []
        self.all_logprobs = []
        self.all_rewards = []
        self.all_gammas = []
        self.all_next_obs = []
        self.all_recons_loss = []
        self.all_dones = []

    def insert_step_data(self, obs, action, logprob, reward, next_obs, recons_loss, done):
        reward = process_step_reward(
            reward, obs['origin_num_nodes'],
            reward_reflect=self.config.reward_reflect,
            reward_normalization=self.config.reward_normalization,
            multi_step=self.config.algorithm_name == 'hie-policy'
        )
        self.all_obs.append(obs)
        self.all_actions.append(action)
        self.all_logprobs.append(logprob)
        self.all_rewards.append(reward)
        self.all_gammas.append(pow(self.config.reward_gamma, action.shape[0]))
        self.all_next_obs.append(next_obs)
        self.all_recons_loss.append(recons_loss)
        self.all_dones.append(done)

    def clear(self):
        # clear memory
        del self.all_obs
        del self.all_actions
        del self.all_logprobs
        del self.all_rewards
        del self.all_next_obs
        del self.all_recons_loss
        del self.all_dones
        # resign memory
        self.all_obs = []
        self.all_actions = []
        self.all_logprobs = []
        self.all_rewards = []
        self.all_gammas = []
        self.all_next_obs = []
        self.all_recons_loss = []
        self.all_dones = []

    def get_buffer_size(self):
        return len(self.all_obs)

    def get_all_data(self):
        idxes = np.arange(len(self.all_obs))
        # split all obs into different types
        batch_features = [torch.from_numpy(self.all_obs[i]['features']).to(dtype=torch.float32, device=self.device) for i in idxes]
        batch_adj_lists = [self.all_obs[i]['adjacency_list'] for i in idxes]
        batch_nodes = [torch.from_numpy(self.all_obs[i]['nodes']).to(dtype=torch.int32) for i in idxes]
        batch_adj_matrixs = [torch.from_numpy(self.all_obs[i]['adjacency_matrix']).to(dtype=torch.int32, device=self.device) for i in idxes]
        # actions, rewards, dones and logprobs(keep grad)
        # batch_actions = torch.cat([
        #     torch.from_numpy(np.array(self.all_actions[i])).unsqueeze(0) for i in idxes
        # ], dim=0).to(device=self.device).unsqueeze(-1)
        batch_actions = [torch.from_numpy(self.all_actions[i]).to(device=self.device) for i in idxes]
        batch_logprobs = torch.cat([self.all_logprobs[i].unsqueeze(0) for i in idxes], dim=0)
        batch_rewards = torch.cat([
            torch.from_numpy(np.array(self.all_rewards[i])).unsqueeze(0) for i in idxes
        ], dim=0).to(dtype=torch.float32, device=self.device).unsqueeze(-1)
        if self.config.reward_normalization:
            batch_rewards = (batch_rewards - batch_rewards.mean(dim=0)) / (batch_rewards.std(dim=0) + 1e-6)
        batch_gammas = torch.cat([
            torch.from_numpy(np.array(self.all_rewards[i])).unsqueeze(0) for i in idxes
        ], dim=0).to(dtype=torch.float32, device=self.device).unsqueeze(-1)
        batch_recons_loss = torch.cat([self.all_recons_loss[i].unsqueeze(0) for i in idxes], dim=0) \
            if self.config.add_graph_reconstruction_loss else None
        batch_dones = torch.cat([
            torch.from_numpy(np.array(self.all_dones[i])).unsqueeze(0) for i in idxes
        ], dim=0).to(dtype=torch.float32, device=self.device).unsqueeze(-1)
        # split all next_obs into different types
        batch_next_features = [torch.from_numpy(self.all_next_obs[i]['features']).to(dtype=torch.float32, device=self.device) for i in idxes]
        batch_next_adj_lists = [self.all_next_obs[i]['adjacency_list'] for i in idxes]
        batch_next_nodes = [torch.from_numpy(self.all_next_obs[i]['nodes']).to(dtype=torch.int32) for i in idxes]
        batch_next_adj_matrixs = [torch.from_numpy(self.all_next_obs[i]['adjacency_matrix']).to(dtype=torch.int32, device=self.device) for i in idxes]
        # clear data buffer for next batch
        self.clear()

        return batch_features, batch_adj_lists, batch_nodes, batch_adj_matrixs, \
               batch_actions, batch_logprobs, batch_rewards, batch_gammas, batch_dones, batch_recons_loss, \
               batch_next_features, batch_next_adj_lists, batch_next_nodes, batch_next_adj_matrixs


class BatchBuffer:
    def __init__(self, config):
        self.config = config
        # self.size = config.batch_size
        self.size = config.memory_capacity
        self.device = config.device
        # config all training data
        self.all_obs = []
        self.all_actions_probs = []
        self.all_rewards = []
        # pointer
        self.step = 0

    def get_buffer_size(self):
        return len(self.all_obs)

    def insert_all_data(self, all_states, all_actions_probs, all_rewards):
        for state, action_prob, reward in zip(all_states, all_actions_probs, all_rewards):
            if len(self.all_obs) >= self.size:
                self.all_obs[self.step] = state
                self.all_actions_probs[self.step] = action_prob
                self.all_rewards[self.step] = reward
                self.step = (self.step + 1) % self.size
            else:
                self.all_obs.append(state)
                self.all_actions_probs.append(action_prob)
                self.all_rewards.append(reward)

    def sample(self, batch_size):
        idxes = np.random.choice(len(self.all_obs), batch_size, replace=False)
        # split all obs into different types
        batch_features = torch.from_numpy(np.array(
            [self.all_obs[i]['features'] for i in idxes])).to(dtype=torch.float32, device=self.device)
        batch_adjacency_matrixs = torch.from_numpy(np.array(
            [self.all_obs[i]['adjacency_matrix'] for i in idxes])).to(dtype=torch.float32, device=self.device)
        # actions_probs, rewards
        batch_actions_probs = torch.cat([
            torch.from_numpy(np.array(self.all_actions_probs[i])).unsqueeze(0) for i in idxes
        ], dim=0).to(device=self.device)
        batch_rewards = torch.cat([
            torch.from_numpy(np.array(self.all_rewards[i])).unsqueeze(0) for i in idxes
        ], dim=0).to(dtype=torch.float32, device=self.device).unsqueeze(-1)

        return batch_features, batch_adjacency_matrixs, batch_actions_probs, batch_rewards


@ray.remote
class ShareBuffer:
    def __init__(self, config):
        self.config = config
        # self.size = config.batch_size
        self.size = config.memory_capacity
        self.device = config.device
        # config all training data
        self.all_obs = []
        self.all_actions_probs = []
        self.all_rewards = []
        # pointer
        self.step = 0

    def get_buffer_size(self):
        return len(self.all_obs)

    def insert_all_data(self, all_states, all_actions_probs, all_rewards):
        for state, action_prob, reward in zip(all_states, all_actions_probs, all_rewards):
            if len(self.all_obs) >= self.size:
                self.all_obs[self.step] = state
                self.all_actions_probs[self.step] = action_prob
                self.all_rewards[self.step] = reward
                self.step = (self.step + 1) % self.size
            else:
                self.all_obs.append(state)
                self.all_actions_probs.append(action_prob)
                self.all_rewards.append(reward)

    def sample(self, batch_size):
        idxes = np.random.choice(len(self.all_obs), batch_size, replace=False)
        # split all obs into different types
        batch_features = torch.from_numpy(np.array(
            [self.all_obs[i]['features'] for i in idxes])).to(dtype=torch.float32, device=self.device)
        batch_adjacency_matrixs = torch.from_numpy(np.array(
            [self.all_obs[i]['adjacency_matrix'] for i in idxes])).to(dtype=torch.float32, device=self.device)
        # actions_probs, rewards
        batch_actions_probs = torch.cat([
            torch.from_numpy(np.array(self.all_actions_probs[i])).unsqueeze(0) for i in idxes
        ], dim=0).to(device=self.device)
        batch_rewards = torch.cat([
            torch.from_numpy(np.array(self.all_rewards[i])).unsqueeze(0) for i in idxes
        ], dim=0).to(dtype=torch.float32, device=self.device).unsqueeze(-1)

        return batch_features, batch_adjacency_matrixs, batch_actions_probs, batch_rewards
