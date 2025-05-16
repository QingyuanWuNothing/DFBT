import d4rl
import gym
import numpy as np
import torch
from tqdm import trange
from rich import print
import gym
from collections import deque

def compute_mean_std(data, eps=1e-3):
    mean = data.mean(0)
    std = data.std(0) + eps
    return mean, std

def normalize_data(data, mean, std):
    return (data - mean) / std

def wrap_env(env, state_mean, state_std, reward_scale=1.0):
    def normalize_state(state):
        return (state - state_mean) / state_std
    def scale_reward(reward):
        return reward_scale * reward
    env = gym.wrappers.TransformObservation(env, normalize_state)
    if reward_scale != 1.0:
        env = gym.wrappers.TransformReward(env, scale_reward)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env.state_mean = state_mean
    env.state_std = state_std
    
    return env

class ReplayBuffer:
    def __init__(self, observation_dim, action_dim, buffer_size=int(1e7), device="cpu"):

        self._observation_dim = observation_dim
        self._action_dim = action_dim
        self._buffer_size = buffer_size
        self._pointer = 0
        self._size = 0

        self._observations = torch.zeros((buffer_size, observation_dim), dtype=torch.float32, device=device)
        self._actions = torch.zeros((buffer_size, action_dim), dtype=torch.float32, device=device)
        self._rewards = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._next_observations = torch.zeros((buffer_size, observation_dim), dtype=torch.float32, device=device)
        self._dones = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._device = device

    def _to_tensor(self, data):
        return torch.tensor(data, dtype=torch.float32, device=self._device)

    def load_d4rl_dataset(self, dataset_name):
        env = gym.make(dataset_name)
        dataset = env.get_dataset()
        n_transitions = dataset["observations"].shape[0]
        self._observations[self._pointer: self._pointer+n_transitions] = self._to_tensor(dataset["observations"])
        self._actions[self._pointer: self._pointer+n_transitions] = self._to_tensor(dataset["actions"])
        self._rewards[self._pointer: self._pointer+n_transitions] = self._to_tensor(dataset["rewards"]).unsqueeze(-1)
        self._next_observations[self._pointer: self._pointer+n_transitions] = self._to_tensor(dataset["next_observations"])
        self._dones[self._pointer: self._pointer+n_transitions] = self._to_tensor(np.logical_or(dataset["terminals"], dataset["timeouts"])).unsqueeze(-1)
        self._pointer += n_transitions
        self._size += n_transitions
        # print(f'loaded {dataset_name}, n_transitions {n_transitions}')

    def generate_sample_prior(self, batch_size=2048):
        sample_prior = np.arange(self._size)
        np.random.shuffle(sample_prior)
        self._sample_prior = np.array_split(
            sample_prior, 
            self._size // batch_size
        )
        return self._sample_prior

    def sample(self, indices=None, batch_size=2048):
        if indices is None:
            indices = np.random.randint(0, self._size, size=batch_size)
        observations = self._observations[indices]
        actions = self._actions[indices]
        rewards = self._rewards[indices]
        next_observations = self._next_observations[indices]
        dones = self._dones[indices]
        return [observations, actions, rewards, next_observations, dones]

    def normalize(self):
        mean, std = compute_mean_std(self._rewards[:self._pointer])
        self._rewards[:self._pointer] = normalize_data(self._rewards[:self._pointer], mean, std)

class DelayBuffer:
    def __init__(self, observation_dim, action_dim, delay=5, buffer_size=int(3e6), device="cpu"):

        self._observation_dim = observation_dim
        self._action_dim = action_dim
        self._delay = delay
        self._buffer_size = buffer_size
        self._pointer = 0
        self._size = 0

        self._observations = torch.zeros((buffer_size, delay + 1, observation_dim), dtype=torch.float32, device=device)
        self._actions = torch.zeros((buffer_size, delay + 1, action_dim), dtype=torch.float32, device=device)
        self._rewards = torch.zeros((buffer_size, delay + 1, 1), dtype=torch.float32, device=device)
        self._dones = torch.zeros((buffer_size, delay + 1, 1), dtype=torch.float32, device=device)
        self._masks = torch.zeros((buffer_size, delay + 1, 1), dtype=torch.float32, device=device)
        self._device = device

        self._padding_observations = torch.zeros((delay + 1, observation_dim), dtype=torch.float32, device=device)
        self._padding_actions = torch.zeros((delay + 1, action_dim), dtype=torch.float32, device=device)
        self._padding_rewards = torch.zeros((delay + 1, 1), dtype=torch.float32, device=device)
        self._padding_dones = torch.zeros((delay + 1, 1), dtype=torch.float32, device=device)

    def _to_tensor(self, data):
        return torch.tensor(data, dtype=torch.float32, device=self._device)

    def load_d4rl_dataset(self, dataset_name):
        env = gym.make(dataset_name)
        dataset = env.get_dataset()
        n_transitions = dataset["observations"].shape[0]
        delay_seq = {
            'observations': deque(maxlen=self._delay+1),
            'actions': deque(maxlen=self._delay+1),
            'rewards': deque(maxlen=self._delay+1),
            'dones': deque(maxlen=self._delay+1),
        }
        # for i in trange(n_transitions):
        for i in trange(5000):
            delay_seq["observations"].append(dataset["observations"][i])
            delay_seq["actions"].append(dataset["actions"][i])
            delay_seq["rewards"].append(dataset["rewards"][i])
            delay_seq["dones"].append(np.logical_or(dataset["terminals"][i], dataset["timeouts"][i]))
            if len(delay_seq['observations']) != self._delay + 1:
                # continue no padding
                # print('no padding')
                continue
                # padding
                padding_length = self._delay+1 - len(delay_seq['observations'])
                self._observations[self._size] = torch.cat(
                    (self._to_tensor(np.array(list(delay_seq["observations"]))),
                     self._padding_observations[:padding_length]), dim=0)
                self._actions[self._size] = torch.cat(
                    (self._to_tensor(np.array(list(delay_seq["actions"]))),
                     self._padding_actions[:padding_length]), dim=0)
                self._rewards[self._size] = torch.cat(
                    (self._to_tensor(np.array(list(delay_seq["rewards"]))).unsqueeze(-1),
                     self._padding_rewards[:padding_length]), dim=0)
                self._dones[self._size] = torch.cat(
                    (self._to_tensor(np.array(list(delay_seq["dones"]))).unsqueeze(-1),
                     self._padding_dones[:padding_length]), dim=0)
                self._masks[self._size][padding_length:] = 1
            else:
                self._observations[self._size] = self._to_tensor(np.array(list(delay_seq["observations"])))
                self._actions[self._size] = self._to_tensor(np.array(list(delay_seq["actions"])))
                self._rewards[self._size] = self._to_tensor(np.array(list(delay_seq["rewards"]))).unsqueeze(-1)
                self._dones[self._size] = self._to_tensor(np.array(list(delay_seq["dones"]))).unsqueeze(-1)
            self._pointer += 1
            self._size += 1
        # print(f'loaded {dataset_name}, n_transitions {n_transitions}')
        # print(f'loaded {dataset_name}, n_transitions {self._size}')

    def generate_sample_prior(self, batch_size=256):
        sample_prior = np.arange((self._size // batch_size) * batch_size)
        np.random.shuffle(sample_prior)
        self._sample_prior = np.array_split(
            sample_prior, 
            self._size // batch_size
        )
        # print(f'generating sample prior {len(self._sample_prior)}')
        return self._sample_prior

    def sample(self, indices=None, batch_size=256):
        if indices is None:
            indices = np.random.randint(0, self._size, size=batch_size)
        observations = self._observations[indices]
        actions = self._actions[indices]
        rewards = self._rewards[indices]
        dones = self._dones[indices]
        masks = self._masks[indices]
        return [observations, actions, rewards, dones, masks]

    def normalize_reward(self):
        self.reward_mean = self._rewards[:self._size].mean()
        self.reward_std = self._rewards[:self._size].std()
        self._rewards[:self._size] -= self.reward_mean
        self._rewards[:self._size] /= self.reward_std
