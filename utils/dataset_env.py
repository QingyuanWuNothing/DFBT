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
    def __init__(self, dataset, buffer_size=1e7, batch_size=512, device="cpu"):
        buffer_size = dataset["observations"].shape[0]
        state_dim = dataset["observations"].shape[1]
        action_dim = dataset["actions"].shape[1]

        self._buffer_size = buffer_size
        self._batch_size = batch_size
        self._pointer = 0
        self._size = 0

        self._states = torch.zeros((buffer_size, state_dim), dtype=torch.float32, device=device)
        self._actions = torch.zeros((buffer_size, action_dim), dtype=torch.float32, device=device)
        self._rewards = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._next_states = torch.zeros((buffer_size, state_dim), dtype=torch.float32, device=device)
        self._dones = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._device = device

        self.load_d4rl_dataset(dataset)
        self.generate_sample_prior()

    def _to_tensor(self, data):
        return torch.tensor(data, dtype=torch.float32, device=self._device)

    def load_d4rl_dataset(self, dataset):
        n_transitions = dataset["observations"].shape[0]
        if n_transitions > self._buffer_size:
            raise ValueError(f"Replay buffer size {self._buffer_size} is smaller than the Dataset size {n_transitions}!")
        else:
            print(f"Replay buffer size {self._buffer_size}, Dataset size {n_transitions}.")

        self._states[:n_transitions] = self._to_tensor(dataset["observations"])
        self._actions[:n_transitions] = self._to_tensor(dataset["actions"])
        self._rewards[:n_transitions] = self._to_tensor(dataset["rewards"]).unsqueeze(-1)
        self._next_states[:n_transitions] = self._to_tensor(dataset["next_observations"])
        self._dones[:n_transitions] = self._to_tensor(np.logical_or(dataset["terminals"], dataset["timeouts"])).unsqueeze(-1)
        self._size += n_transitions
        self._pointer = min(self._size, n_transitions)

    def sample(self, indices=None):
        if indices is None:
            indices = np.random.randint(0, min(self._size, self._pointer), size=self.batch_size)
        states = self._states[indices]
        actions = self._actions[indices]
        rewards = self._rewards[indices]
        next_states = self._next_states[indices]
        dones = self._dones[indices]
        return [states, actions, rewards, next_states, dones]

    def generate_sample_prior(self):
        sample_prior = np.arange(self._size)
        np.random.shuffle(sample_prior)
        self._sample_prior = np.array_split(
            sample_prior, 
            self._size // self._batch_size
        )

    def add_transition(self):
        raise NotImplementedError

class TrajectoryBuffer:
    def __init__(self, dataset, device="cpu"):
        self.trajectories_ = []
        self.sampling_prior_ = []
        self._device = device
        self.load_d4rl_trajectories(dataset)

    def _to_tensor(self, data):
        return torch.tensor(data, dtype=torch.float32, device=self._device)

    def load_d4rl_trajectories(self, dataset):
        n_transitions = dataset["observations"].shape[0]
        def init_trajectory():
            trajectory = {
                'states': [],
                'actions': [],
                'rewards': [],
                'next_states': [],
                'dones': [],
            }
            return trajectory
        def trajectory_to_tensor(trajectory):
            for k in trajectory.keys():
                trajectory[k] = self._to_tensor(np.array(trajectory[k]))
            return trajectory
        trajectory = init_trajectory()
        for i in trange(n_transitions):
            trajectory["states"].append(dataset["observations"][i])
            trajectory["actions"].append(dataset["actions"][i])
            trajectory["rewards"].append(dataset["rewards"][i])
            trajectory["next_states"].append(dataset["next_observations"][i])
            trajectory["dones"].append(np.logical_or(dataset["terminals"][i], dataset["timeouts"][i]))
            done = trajectory["dones"][-1]
            if done:
                self.sampling_prior_.append(len(trajectory["states"]))
                trajectory = trajectory_to_tensor(trajectory)
                self.trajectories_.append(trajectory)
                trajectory = init_trajectory()

        self.sampling_prior_ = np.array(self.sampling_prior_) / sum(self.sampling_prior_)

    def sample(self):
        trajectory = np.random.choice(self.trajectories_, p=self.sampling_prior_)
        return trajectory

    def add_transition(self):
        raise NotImplementedError

class DelayBuffer:
    def __init__(self, dataset, buffer_size=1e7, batch_size=512, device="cpu", delay=5):
        buffer_size = dataset["observations"].shape[0]
        state_dim = dataset["observations"].shape[1]
        action_dim = dataset["actions"].shape[1]
        self._delay = delay
        self._buffer_size = buffer_size
        self._batch_size = batch_size
        self._pointer = 0
        self._size = 0

        self._states = torch.zeros((buffer_size, delay + 1, state_dim), dtype=torch.float32, device=device)
        self._actions = torch.zeros((buffer_size, delay + 1, action_dim), dtype=torch.float32, device=device)
        self._rewards = torch.zeros((buffer_size, delay + 1, 1), dtype=torch.float32, device=device)
        self._dones = torch.zeros((buffer_size, delay + 1, 1), dtype=torch.float32, device=device)
        self._masks = torch.zeros((buffer_size, delay + 1, 1), dtype=torch.float32, device=device)
        self._device = device

        self._padding_states = torch.zeros((delay + 1, state_dim), dtype=torch.float32, device=device)
        self._padding_actions = torch.zeros((delay + 1, action_dim), dtype=torch.float32, device=device)
        self._padding_rewards = torch.zeros((delay + 1, 1), dtype=torch.float32, device=device)
        self._padding_dones = torch.zeros((delay + 1, 1), dtype=torch.float32, device=device)

        self.load_d4rl_dataset(dataset)
        self.generate_sample_prior()

    def _to_tensor(self, data):
        return torch.tensor(data, dtype=torch.float32, device=self._device)

    def load_d4rl_dataset(self, dataset):
        n_transitions = dataset["observations"].shape[0]
        if n_transitions > self._buffer_size:
            raise ValueError(f"Replay buffer size {self._buffer_size} is smaller than the Dataset size {n_transitions}!")
        else:
            print(f"Replay buffer size {self._buffer_size}, Dataset size {n_transitions}.")
        delay_seq = {
            'states': deque(maxlen=self._delay+1),
            'actions': deque(maxlen=self._delay+1),
            'rewards': deque(maxlen=self._delay+1),
            'dones': deque(maxlen=self._delay+1),
        }
        for i in trange(n_transitions):
            delay_seq["states"].append(dataset["observations"][i])
            delay_seq["actions"].append(dataset["actions"][i])
            delay_seq["rewards"].append(dataset["rewards"][i])
            delay_seq["dones"].append(np.logical_or(dataset["terminals"][i], dataset["timeouts"][i]))
            if len(delay_seq['states']) != self._delay+1:
                # padding
                padding_length = self._delay+1 - len(delay_seq['states'])
                self._states[self._size] = torch.cat(
                    (self._to_tensor(np.array(list(delay_seq["states"]))),
                     self._padding_states[:padding_length]), dim=0)
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
                self._states[self._size] = self._to_tensor(np.array(list(delay_seq["states"])))
                self._actions[self._size] = self._to_tensor(np.array(list(delay_seq["actions"])))
                self._rewards[self._size] = self._to_tensor(np.array(list(delay_seq["rewards"]))).unsqueeze(-1)
                self._dones[self._size] = self._to_tensor(np.array(list(delay_seq["dones"]))).unsqueeze(-1)
            self._size += 1

            done = delay_seq["dones"][-1]

            if done:
                delay_seq = {
                    'states': deque(maxlen=self._delay+1),
                    'actions': deque(maxlen=self._delay+1),
                    'rewards': deque(maxlen=self._delay+1),
                    'dones': deque(maxlen=self._delay+1),
                }


    def sample(self, indices=None):
        if indices is None:
            indices = np.random.randint(0, self._size, size=self._batch_size)
        states = self._states[indices]
        actions = self._actions[indices]
        rewards = self._rewards[indices]
        dones = self._dones[indices]
        masks = self._masks[indices]
        return [states, actions, rewards, dones, masks]

    def generate_sample_prior(self):
        sample_prior = np.arange(self._size)
        np.random.shuffle(sample_prior)
        self._sample_prior = [sample_prior[i: i + self._batch_size] for i in range(0, self._size, self._batch_size)]

    def get_sample_prior(self):
        return self._sample_prior

    def add_transition(self):
        raise NotImplementedError

def make_replay_buffer_env(dataset_name):
    env = gym.make(dataset_name)
    # dataset = d4rl.qlearning_dataset(env)
    dataset = env.get_dataset()
    state_mean, state_std = compute_mean_std(dataset["observations"], eps=1e-7)
    dataset["observations"] = normalize_data(dataset["observations"], state_mean, state_std)
    dataset["next_observations"] = normalize_data(dataset["next_observations"], state_mean, state_std)
    
    action_mean, action_std = compute_mean_std(dataset["actions"], eps=1e-7)
    dataset["actions"] = normalize_data(dataset["actions"], action_mean, action_std)

    reward_mean, reward_std = compute_mean_std(dataset["rewards"], eps=1e-7)
    dataset["rewards"] = normalize_data(dataset["rewards"], reward_mean, reward_std)

    env = wrap_env(env, state_mean=state_mean, state_std=state_std)
    replay_buffer = ReplayBuffer(dataset)
    return replay_buffer, env

def make_trajectory_buffer_env(dataset_name):
    env = gym.make(dataset_name)
    # dataset = d4rl.qlearning_dataset(env)
    dataset = env.get_dataset()
    state_mean, state_std = compute_mean_std(dataset["observations"], eps=1e-7)
    dataset["observations"] = normalize_data(dataset["observations"], state_mean, state_std)
    dataset["next_observations"] = normalize_data(dataset["next_observations"], state_mean, state_std)
    env = wrap_env(env, state_mean=state_mean, state_std=state_std)
    replay_buffer = TrajectoryBuffer(dataset)
    return replay_buffer, env

def make_delay_buffer_env(dataset_name, delay, batch_size, state_mean=None, state_std=None):
    env = gym.make(dataset_name)
    # dataset = d4rl.qlearning_dataset(env)
    dataset = env.get_dataset()
    if state_mean is None and state_std is None:
        state_mean, state_std = compute_mean_std(dataset["observations"], eps=1e-7)
    else:
        print(f'mean {state_mean}, std {state_std}')
    dataset["observations"] = normalize_data(dataset["observations"], state_mean, state_std)
    dataset["next_observations"] = normalize_data(dataset["next_observations"], state_mean, state_std)

    env = wrap_env(env, state_mean=state_mean, state_std=state_std)
    delay_buffer = DelayBuffer(dataset=dataset, batch_size=batch_size, delay=delay)
    return delay_buffer, env

