import torch
import numpy as np
import random
class ReplayBuffer():
    def __init__(self, buffer_size, observation_dim, action_dim):
        super().__init__()
        self.buffer = {
            'obs': torch.zeros((buffer_size, observation_dim), dtype=torch.float32),
            'action': torch.zeros((buffer_size, action_dim), dtype=torch.float32),
            'reward': torch.zeros(buffer_size, 1, dtype=torch.float32),
            'next_obs': torch.zeros((buffer_size, observation_dim), dtype=torch.float32),
            'done': torch.zeros(buffer_size, 1, dtype=torch.float32),
        }

        self.buffer_size = buffer_size
        self.buffer_len = 0
        self.buffer_ptr = 0

    def store(self, obs, action, reward, next_obs, done):
        self.buffer['obs'][self.buffer_ptr] = obs
        self.buffer['action'][self.buffer_ptr] = torch.FloatTensor(action)
        self.buffer['reward'][self.buffer_ptr] = torch.tensor(reward, dtype=torch.float32).unsqueeze(0)
        self.buffer['next_obs'][self.buffer_ptr] = next_obs
        self.buffer['done'][self.buffer_ptr] = torch.tensor(done, dtype=torch.float32).unsqueeze(0)

        self.buffer_ptr += 1
        if self.buffer_ptr >= self.buffer_size:
            self.buffer_ptr = 0

        if self.buffer_len < self.buffer_size:
            self.buffer_len += 1

    def sample(self, batch_size, device):
        indices = np.random.choice(self.buffer_len, size=batch_size)
        b_obs = self.buffer['obs'][indices].to(device)
        b_action = self.buffer['action'][indices].to(device)
        b_reward = self.buffer['reward'][indices].to(device)
        b_next_obs = self.buffer['next_obs'][indices].to(device)
        b_done = self.buffer['done'][indices].to(device)
        return b_obs, b_action, b_reward, b_next_obs, b_done

class LatentBuffer():
    def __init__(self, buffer_size, observation_dim, latent_dim, action_dim):
        super().__init__()
        self.buffer = {
            'obs': torch.zeros((buffer_size, observation_dim), dtype=torch.float32),
            'latent': torch.zeros((buffer_size, latent_dim), dtype=torch.float32),
            'action': torch.zeros((buffer_size, action_dim), dtype=torch.float32),
            'reward': torch.zeros(buffer_size, 1, dtype=torch.float32),
            'next_obs': torch.zeros((buffer_size, observation_dim), dtype=torch.float32),
            'next_latent': torch.zeros((buffer_size, latent_dim), dtype=torch.float32),
            'done': torch.zeros(buffer_size, 1, dtype=torch.float32),
        }

        self.buffer_size = buffer_size
        self.buffer_len = 0
        self.buffer_ptr = 0

    def store(self, obs, latent, action, reward, next_obs, next_latent, done):
        self.buffer['obs'][self.buffer_ptr] = torch.FloatTensor(obs)
        self.buffer['latent'][self.buffer_ptr] = latent
        self.buffer['action'][self.buffer_ptr] = torch.FloatTensor(action)
        self.buffer['reward'][self.buffer_ptr] = torch.tensor(reward, dtype=torch.float32).unsqueeze(0)
        self.buffer['next_obs'][self.buffer_ptr] = torch.FloatTensor(next_obs)
        self.buffer['next_latent'][self.buffer_ptr] = next_latent
        self.buffer['done'][self.buffer_ptr] = torch.tensor(done, dtype=torch.float32).unsqueeze(0)

        self.buffer_ptr += 1
        if self.buffer_ptr >= self.buffer_size:
            self.buffer_ptr = 0

        if self.buffer_len < self.buffer_size:
            self.buffer_len += 1

    def sample(self, batch_size, device):
        indices = np.random.choice(self.buffer_len, size=batch_size)
        b_obs = self.buffer['obs'][indices].to(device)
        b_latent = self.buffer['latent'][indices].to(device)
        b_action = self.buffer['action'][indices].to(device)
        b_reward = self.buffer['reward'][indices].to(device)
        b_next_obs = self.buffer['next_obs'][indices].to(device)
        b_next_latent = self.buffer['next_latent'][indices].to(device)
        b_done = self.buffer['done'][indices].to(device)
        return b_obs, b_latent, b_action, b_reward, b_next_obs, b_next_latent, b_done

class LatentsBuffer():
    def __init__(self, buffer_size, observation_dim, latent_dim, action_dim):
        super().__init__()
        self.buffer = {
            'obs': torch.zeros((buffer_size, observation_dim), dtype=torch.float32),
            'latent': torch.zeros((buffer_size, latent_dim), dtype=torch.float32),
            'df_latent': torch.zeros((buffer_size, latent_dim), dtype=torch.float32),
            'action': torch.zeros((buffer_size, action_dim), dtype=torch.float32),
            'reward': torch.zeros(buffer_size, 1, dtype=torch.float32),
            'next_obs': torch.zeros((buffer_size, observation_dim), dtype=torch.float32),
            'next_latent': torch.zeros((buffer_size, latent_dim), dtype=torch.float32),
            'next_df_latent': torch.zeros((buffer_size, latent_dim), dtype=torch.float32),
            'done': torch.zeros(buffer_size, 1, dtype=torch.float32),
        }

        self.buffer_size = buffer_size
        self.buffer_len = 0
        self.buffer_ptr = 0

    def store(self, obs, latent, df_latent, action, reward, next_obs, next_latent, next_df_latent, done):
        self.buffer['obs'][self.buffer_ptr] = torch.FloatTensor(obs)
        self.buffer['latent'][self.buffer_ptr] = latent
        self.buffer['df_latent'][self.buffer_ptr] = df_latent
        self.buffer['action'][self.buffer_ptr] = torch.FloatTensor(action)
        self.buffer['reward'][self.buffer_ptr] = torch.tensor(reward, dtype=torch.float32).unsqueeze(0)
        self.buffer['next_obs'][self.buffer_ptr] = torch.FloatTensor(next_obs)
        self.buffer['next_latent'][self.buffer_ptr] = next_latent
        self.buffer['next_df_latent'][self.buffer_ptr] = next_df_latent
        self.buffer['done'][self.buffer_ptr] = torch.tensor(done, dtype=torch.float32).unsqueeze(0)

        self.buffer_ptr += 1
        if self.buffer_ptr >= self.buffer_size:
            self.buffer_ptr = 0

        if self.buffer_len < self.buffer_size:
            self.buffer_len += 1

    def sample(self, batch_size, device):
        indices = np.random.choice(self.buffer_len, size=batch_size)
        b_obs = self.buffer['obs'][indices].to(device)
        b_latent = self.buffer['latent'][indices].to(device)
        b_df_latent = self.buffer['df_latent'][indices].to(device)
        b_action = self.buffer['action'][indices].to(device)
        b_reward = self.buffer['reward'][indices].to(device)
        b_next_obs = self.buffer['next_obs'][indices].to(device)
        b_next_latent = self.buffer['next_latent'][indices].to(device)
        b_next_df_latent = self.buffer['next_df_latent'][indices].to(device)
        b_done = self.buffer['done'][indices].to(device)
        return b_obs, b_latent, b_df_latent, b_action, b_reward, b_next_obs, b_next_latent, b_next_df_latent, b_done

class LatentRecBuffer():
    def __init__(self, buffer_size, observation_dim, latent_dim, action_dim):
        super().__init__()
        self.buffer = {
            'obs': torch.zeros((buffer_size, observation_dim), dtype=torch.float32),
            'rec_obs': torch.zeros((buffer_size, observation_dim), dtype=torch.float32),
            'latent': torch.zeros((buffer_size, latent_dim), dtype=torch.float32),
            'action': torch.zeros((buffer_size, action_dim), dtype=torch.float32),
            'reward': torch.zeros(buffer_size, 1, dtype=torch.float32),
            'next_obs': torch.zeros((buffer_size, observation_dim), dtype=torch.float32),
            'next_rec_obs': torch.zeros((buffer_size, observation_dim), dtype=torch.float32),
            'next_latent': torch.zeros((buffer_size, latent_dim), dtype=torch.float32),
            'done': torch.zeros(buffer_size, 1, dtype=torch.float32),
        }

        self.buffer_size = buffer_size
        self.buffer_len = 0
        self.buffer_ptr = 0

    def store(self, obs, rec_obs, latent, action, reward, next_obs, next_rec_obs, next_latent, done):
        self.buffer['obs'][self.buffer_ptr] = torch.FloatTensor(obs)
        self.buffer['rec_obs'][self.buffer_ptr] = rec_obs
        self.buffer['latent'][self.buffer_ptr] = latent
        self.buffer['action'][self.buffer_ptr] = torch.FloatTensor(action)
        self.buffer['reward'][self.buffer_ptr] = torch.tensor(reward, dtype=torch.float32).unsqueeze(0)
        self.buffer['next_obs'][self.buffer_ptr] = torch.FloatTensor(next_obs)
        self.buffer['next_rec_obs'][self.buffer_ptr] = next_rec_obs
        self.buffer['next_latent'][self.buffer_ptr] = next_latent
        self.buffer['done'][self.buffer_ptr] = torch.tensor(done, dtype=torch.float32).unsqueeze(0)

        self.buffer_ptr += 1
        if self.buffer_ptr >= self.buffer_size:
            self.buffer_ptr = 0

        if self.buffer_len < self.buffer_size:
            self.buffer_len += 1

    def sample(self, batch_size, device):
        indices = np.random.choice(self.buffer_len, size=batch_size)
        b_obs = self.buffer['obs'][indices].to(device)
        b_rec_obs = self.buffer['rec_obs'][indices].to(device)
        b_latent = self.buffer['latent'][indices].to(device)
        b_action = self.buffer['action'][indices].to(device)
        b_reward = self.buffer['reward'][indices].to(device)
        b_next_obs = self.buffer['next_obs'][indices].to(device)
        b_next_rec_obs = self.buffer['next_rec_obs'][indices].to(device)
        b_next_latent = self.buffer['next_latent'][indices].to(device)
        b_done = self.buffer['done'][indices].to(device)
        return b_obs, b_rec_obs, b_latent, b_action, b_reward, b_next_obs, b_next_rec_obs, b_next_latent, b_done


class BeliefBuffer():
    def __init__(self, buffer_size, observation_dim, action_dim, delay):
        super().__init__()
        self.buffer = {
            'obs': torch.zeros((buffer_size, observation_dim), dtype=torch.float32),
            'action': torch.zeros((buffer_size, action_dim), dtype=torch.float32),
            'reward': torch.zeros(buffer_size, 1, dtype=torch.float32),
            'next_obs': torch.zeros((buffer_size, observation_dim), dtype=torch.float32),
            'done': torch.zeros(buffer_size, 1, dtype=torch.float32),

            'belief_obs': torch.zeros((buffer_size, observation_dim), dtype=torch.float32),
            'next_belief_obs': torch.zeros((buffer_size, observation_dim), dtype=torch.float32),
            'belief_actions': torch.zeros((buffer_size, delay, action_dim), dtype=torch.float32),
            'next_belief_actions': torch.zeros((buffer_size, delay, action_dim), dtype=torch.float32),
            'belief_masks': torch.zeros(buffer_size, delay, dtype=torch.float32),
            'next_belief_masks': torch.zeros(buffer_size, delay, dtype=torch.float32),
            'belief_targets': torch.zeros((buffer_size, delay, observation_dim), dtype=torch.float32),
            'next_belief_targets': torch.zeros((buffer_size, delay, observation_dim), dtype=torch.float32),

        }

        self.buffer_size = buffer_size
        self.buffer_len = 0
        self.buffer_ptr = 0

    def store(self, obs, action, reward, next_obs, done,
              belief_obs, next_belief_obs,
              belief_actions, next_belief_actions,
              belief_masks, next_belief_masks,
              belief_targets, next_belief_targets,):
        self.buffer['obs'][self.buffer_ptr] = torch.FloatTensor(obs)
        self.buffer['action'][self.buffer_ptr] = torch.FloatTensor(action)
        self.buffer['reward'][self.buffer_ptr] = torch.tensor(reward, dtype=torch.float32).unsqueeze(0)
        self.buffer['next_obs'][self.buffer_ptr] = torch.FloatTensor(next_obs)
        self.buffer['done'][self.buffer_ptr] = torch.tensor(done, dtype=torch.float32).unsqueeze(0)

        self.buffer['belief_obs'][self.buffer_ptr] = torch.FloatTensor(belief_obs.cpu())
        self.buffer['next_belief_obs'][self.buffer_ptr] = torch.FloatTensor(next_belief_obs.cpu())
        
        self.buffer['belief_actions'][self.buffer_ptr] = torch.FloatTensor(belief_actions.cpu())
        self.buffer['next_belief_actions'][self.buffer_ptr] = torch.FloatTensor(next_belief_actions.cpu())
        
        self.buffer['belief_masks'][self.buffer_ptr] = torch.FloatTensor(belief_masks.cpu())
        self.buffer['next_belief_masks'][self.buffer_ptr] = torch.FloatTensor(next_belief_masks.cpu())

        self.buffer['belief_targets'][self.buffer_ptr] = torch.FloatTensor(belief_targets.cpu())
        self.buffer['next_belief_targets'][self.buffer_ptr] = torch.FloatTensor(next_belief_targets.cpu())

        self.buffer_ptr += 1
        if self.buffer_ptr >= self.buffer_size:
            self.buffer_ptr = 0

        if self.buffer_len < self.buffer_size:
            self.buffer_len += 1

    def sample(self, batch_size, device):
        indices = np.random.choice(self.buffer_len, size=batch_size)
        b_obs = self.buffer['obs'][indices].to(device)
        b_action = self.buffer['action'][indices].to(device)
        b_reward = self.buffer['reward'][indices].to(device)
        b_next_obs = self.buffer['next_obs'][indices].to(device)
        b_done = self.buffer['done'][indices].to(device)

        b_belief_obs = self.buffer['belief_obs'][indices].to(device)
        b_next_belief_obs = self.buffer['next_belief_obs'][indices].to(device)
        
        b_belief_actions = self.buffer['belief_actions'][indices].to(device)
        b_next_belief_actions = self.buffer['next_belief_actions'][indices].to(device)
        
        b_belief_masks = self.buffer['belief_masks'][indices].to(device)
        b_next_belief_masks = self.buffer['next_belief_masks'][indices].to(device)
        
        b_belief_targets = self.buffer['belief_targets'][indices].to(device)
        b_next_belief_targets = self.buffer['next_belief_targets'][indices].to(device)


        return b_obs, b_action, b_reward, b_next_obs, b_done, b_belief_obs, b_next_belief_obs, b_belief_actions, b_next_belief_actions, b_belief_masks, b_next_belief_masks, b_belief_targets, b_next_belief_targets



class MultiStepBuffer():
    def __init__(self, buffer_size, observation_dim, latent_dim, action_dim, step):
        super().__init__()
        self.buffer = {
            'obs': torch.zeros((buffer_size, step, observation_dim), dtype=torch.float32),
            'rec_obs': torch.zeros((buffer_size, step, observation_dim), dtype=torch.float32),
            'latent': torch.zeros((buffer_size, step, latent_dim), dtype=torch.float32),
            'action': torch.zeros((buffer_size, step, action_dim), dtype=torch.float32),
            'n_return': torch.zeros(buffer_size, step, 1, dtype=torch.float32),
            'n_gamma': torch.zeros(buffer_size, step, 1, dtype=torch.float32),
            'next_obs': torch.zeros((buffer_size, step, observation_dim), dtype=torch.float32),
            'next_rec_obs': torch.zeros((buffer_size, step, observation_dim), dtype=torch.float32),
            'next_latent': torch.zeros((buffer_size, step, latent_dim), dtype=torch.float32),
            'done': torch.zeros(buffer_size, step, 1, dtype=torch.float32),
        }

        self.buffer_size = buffer_size
        self.buffer_len = 0
        self.buffer_ptr = 0
        self.step = step

    def store(self, obs, rec_obs, latent, action, n_return, n_gamma, next_obs, next_rec_obs, next_latent, done):
        self.buffer['obs'][self.buffer_ptr] = obs
        self.buffer['rec_obs'][self.buffer_ptr] = rec_obs
        self.buffer['latent'][self.buffer_ptr] = latent
        self.buffer['action'][self.buffer_ptr] = action
        self.buffer['n_return'][self.buffer_ptr] = n_return
        self.buffer['n_gamma'][self.buffer_ptr] = n_gamma
        self.buffer['next_obs'][self.buffer_ptr] = next_obs
        self.buffer['next_rec_obs'][self.buffer_ptr] = next_rec_obs
        self.buffer['next_latent'][self.buffer_ptr] = next_latent
        self.buffer['done'][self.buffer_ptr] = done

        self.buffer_ptr += 1
        if self.buffer_ptr >= self.buffer_size:
            self.buffer_ptr = 0

        if self.buffer_len < self.buffer_size:
            self.buffer_len += 1

    def sample(self, batch_size, device):
        indices = np.random.choice(self.buffer_len, size=batch_size)
        step = random.randint(0, self.step-1)
        # print(step)
        b_obs = self.buffer['obs'][indices].to(device)[:, step, :]
        b_rec_obs = self.buffer['rec_obs'][indices].to(device)[:, step, :]
        b_latent = self.buffer['latent'][indices].to(device)[:, step, :]
        b_action = self.buffer['action'][indices].to(device)[:, step, :]
        b_n_return = self.buffer['n_return'][indices].to(device)[:, step, :]
        b_n_gamma = self.buffer['n_gamma'][indices].to(device)[:, step, :]
        b_next_obs = self.buffer['next_obs'][indices].to(device)[:, step, :]
        b_next_rec_obs = self.buffer['next_rec_obs'][indices].to(device)[:, step, :]
        b_next_latent = self.buffer['next_latent'][indices].to(device)[:, step, :]
        b_done = self.buffer['done'][indices].to(device)[:, step, :]
        return b_obs, b_rec_obs, b_latent, b_action, b_n_return, b_n_gamma, b_next_obs, b_next_rec_obs, b_next_latent, b_done

