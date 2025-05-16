import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from utils.network import Actor, Critic
from utils.network import AutoEncoder, Direct_Forecasting_Belief
from utils.tool import get_configs
from utils.replay_buffer import MultiStepBuffer
from tqdm import trange
from tensorboardX import SummaryWriter
from rich import print
from copy import deepcopy
from collections import deque
import gym
import argparse

class Trainer():
    def __init__(self, config):
        self.logger = SummaryWriter(config['exp_tag'])
        self.logger.add_text(
            "config",
            "|parametrix|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in config.items()])),
        )
        self.config = config
        self.env = gym.make(f"{config['env_name']}")
        self.env = gym.wrappers.RecordEpisodeStatistics(self.env)
        self.eval_env = deepcopy(self.env)
        observation_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        action_high = float(self.env.action_space.high[0])
        action_low = float(self.env.action_space.low[0])

        self.auto_encoder = AutoEncoder(
            input_dim=observation_dim, 
            hidden_dim=256, 
            latent_dim=config['latent_dim']).to(config['device'])
        if config['env_name'] == 'HalfCheetah-v2':
            config['dataset_name'] = 'halfcheetah'
        elif config['env_name'] == 'Hopper-v2':
            config['dataset_name'] = 'hopper'
        elif config['env_name'] == 'Walker2d-v2':
            config['dataset_name'] = 'walker2d'
        else:
            NotImplementedError
        checkpoint = torch.load(f"dfbt_checkpoints/{config['dataset_name']}_Delay_{config['delay']}.pth", map_location=torch.device('cpu'))

        self.reward_mean = checkpoint['reward_mean']
        self.reward_std = checkpoint['reward_std']
        self.auto_encoder.load_state_dict(checkpoint['auto_encoder'])
        self.auto_encoder.eval()
        self.latent_dynamic = Direct_Forecasting_Belief(latent_dim=config['latent_dim'], 
                                        condition_dim=action_dim, 
                                        seq_len=config['delay'], 
                                        hidden_dim=config['latent_dim'],
                                        num_layers=config['num_layers'],
                                        num_heads=config['num_heads'],
                                        ).to(config['device'])
        self.latent_dynamic.load_state_dict(checkpoint['latent_dynamic'])
        self.latent_dynamic.eval()
        print('loaded direct belief transformer')
        
        self.replay_buffer = MultiStepBuffer(buffer_size=config['buffer_size'], observation_dim=observation_dim, latent_dim=config['latent_dim'], action_dim=action_dim, step=1)

        self.actor = Actor(
            latent_dim=observation_dim, 
            action_dim=action_dim,
            action_high=action_high,
            action_low=action_low).to(config['device'])
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config['lr_actor'])

        self.critic_1 = Critic(latent_dim=observation_dim, action_dim=action_dim).to(config['device'])
        self.target_1 = Critic(latent_dim=observation_dim, action_dim=action_dim).to(config['device'])
        self.target_1.load_state_dict(self.critic_1.state_dict())
        self.target_1.eval()
        self.critic_2 = Critic(latent_dim=observation_dim, action_dim=action_dim).to(config['device'])
        self.target_2 = Critic(latent_dim=observation_dim, action_dim=action_dim).to(config['device'])
        self.target_2.load_state_dict(self.critic_2.state_dict())
        self.target_2.eval()
        self.critic_optimizer = optim.Adam(list(self.critic_1.parameters()) + list(self.critic_2.parameters()), lr=config['lr_critic'])

        self.target_entropy = -torch.prod(torch.Tensor(self.env.action_space.shape).to(config['device'])).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=config['device'])
        self.alpha = self.log_alpha.exp().item()
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=config['lr_alpha'])

        self.log_metric = {}

    def train(self):
        obs = self.env.reset()
        with torch.no_grad():
            latent = self.auto_encoder.encode(torch.FloatTensor(obs).to(self.config['device']))
        delayed_deque = {
            'obs': deque(maxlen=self.config['delay'] + 1),
            'latent': deque(maxlen=self.config['delay'] + 1),
            'rec_obs': deque(maxlen=self.config['delay'] + 1),
            'action': deque(maxlen=self.config['delay']),
            'reward': deque(maxlen=self.config['delay']),
            'done': deque(maxlen=self.config['delay']),
        }
        delayed_deque['obs'].append(torch.FloatTensor(obs).to(self.config['device']))
        delayed_deque['latent'].append(latent)
        rec_obs = torch.FloatTensor(obs).to(self.config['device'])
        delayed_deque['rec_obs'].append(rec_obs)

        for self.global_step in trange(1, self.config['total_step']+1):
            if self.global_step < self.config['learn_start']:
                action = self.env.action_space.sample()
            else:
                with torch.no_grad():
                    action, _, _ = self.actor.get_action(rec_obs)
                    action = action.squeeze().cpu().numpy()
            
            next_obs, reward, done, info = self.env.step(action)

            delayed_deque['obs'].append(torch.FloatTensor(next_obs).to(self.config['device']))
            delayed_deque['action'].append(action)
            delayed_deque['reward'].append(reward)
            delayed_deque['done'].append(done)

            with torch.no_grad():
                if self.config['stochastic']:
                    random_delay = np.random.randint(0, len(delayed_deque['obs']) - 1)
                else:
                    random_delay = 0
                random_delay = 0
                next_latent = self.auto_encoder.encode(delayed_deque['obs'][random_delay])
                next_latent, latents_seq = self.get_next_latent(next_latent, list(delayed_deque['action'])[random_delay:], list(delayed_deque['reward'])[random_delay:])
                next_rec_obs = self.auto_encoder.decode(next_latent)

                max_n = len(delayed_deque['action'])
                latents_seq = latents_seq[:, :max_n+1, :]
                rec_obs_seq = self.auto_encoder.decode(latents_seq)
                
            delayed_deque['latent'].append(next_latent)
            delayed_deque['rec_obs'].append(next_rec_obs)


            for start_n in reversed(range(max(0, max_n - self.config['n_steps']), max_n)):
                n_return = np.array([pow(self.config['gamma'], n_gamma) for n_gamma in range(max_n - start_n)]) * np.array(list(delayed_deque['reward'])[start_n: max_n])
                n_return = n_return.sum()
                n_gamma = pow(self.config['gamma'], max_n - start_n)

                self.replay_buffer.store(
                    obs=delayed_deque['obs'][start_n], 
                    rec_obs=rec_obs_seq[0, start_n, :], 
                    latent=latents_seq[0, start_n, :], 
                    action=torch.tensor(delayed_deque['action'][start_n]), 
                    n_return=n_return, 
                    n_gamma=n_gamma,
                    next_obs=delayed_deque['obs'][max_n], 
                    next_rec_obs=rec_obs_seq[0, max_n, :], 
                    next_latent=latents_seq[0, max_n, :], 
                    done=delayed_deque['done'][max_n-1],
                )

            latent = next_latent
            obs = next_obs
            rec_obs = next_rec_obs
            if done:
                obs = self.env.reset()
                with torch.no_grad():
                    latent = self.auto_encoder.encode(torch.FloatTensor(obs).to(self.config['device']))
                delayed_deque = {
                    'obs': deque(maxlen=self.config['delay'] + 1),
                    'latent': deque(maxlen=self.config['delay'] + 1),
                    'rec_obs': deque(maxlen=self.config['delay'] + 1),
                    'action': deque(maxlen=self.config['delay']),
                    'reward': deque(maxlen=self.config['delay']),
                    'done': deque(maxlen=self.config['delay']),
                }
                delayed_deque['obs'].append(torch.FloatTensor(obs).to(self.config['device']))
                delayed_deque['latent'].append(latent)
                rec_obs = torch.FloatTensor(obs).to(self.config['device'])
                delayed_deque['rec_obs'].append(rec_obs)
            
            if self.global_step > self.config['learn_start']:
                b_obs, b_rec_obs, b_latent, b_action, b_n_return, b_n_gamma, b_next_obs, b_next_rec_obs, b_next_latent, b_done = self.replay_buffer.sample(batch_size=self.config['batch_size'], device=self.config['device'])

                with torch.no_grad():
                    p_next_action, p_next_log_pi, _ = self.actor.get_action(b_next_rec_obs)
                    target_1_next_val = self.target_1(b_next_obs, p_next_action)
                    target_2_next_val = self.target_2(b_next_obs, p_next_action)
                    target_next_val = torch.min(target_1_next_val, target_2_next_val) - self.alpha * p_next_log_pi
                    next_q_val = b_n_return + (1 - b_done) * b_n_gamma * (target_next_val)
                critic_1_val = self.critic_1(b_obs, b_action)
                critic_2_val = self.critic_2(b_obs, b_action)
                loss_critic_1 = F.mse_loss(critic_1_val, next_q_val)
                loss_critic_2 = F.mse_loss(critic_2_val, next_q_val)
                loss_critic = loss_critic_1 + loss_critic_2
                self.critic_optimizer.zero_grad()
                loss_critic.backward()
                self.critic_optimizer.step()
                self.log_metric['loss_critic'] = loss_critic.item()
                if self.global_step % 2 == 0:
                    for _ in range(2):
                        p_action, p_log_prob, _ = self.actor.get_action(b_rec_obs)
                        critic_1_q_val = self.critic_1(b_obs, p_action)
                        critic_2_q_val = self.critic_2(b_obs, p_action)
                        critic_q_val = torch.min(critic_1_q_val, critic_2_q_val)
                        loss_actor = ((self.alpha * p_log_prob) - critic_q_val).mean()
                        self.actor_optimizer.zero_grad()
                        loss_actor.backward()
                        self.actor_optimizer.step()
                        self.log_metric['loss_actor'] = loss_actor.item()

                        with torch.no_grad():
                            _, p_log_prob, _ = self.actor.get_action(b_rec_obs)
                        loss_alpha = (-self.log_alpha.exp() * (p_log_prob + self.target_entropy)).mean()
                        self.alpha_optimizer.zero_grad()
                        loss_alpha.backward()
                        self.alpha_optimizer.step()
                        self.alpha = self.log_alpha.exp().item()
                        self.log_metric['loss_alpha'] = loss_alpha.item()

                for param, target_param in zip(self.critic_1.parameters(), self.target_1.parameters()):
                    target_param.data.copy_(self.config['soft_update_factor'] * param.data + (1 - self.config['soft_update_factor']) * target_param.data)
                for param, target_param in zip(self.critic_2.parameters(), self.target_2.parameters()):
                    target_param.data.copy_(self.config['soft_update_factor'] * param.data + (1 - self.config['soft_update_factor']) * target_param.data)

            if self.global_step % self.config['evaluate_freq'] == 0:
                self.evaluate()

            self.logging()
        self.logger.close()
    

    def get_next_latent(self, latent, actions, rewards):
        delayed_idx = len(actions) - 1
        latent = latent.unsqueeze(0)
        timesteps_seq = torch.arange(0, self.config['delay'], dtype=torch.int32).to(self.config['device'])
        masks = torch.zeros(len(actions)).unsqueeze(0).to(self.config['device'])
        pad_masks = torch.ones(self.config['delay'] - len(actions)).unsqueeze(0).to(self.config['device'])
        masks_seq = torch.concat((masks, pad_masks), dim=-1)

        action_dim = actions[0].shape[0]
        pad_actions = torch.zeros((1, self.config['delay'] - len(actions), action_dim)).to(self.config['device'])
        actions_seq = torch.concat((torch.FloatTensor(np.array(list(actions))).unsqueeze(0).to(self.config['device']), pad_actions), dim=1)

        pad_rewards = torch.zeros((1, self.config['delay'] - len(actions))).to(self.config['device'])
        rewards_seq = torch.concat((torch.FloatTensor(np.array(list(rewards))).unsqueeze(0).to(self.config['device']), pad_rewards), dim=1).unsqueeze(-1)

        latents_seq = self.latent_dynamic(
            latents=latent, 
            actions=actions_seq,
            rewards=(rewards_seq - self.reward_mean) / self.reward_std,
            timesteps=timesteps_seq,
            masks=masks_seq
        )
        next_latent = latents_seq[:, delayed_idx, :].squeeze(0)
        latents_seq = torch.concat((
            latent.unsqueeze(1), latents_seq
        ), dim=1)
        return next_latent, latents_seq

    def evaluate(self):
        self.log_metric['eval_r'] = []
        self.log_metric['eval_l'] = []

        obs = self.eval_env.reset()
        with torch.no_grad():
            latent = self.auto_encoder.encode(torch.FloatTensor(obs).to(self.config['device']))
        delayed_deque = {
            'obs': deque(maxlen=self.config['delay'] + 1),
            'action': deque(maxlen=self.config['delay']),
            'reward': deque(maxlen=self.config['delay']),
        }
        delayed_deque['obs'].append(torch.FloatTensor(obs).to(self.config['device']))
        rec_obs = torch.FloatTensor(obs).to(self.config['device'])

        while len(self.log_metric['eval_r']) < 10:
            with torch.no_grad():
                action, _, _ = self.actor.get_action(rec_obs)
                action = action.squeeze().cpu().numpy()
            
            next_obs, reward, done, info = self.eval_env.step(action)

            delayed_deque['obs'].append(torch.FloatTensor(next_obs).to(self.config['device']))
            delayed_deque['action'].append(action)
            delayed_deque['reward'].append(reward)

            with torch.no_grad():
                if self.config['stochastic']:
                    random_delay = np.random.randint(0, len(delayed_deque['obs']) - 1)
                else:
                    random_delay = 0
                next_latent = self.auto_encoder.encode(delayed_deque['obs'][random_delay])
                next_latent, _ = self.get_next_latent(next_latent, list(delayed_deque['action'])[random_delay:], list(delayed_deque['reward'])[random_delay:])
                next_rec_obs = self.auto_encoder.decode(next_latent)

            latent = next_latent
            obs = next_obs
            rec_obs = next_rec_obs
            if done:
                self.log_metric['eval_r'].append(info['episode']['r'])
                self.log_metric['eval_l'].append(info['episode']['l'])
                obs = self.eval_env.reset()
                with torch.no_grad():
                    latent = self.auto_encoder.encode(torch.FloatTensor(obs).to(self.config['device']))
                delayed_deque = {
                    'obs': deque(maxlen=self.config['delay'] + 1),
                    'action': deque(maxlen=self.config['delay']),
                    'reward': deque(maxlen=self.config['delay']),
                }
                delayed_deque['obs'].append(torch.FloatTensor(obs).to(self.config['device']))
                rec_obs = torch.FloatTensor(obs).to(self.config['device'])

        self.log_metric['eval_r'] = np.mean(self.log_metric['eval_r'])
        self.log_metric['eval_l'] = np.mean(self.log_metric['eval_l'])
        print(f"Step: {self.global_step}, R: {self.log_metric['eval_r']}, L: {self.log_metric['eval_l']}")


    def logging(self):
        for k in self.log_metric.keys():
            self.logger.add_scalar(k, self.log_metric[k], global_step=self.global_step)
        self.log_metric = {}

parser = argparse.ArgumentParser()
parser.add_argument('--env_name', type=str, default='HalfCheetah-v2')
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
parser.add_argument('--seed', type=int, default=2025)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--total_step', type=int, default=int(1e6))
parser.add_argument('--buffer_size', type=int, default=int(1e6))
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--lr_actor', type=float, default=3e-4)
parser.add_argument('--lr_critic', type=float, default=1e-3)
parser.add_argument('--lr_alpha', type=float, default=1e-3)
parser.add_argument('--latent_dim', type=int, default=256)
parser.add_argument('--num_layers', type=int, default=10)
parser.add_argument('--num_heads', type=int, default=4)
parser.add_argument('--attention_dropout', type=float, default=0.1)
parser.add_argument('--residual_dropout', type=float, default=0.1)
parser.add_argument('--hidden_dropout', type=float, default=0.1)
parser.add_argument('--soft_update_factor', type=float, default=5e-3)
parser.add_argument('--learn_start', type=int, default=int(5e3))
parser.add_argument('--evaluate_freq', type=int, default=int(1e4))
parser.add_argument('--delay', type=int, default=128)
parser.add_argument('--n_steps', type=int, default=8)
parser.add_argument('--stochastic', action='store_true')

if __name__ == "__main__":
    config = vars(parser.parse_args())
    assert config['n_steps'] <= config['delay'], "n_steps should be less than delay"
    if config['stochastic']:
        config['exp_tag'] = f"logs_dfbt_sac/dfbt_sac_{config['n_steps']}/{config['env_name']}/stochastic_{config['delay']}/SEED_{config['seed']}"
    else:
        config['exp_tag'] = f"logs_dfbt_sac/dfbt_sac_{config['n_steps']}/{config['env_name']}/deterministic_{config['delay']}/SEED_{config['seed']}"
    trainer = Trainer(config)
    trainer.train()
