import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)

import torch
import torch.nn.functional as F
import torch.optim as optim
from utils.network import AutoEncoder, Direct_Forecasting_Belief
from utils.tool import get_configs
from utils.dataset import DelayBuffer
from tqdm import trange, tqdm
from tensorboardX import SummaryWriter
import gym
import numpy as np
import argparse

class BeliefTrainer():
    def __init__(self, config):
        self.config = config
        self.logger = SummaryWriter(config['exp_tag'])
        self.logger.add_text(
            "config",
            "|parametrix|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in config.items()])),
        )
        self.log_dict = {}
        env = gym.make(f"{config['dataset_name']}-random-v2")
        self.observation_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]

    def train(self):
        self.replay_buffer = DelayBuffer(
            self.observation_dim, 
            action_dim=self.action_dim,
            delay=self.config['delay'],
        )
        for policy in ['random', 'medium', 'expert']:
            dataset_name = f"{config['dataset_name']}-{policy}-v2"
            self.replay_buffer.load_d4rl_dataset(dataset_name)
        self.replay_buffer.normalize_reward()

        self.epoch = 0
        self.auto_encoder = AutoEncoder(
            input_dim=self.observation_dim, 
            hidden_dim=256, 
            latent_dim=self.config['latent_dim']).to("cuda")
        self.dynamic = Direct_Forecasting_Belief(latent_dim=config['latent_dim'], 
                                        condition_dim=self.action_dim, 
                                        seq_len=self.config['delay'], 
                                        hidden_dim=self.config['latent_dim'],
                                        num_layers=self.config['num_layers'],
                                        num_heads=self.config['num_heads'],
                                        ).to(self.config['device'])

        self.optimizer = optim.AdamW(
            list(self.auto_encoder.parameters()) + list(self.dynamic.parameters()), 
            lr=self.config['lr'],
            weight_decay=1e-4,
            betas=(0.9, 0.999),
        )

        self.replay_buffer.generate_sample_prior()
        for self.epoch in trange(1, self.config['total_epoch'] + 1):
            self.train_directly_forecasting_belief()
            if self.epoch % 10 == 0:
                # print("saving")
                torch.save({
                    'step': self.epoch,
                    'auto_encoder': self.auto_encoder.state_dict(), 
                    'latent_dynamic': self.dynamic.state_dict(), 
                    'reward_mean': self.replay_buffer.reward_mean,
                    'reward_std': self.replay_buffer.reward_std,
                    },
                    f"dfbt_checkpoints/{self.config['dataset_name']}_Delay_{self.config['delay']}.pth")
                # print("saved")

    def train_directly_forecasting_belief(self):
        for indices in self.replay_buffer._sample_prior:
            print(len(indices))
            exit()
            states, actions, rewards, dones, masks = self.replay_buffer.sample(indices)
            states = states.to(self.config['device'])
            actions = actions.to(self.config['device'])
            rewards = rewards.to(self.config['device'])
            masks = masks[:, 1:, 0].to(self.config['device'])

            latents = self.auto_encoder.encode(states)
            timesteps = torch.arange(0, self.config['delay'], dtype=torch.int32).to(self.config['device'])
            z = self.dynamic(latents=latents[:, :1, :], 
                             actions=actions[:, :self.config['delay'], :],
                             rewards=rewards[:, :self.config['delay'], :],
                             timesteps=timesteps,
                             masks=masks)
            rec_states = self.auto_encoder.decode(z)
            loss = F.mse_loss(rec_states, states[:, 1:, :], reduction='none').mean(-1)
            loss = ((1-masks) * loss).mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        self.log_dict["loss"] = loss.item()


    def logging(self):
        for k in self.log_dict.keys():
            self.logger.add_scalar(k, self.log_dict[k], global_step=self.epoch)
        self.log_dict = {}

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, default="halfcheetah")
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument('--seed', type=int, default=2025)
parser.add_argument("--total_epoch", type=int, default=int(1e3))
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--delay", type=int, default=128)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--latent_dim", type=int, default=256)
parser.add_argument("--hidden_dim", type=int, default=256)
parser.add_argument("--num_layers", type=int, default=10)
parser.add_argument("--num_heads", type=int, default=4)
parser.add_argument("--attention_dropout", type=float, default=0.1)
parser.add_argument("--residual_dropout", type=float, default=0.1)
parser.add_argument("--hidden_dropout", type=float, default=0.1)

if __name__ == "__main__":
    config = vars(parser.parse_args())
    config['exp_tag'] = f"logs_dfbt/{config['dataset_name']}/{config['delay']}"
    trainer = BeliefTrainer(config)
    trainer.train()