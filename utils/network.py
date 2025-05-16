import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import trange

class AutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim, bias=False),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim, bias=False),
        )

    def forward(self, x):
        z = self.encoder(x)
        x_rec = self.decoder(z)
        return x_rec, z

    def encode(self, x):
        z = self.encoder(x)
        return z

    def decode(self, z):
        x_ = self.decoder(z)
        return x_

class TransformerBlock(nn.Module):
    def __init__(
        self,
        seq_len,
        hidden_dim,
        num_heads,
        attention_dropout,
        residual_dropout,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(residual_dropout)

        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads, attention_dropout, batch_first=True
        )
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(residual_dropout),
        )
        self.register_buffer(
            "causal_mask", ~torch.tril(torch.ones(seq_len, seq_len)).to(bool)
        )
        self.seq_len = seq_len

    def forward(self, x, padding_mask):
        causal_mask = self.causal_mask[: x.shape[1], : x.shape[1]]
        x = self.norm1(x)
        attention_out = self.attention(
            query=x,
            key=x,
            value=x,
            attn_mask=causal_mask,
            key_padding_mask=padding_mask,
            need_weights=False,
        )[0]
        x = x + self.drop(attention_out)
        x = x + self.mlp(self.norm2(x))
        return x
    

class Direct_Forecasting_Belief(nn.Module):
    def __init__(
        self,
        latent_dim,
        condition_dim,
        seq_len,
        hidden_dim,
        num_layers=10,
        num_heads=4,
        attention_dropout=0.1,
        residual_dropout=0.1,
        hidden_dropout=0.1,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim

        self.condition_layer = nn.Linear(condition_dim, latent_dim, bias=False)
        self.reward_layer = nn.Linear(1, latent_dim, bias=False)

        self.hidden_emb = nn.Linear(latent_dim + latent_dim + latent_dim, hidden_dim, bias=False)
        self.timestep_emb = nn.Embedding(seq_len, hidden_dim)

        self.hidden_drop = nn.Dropout(hidden_dropout)
        self.hidden_norm = nn.LayerNorm(hidden_dim)
        self.out_norm = nn.LayerNorm(hidden_dim)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    seq_len=seq_len,
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    attention_dropout=attention_dropout,
                    residual_dropout=residual_dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.out_layer = nn.Linear(hidden_dim, latent_dim, bias=False)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, latents, actions, rewards, timesteps, masks):

        batch_size, seq_len = actions.shape[0], actions.shape[1]

        latents = latents.repeat(1, seq_len, 1)
        timesteps = timesteps.repeat(batch_size, 1)
        actions_emb = self.condition_layer(actions)
        rewards_emb = self.reward_layer(rewards)
        concat_latents = torch.concat((latents, actions_emb, rewards_emb), dim=-1)

        time_emb = self.timestep_emb(timesteps)

        z = self.hidden_emb(concat_latents) + time_emb
        z = self.hidden_norm(z)
        z = self.hidden_drop(z)

        for block in self.blocks:
            z = block(z, padding_mask=masks)
        z_ = self.out_norm(z)
        z_ = self.out_layer(z_)
        return z_


class Actor(nn.Module):
    def __init__(self, latent_dim, action_dim,
                action_high=1, action_low=-1,
                logstd_min=-5, logstd_max=2):
        super().__init__()
        self.actor_public = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.actor_mean = nn.Sequential(
            nn.Linear(256, action_dim),
        )
        self.actor_logstd = nn.Sequential(
            nn.Linear(256, action_dim),
            nn.Tanh(),
        )
        self.register_buffer(
            "action_scale", torch.tensor((action_high - action_low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((action_high + action_low) / 2.0, dtype=torch.float32)
        )
        self.logstd_min = logstd_min
        self.logstd_max = logstd_max

    def forward(self, x):
        public_x = self.actor_public(x)
        mean = self.actor_mean(public_x)
        logstd = self.actor_logstd(public_x)
        logstd = self.logstd_min + 0.5 * (self.logstd_max - self.logstd_min) * (logstd + 1)
        return mean, logstd

    def get_mean_std(self, x):
        mean, logstd = self(x)
        std = logstd.exp()
        return mean, std

    def get_action(self, x):
        mean, logstd = self(x)
        std = logstd.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

class Critic(nn.Module):
    def __init__(self, latent_dim, action_dim):
        super().__init__()

        self.critic = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x, a):
        x = torch.cat([x, a], -1)
        return self.critic(x)
