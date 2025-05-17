import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BernoulliDiffusionModel(nn.Module):
    def __init__(self, context_dim, hidden_dim=128, time_dim=32):
        super(BernoulliDiffusionModel, self).__init__()
        self.time_embedding = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )
        self.context_embedding = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.net = nn.Sequential(
            nn.Linear(1 + hidden_dim + time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, context, t):
        time_emb = self.time_embedding(t.float().unsqueeze(-1))
        context_emb = self.context_embedding(context)
        x_input = torch.cat([x, context_emb, time_emb], dim=1)
        return self.net(x_input)

class BernoulliDiffusionBandit:
    def __init__(self, context_dim, n_timesteps=100, noise_schedule='linear', hidden_dim=128, time_dim=32, lr=1e-4):
        self.context_dim = context_dim
        self.n_timesteps = n_timesteps
        self.model = BernoulliDiffusionModel(context_dim, hidden_dim, time_dim).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        if noise_schedule == 'linear':
            self.betas = torch.linspace(0.001, 0.499, n_timesteps).to(device)
        else:
            self.betas = 0.25 * (1 - torch.cos(torch.linspace(0, np.pi, n_timesteps))).to(device)
        self.context_list = []
        self.reward_list = []

    def q_sample(self, x_0, t):
        batch_size = x_0.shape[0]
        random_flips = torch.rand(batch_size, 1).to(device)
        flip_probs = self.betas[t].view(-1, 1)
        flip_mask = (random_flips < flip_probs).float()
        x_t = (1 - x_0) * flip_mask + x_0 * (1 - flip_mask)
        return x_t, flip_mask

    def update(self, context, reward):
        binary_reward = 1.0 if reward > 0.5 else 0.0
        self.context_list.append(torch.tensor(context, dtype=torch.float32))
        self.reward_list.append(torch.tensor([binary_reward], dtype=torch.float32))

    def train(self, batch_size=32, epochs=1):
        if len(self.context_list) < batch_size:
            return 0.0
        total_loss = 0.0
        for _ in range(epochs):
            indices = np.random.choice(len(self.context_list), min(batch_size, len(self.context_list)), replace=False)
            context_batch = torch.stack([self.context_list[i] for i in indices]).to(device)
            reward_batch = torch.stack([self.reward_list[i] for i in indices]).to(device)
            t = torch.randint(0, self.n_timesteps, (min(batch_size, len(self.context_list)),), device=device)
            x_t, flip_mask = self.q_sample(reward_batch, t)
            pred_logits = self.model(x_t, context_batch, t)
            loss = F.binary_cross_entropy_with_logits(pred_logits, reward_batch)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / epochs

    def p_sample(self, context, t):
        context_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0).to(device)
        x = torch.bernoulli(torch.ones(1, 1) * 0.5).to(device)
        for time_step in reversed(range(t)):
            time_tensor = torch.tensor([time_step], device=device)
            pred_logits = self.model(x, context_tensor, time_tensor)
            probs = torch.sigmoid(pred_logits)
            x = torch.bernoulli(probs).to(device) if time_step > 0 else (probs > 0.5).float()
        return x.cpu().detach().numpy()[0, 0]

    def p_sample_likelihood(self, context, t, n_estimation_samples):
        context_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0).to(device)
        probs_list = []
        for _ in range(n_estimation_samples):
            x = torch.bernoulli(torch.ones(1, 1) * 0.5).to(device)
            for time_step in reversed(range(t)):
                time_tensor = torch.tensor([time_step], device=device)
                pred_logits = self.model(x, context_tensor, time_tensor)
                probs = torch.sigmoid(pred_logits)
                if time_step > 0:
                    x = torch.bernoulli(probs).to(device)
                else:
                    probs_list.append(probs.item())
        return np.mean(probs_list)

    def select(self, contexts, beta=1.0, n_samples=10):
        probs_reward_1 = []
        for context in contexts:
            prob = self.p_sample_likelihood(context, self.n_timesteps, n_samples)
            probs_reward_1.append(prob)
        arm = np.argmax(probs_reward_1)
        return arm, probs_reward_1

    def predict_reward_probability(self, context):
        with torch.no_grad():
            x = torch.ones(1, 1).to(device) * 0.5
            time_tensor = torch.tensor([0], device=device)
            pred_logits = self.model(x, context, time_tensor)
            prob = torch.sigmoid(pred_logits).item()
            return prob


