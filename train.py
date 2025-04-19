import numpy as np
import math

import torch
import torch.nn as nn

from tqdm.auto import tqdm
from vis_utils import plot_model_features, plot_hidden_states
from collections import defaultdict
import fire

# не люблю такие названия, но хотел сделать точь-в-точь как в статье, чтобы и мне было удобнее разбираться
def get_data(T=3, n=10_000, S=0.999):
    data = torch.rand(n, T)
    mask = data < S
    data[mask] = 0

    # нормализация
    data = data / torch.norm(data, dim=0, keepdim=True)
    return data


def criterion(x, x_hat):
    return torch.sum((x - x_hat)**2) / x.shape[1]


class ToyModel(nn.Module):
    def __init__(self, n=10_000, m=2):
        super().__init__()
        self.w = nn.Parameter(torch.empty(m, n))
        nn.init.xavier_uniform_(self.w)
        self.b = nn.Parameter(torch.zeros(n, 1))
        
        self.relu = nn.ReLU()

    def forward(self, x, return_hidden_state=False):
        # m n, n T -> m T
        x = self.w @ x
        if return_hidden_state:
            return x
        # n m, m T -> n T
        x = self.w.T @ x + self.b
        x = self.relu(x)
        return x
    

def train(model, data, lr=1e-3, num_steps=50_000, warmup_steps=2_500, device="cpu"):
    model.to(device)
    data = data.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), weight_decay=1e-2, lr=lr)
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (num_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    T = data.shape[1]
    for _ in tqdm(range(num_steps), leave=False, desc=f"running for T={T}"):
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(data, outputs)
        loss.backward()
        optimizer.step()
        scheduler.step()

    
    torch.save(model.state_dict(), f'models/model_T{T}.pt')
    return model


def run_experiment(T, seed=69):
    torch.manual_seed(seed)
    
    n = 10_000
    train_data = get_data(T=T, n=n)

    test_data = get_data(T=1000, n=n)

    model = ToyModel(n=n)
    device = "cuda"
    train(model, train_data, device=device)

    train_data = train_data.to(device)
    test_data = test_data.to(device)

    model_features = plot_model_features(model, T)
    hidden_states = plot_hidden_states(train_data, model, T)

    with torch.no_grad():
        x_hat = model(test_data)
        loss = criterion(test_data, x_hat)
        print(f'test loss: {loss.item():.6f}')

    return model_features, hidden_states, loss.item()


if __name__ == "__main__":
    fire.Fire(run_experiment)