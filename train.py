import numpy as np
import math

import torch
import torch.nn as nn

from tqdm.auto import tqdm


# не люблю такие названия, но хотел сделать точь-в-точь как в статье, чтобы и мне было удобнее разбираться
def get_data(n=10_000, T=3, S=0.999):
    data = torch.rand(n, T)
    mask = data < S
    data[mask] = 0

    # нормализация
    data = data / torch.norm(data)
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


def train(data, lr=1e-3, num_steps=50_000, warmup_steps=2_500, device="cpu"):
    model.to(device)
    data = data.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), weight_decay=1e-2)
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (num_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    for step in tqdm(range(num_steps)):
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(data, outputs)
        loss.backward()
        optimizer.step()
        scheduler.step()

    T = data.shape[1]
    torch.save(model.state_dict(), f'models/model_T{T}.pt')




def run_experiment()