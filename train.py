import torch
from torch import nn
from time import sleep

if __name__ == "__main__":
    
    net = nn.Sequential(
        nn.Conv2d(3, 64, 3, padding = 'same'),
        nn.ReLU(),
        nn.Conv2d(64, 64, 3, padding = 'same'),
        nn.ReLU(),
        nn.Conv2d(64, 3, 3, padding = 'same')
    ).cuda()
    
    cnt = 0
    while True:
        data = torch.normal(0, 1, (64, 3, 400, 400)).cuda()
        fwd = net(data)
        print(f"Forwarding: {cnt}")
        cnt += 1
        sleep(0.1)