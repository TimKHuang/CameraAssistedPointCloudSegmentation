import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class FeatureNet(nn.Module):

    def __init__(self, num_class=20) -> None:
        super().__init__()
        self.num_class = num_class

        self.layer1 = nn.Linear(256, 256)
        self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Linear(128, 64)
        self.layer4 = nn.Linear(64, 32)
        self.out = nn.Linear(32, num_class)

    def forward(self, features):

        f = features

        f = F.relu(self.layer1(f))
        f = F.relu(self.layer2(f))
        f = F.relu(self.layer3(f))
        f = F.relu(self.layer4(f))

        f = self.out(f)

        return f
