import torch
import torch.nn as nn


class StatePredictor(nn.Module):
    def __init__(self, state_dim: int,
                 action_dim: int) -> None:
        super().__init__()
        self.in_dim = state_dim + action_dim
        self.hidden_dim = 4 * state_dim

        self.linear1 = nn.Linear(in_features=self.in_dim, out_features=self.hidden_dim)
        self.bn1 = nn.BatchNorm1d(self.hidden_dim)

        self.linear2 = nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim)
        self.bn2 = nn.BatchNorm1d(self.hidden_dim)

        self.linear3 = nn.Linear(in_features=self.hidden_dim, out_features=state_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [b, 258]
        original = x[:, :256]  # original: [b, 256]

        s = self.linear1(x)
        s = self.relu(s)
        s = self.bn1(s)
        s = self.dropout(s)

        s = self.linear2(s)
        s = self.relu(s)
        s = self.bn2(s)

        s = self.linear3(s)  # [b, 256]

        return original + s
