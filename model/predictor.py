import torch
import torch.nn as nn


class StatePredictor(nn.Module):
    def __init__(self, state_dim: int,
                 action_dim: int) -> None:
        super().__init__()
        self.state_ln = nn.LayerNorm(state_dim)

        self.action_linear = nn.Linear(in_features=action_dim, out_features=state_dim)
        self.action_ln = nn.LayerNorm(state_dim)

        self.fusion_linear_1 = nn.Linear(in_features=state_dim, out_features=4 * state_dim)
        self.fusion_ln = nn.LayerNorm(4 * state_dim)
        self.fusion_linear_2 = nn.Linear(in_features=4 * state_dim, out_features=state_dim)

        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(p=0.1)


    def forward(self, states_repr: torch.Tensor,
                actions_repr: torch.Tensor) -> torch.Tensor:
        # states_repr: [b, d]
        # actions_repr: [b, 2]
        states_repr = self.state_ln(states_repr)

        actions_repr = self.action_linear(actions_repr) # [b, d]
        actions_repr = self.action_ln(actions_repr)

        repr = states_repr + actions_repr # [b, d]
        original = repr

        s = self.fusion_linear_1(repr) # [b, 4*d]
        s = self.gelu(s)
        s = self.fusion_ln(s)
        s = self.dropout(s)

        s = self.fusion_linear_2(s)  # [b, d]

        return original + s
