import torch
import torch.nn as nn

from model.blocks import ConvNextBlock


class StateEncoder(nn.Module):
    def __init__(self, hidden_dim: int,
                 embedding_dim: int,
                 block_layers: int) -> None:
        super().__init__()
        self.agent_pwConv = nn.Conv2d(in_channels=1, out_channels=hidden_dim, kernel_size=1, bias=False) # [b, 1, 65, 65] -> [b, h, 65, 65]
        self.agent_blocks = nn.Sequential(*[ConvNextBlock(hidden_dim) for _ in range(block_layers)]) # [b, h, 65, 65]
        self.env_pwConv = nn.Conv2d(in_channels=1, out_channels=hidden_dim, kernel_size=1, bias=False)
        self.env_blocks = nn.Sequential(*[ConvNextBlock(hidden_dim) for _ in range(block_layers)]) # [b, h, 65, 65]

        self.agent_fc = nn.Linear(in_features=hidden_dim * 6 * 6, out_features=embedding_dim)
        self.env_fc = nn.Linear(in_features=hidden_dim * 6 * 6, out_features=embedding_dim)

        self.agent_ln = nn.LayerNorm(embedding_dim)
        self.env_ln = nn.LayerNorm(embedding_dim)

        self.gap = nn.AdaptiveAvgPool2d(output_size=(6, 6))
        self.dropout = nn.Dropout(p=0.1)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [b, 2, 65, 65]
        agent_state = x[:, 0, :, :].unsqueeze(dim=1) # [b, 1, 65, 65]
        env_state = x[:, 1, :, :].unsqueeze(dim=1) # [b, 1, 65, 65]

        agent_rep = self.agent_pwConv(agent_state) # [b, h, 65, 65]
        agent_rep = self.agent_blocks(agent_rep) # [b, h, 65, 65]
        agent_rep = self.gap(agent_rep)
        agent_rep = agent_rep.view(agent_rep.size(0), -1) # [b, h*36]
        agent_rep = self.agent_fc(agent_rep) # [b, d]
        agent_rep = self.agent_ln(agent_rep)
        agent_rep = self.dropout(agent_rep)

        env_rep = self.env_pwConv(env_state)
        env_rep = self.env_blocks(env_rep)
        env_rep = self.gap(env_rep)
        env_rep = env_rep.view(env_rep.size(0), -1)
        env_rep = self.env_fc(env_rep)
        env_rep = self.env_ln(env_rep)
        env_rep = self.dropout(env_rep)

        rep = agent_rep + env_rep

        return rep