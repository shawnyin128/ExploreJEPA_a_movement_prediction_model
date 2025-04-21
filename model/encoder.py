import torch
import torch.nn as nn

from model.blocks import ConvNextBlock

class StateEncoder(nn.Module):
    def __init__(self, input_size: int,
                 hidden_dim: int,
                 embedding_dim: int,
                 block_layers: int) -> None:
        super().__init__()
        self.agent_pwConv = nn.Conv2d(in_channels=1, out_channels=hidden_dim, kernel_size=1, bias=False) # [b, 1, 65, 65] -> [b, 16, 65, 65]
        self.agent_blocks = nn.Sequential(*[ConvNextBlock(hidden_dim) for _ in range(block_layers)]) # [b, 16, 65, 65]

        self.env_pwConv = nn.Conv2d(in_channels=1, out_channels=hidden_dim, kernel_size=1, bias=False)
        self.env_blocks = nn.Sequential(*[ConvNextBlock(hidden_dim) for _ in range(block_layers)]) # [b, 16, 65, 65]

        self.agent_fc = nn.Linear(in_features=hidden_dim * input_size * input_size, out_features=embedding_dim) # [b, 16, 65, 65] -> [b, 256]
        self.env_fc = nn.Linear(in_features=hidden_dim * input_size * input_size, out_features=embedding_dim)

        self.agent_dropout = nn.Dropout(p=0.1)
        self.env_dropout = nn.Dropout(p=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [b, 2, 65, 65]
        agent_state = x[:, 0, :, :].unsqueeze(dim=1) # [b, 1, 65, 65]
        env_state = x[:, 1, :, :].unsqueeze(dim=1) # [b, 1, 65, 65]

        agent_rep = self.agent_pwConv(agent_state) # [b, 16, 65, 65]
        agent_rep = self.agent_blocks(agent_rep) # [b, 16, 65, 65]
        agent_rep = agent_rep.flatten(start_dim=1) # [b, 270400]
        agent_rep = self.agent_dropout(agent_rep)
        agent_rep = self.agent_fc(agent_rep) # [b, 256]

        env_rep = self.env_pwConv(env_state) # [b, 16, 65, 65]
        env_rep = self.env_blocks(env_rep) # [b, 16, 65, 65]
        env_rep = env_rep.flatten(start_dim=1) # [b, 270400]
        env_rep = self.env_dropout(env_rep)
        env_rep = self.env_fc(env_rep) # [b, 256]

        rep = agent_rep+env_rep # [b, 256]

        return rep