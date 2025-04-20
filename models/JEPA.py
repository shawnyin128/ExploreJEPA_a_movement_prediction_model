import torch
import torch.nn as nn

from encoder import StateEncoder
from predictor import StatePredictor

class ExploreJEPA(nn.Module):
    def __init__(self, trajectory_length: int,
                 image_size: int,
                 encoding_hidden_dim: int,
                 encoding_dim: int,
                 encoding_layers: int,
                 action_dim: int = 2,
                 velocity_dim: int = 1) -> None:
        super().__init__()
        self.trajectory_length = trajectory_length
        self.init_state_encoder = StateEncoder(input_size=image_size,
                                               hidden_dim=encoding_hidden_dim,
                                               embedding_dim=encoding_dim,
                                               block_layers=encoding_layers)
        self.later_state_encoder = StateEncoder(input_size=image_size,
                                                hidden_dim=encoding_hidden_dim,
                                                embedding_dim=encoding_dim,
                                                block_layers=encoding_layers)
        self.state_predictor = StatePredictor(state_dim=encoding_dim,
                                              action_dim=action_dim,
                                              velocity_dim=velocity_dim)

    def forward(self, x: torch.Tensor,
                actions: torch.Tensor) -> list[torch.Tensor]:
        # x: [b, 1, 2, 65, 65]
        # actions: [b, 16, 2]
        init_state = x[:, 0, :, :, :] # [b, 2, 65, 65]
        init_state_rep = self.init_state_encoder(init_state) # [b, 256]

        predicted_state_rep = []
        for i in range(self.trajectory_length):
            cur_action = actions[:, i, :] # [b, 2]
            if i == 0:
                cur_velocity = torch.zeros(cur_action.shape[0], 1).to(cur_action.device)
                cur_state_rep = init_state_rep
            elif i == 1:
                cur_velocity = torch.norm(predicted_state_rep[i - 1] - init_state_rep, dim=-1, keepdim=True)
                cur_state_rep = predicted_state_rep[i - 1]
            else:
                cur_velocity = torch.norm(predicted_state_rep[i - 1] - predicted_state_rep[i - 2], dim=-1, keepdim=True)
                cur_state_rep = predicted_state_rep[i - 1]
            cur_input = torch.cat([cur_state_rep, cur_action, cur_velocity], dim=1) # [b, 259]

            cur_predicted_state_rep = self.state_predictor(cur_input)
            predicted_state_rep.append(cur_predicted_state_rep)
        return predicted_state_rep