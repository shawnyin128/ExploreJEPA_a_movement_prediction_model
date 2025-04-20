import torch
import torch.nn as nn

from model.encoder import StateEncoder
from model.predictor import StatePredictor

class ExploreJEPA(nn.Module):
    def __init__(self, image_size: int,
                 encoding_hidden_dim: int,
                 encoding_dim: int,
                 encoding_layers: int,
                 action_dim: int=2,
                 velocity_dim: int=1,
                 device: str="cuda") -> None:
        super().__init__()
        self.device = device
        self.repr_dim = encoding_dim
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
                actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [b, 1, 2, 65, 65]
        # actions: [b, 16, 2]
        init_state = x[:, 0, :, :, :] # [b, 2, 65, 65]
        init_state_repr = self.init_state_encoder(init_state) # [b, 256]

        predicted_state_repr = []
        cur_encoded_state_repr = []
        for i in range(x.shape[1] - 1):
            prev_predicted_state_repr = init_state_repr if i == 0 else predicted_state_repr[i - 1]
            cur_state = x[:, i + 1, :, :, :] # [b, 2, 65, 65]
            cur_action = actions[:, i, :] # [b, 2]
            # predict next state
            if i == 0:
                cur_velocity = torch.zeros(cur_action.shape[0], 1).to(cur_action.device)
            elif i == 1:
                cur_velocity = torch.norm(prev_predicted_state_repr - init_state_repr, dim=-1, keepdim=True)
            else:
                cur_velocity = torch.norm(prev_predicted_state_repr - predicted_state_repr[i - 2], dim=-1, keepdim=True)
            cur_input = torch.cat([prev_predicted_state_repr, cur_action, cur_velocity], dim=1) # [b, 259]

            cur_predicted_state_repr = self.state_predictor(cur_input)
            predicted_state_repr.append(cur_predicted_state_repr)

            # encode next state
            cur_state_repr = self.later_state_encoder(cur_state) # [b, 256]
            cur_encoded_state_repr.append(cur_state_repr)

        predicted = torch.stack(predicted_state_repr, dim=1) # [b, 16, 256]
        encoded = torch.stack(cur_encoded_state_repr, dim=1) # [b, 16, 256]

        return predicted, encoded