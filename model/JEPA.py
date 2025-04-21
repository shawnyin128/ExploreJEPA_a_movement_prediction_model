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
                                              action_dim=action_dim)

    def forward(self, x: torch.Tensor,
                actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [b, 17, 2, 65, 65]
        # actions: [b, 16, 2]
        init_state = x[:, 0, :, :, :] # [b, 2, 65, 65]
        init_state_repr = self.init_state_encoder(init_state) # [b, 256]

        predicted_state_repr = []
        cur_encoded_state_repr = []
        for i in range(x.shape[1] - 1):
            prev_predicted_state_repr = init_state_repr if i == 0 else predicted_state_repr[i - 1]
            cur_action = actions[:, i, :] # [b, 2]
            # predict next state
            cur_input = torch.cat([prev_predicted_state_repr, cur_action], dim=1) # [b, 258]

            cur_predicted_state_repr = self.state_predictor(cur_input)
            predicted_state_repr.append(cur_predicted_state_repr)

            # encode next state
            if self.training:
                cur_state = x[:, i + 1, :, :, :]  # [b, 2, 65, 65]
                cur_state_repr = self.later_state_encoder(cur_state) # [b, 256]
                cur_encoded_state_repr.append(cur_state_repr)

        predicted = torch.stack(predicted_state_repr, dim=1) # [b, 16, 256]
        encoded = None
        if self.training:
            encoded = torch.stack(cur_encoded_state_repr, dim=1) # [b, 16, 256]

        return predicted, encoded
