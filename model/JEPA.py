import torch
import torch.nn as nn

from model.encoder import StateEncoder
from model.predictor import StatePredictor


class ExploreJEPA(nn.Module):
    def __init__(self, encoding_hidden_dim: int,
                 encoding_dim: int,
                 encoding_layers: int,
                 action_dim: int=2,
                 device: str="cuda") -> None:
        super().__init__()
        self.device = device
        self.repr_dim = encoding_dim
        self.init_state_encoder = StateEncoder(hidden_dim=encoding_hidden_dim,
                                               embedding_dim=encoding_dim,
                                               block_layers=encoding_layers)
        self.later_state_encoder = StateEncoder(hidden_dim=encoding_hidden_dim,
                                                embedding_dim=encoding_dim,
                                                block_layers=encoding_layers)
        self.state_predictor = StatePredictor(state_dim=encoding_dim,
                                              action_dim=action_dim)


    def forward(self, x: torch.Tensor,
                actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [b, t, 2, 65, 65]
        # actions: [b, t-1, 2]
        init_state = x[:, 0, :, :, :] # [b, 2, 65, 65]
        init_state_repr = self.init_state_encoder(init_state) # [b, d]

        predicted_state_repr = []
        cur_encoded_state_repr = []
        for i in range(actions.size(1)):
            prev_predicted_state_repr = init_state_repr if i == 0 else predicted_state_repr[i - 1]
            cur_action = actions[:, i, :] # [b, 2]
            # predict next state
            cur_input = torch.cat([prev_predicted_state_repr, cur_action], dim=1) # [b, d+2]

            cur_predicted_state_repr = self.state_predictor(cur_input)
            predicted_state_repr.append(cur_predicted_state_repr)

            # encode next state
            if self.training:
                cur_state = x[:, i + 1, :, :, :]  # [b, 2, 65, 65]
                cur_state_repr = self.later_state_encoder(cur_state) # [b, d]
                cur_encoded_state_repr.append(cur_state_repr)

        predicted = torch.stack(predicted_state_repr, dim=1) # [b, t-1, d]
        if not self.training:
            predicted = torch.cat([init_state_repr.unsqueeze(dim=1), predicted], dim=1) # [b, t, d]

        encoded = None
        if self.training:
            encoded = torch.stack(cur_encoded_state_repr, dim=1) # [b, t-1, d]

        return predicted, encoded
