import torch
import torch.nn as nn

def energy_function(criterion: nn.Module,
                    predicted: torch.Tensor,
                    encoded: torch.Tensor,
                    lambda_d: float,
                    lambda_r: float,
                    beta_r: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    distance = criterion(predicted, encoded)
    var_p = torch.mean(predicted.var(dim=1, unbiased=False) + 1e-4)
    var_e = torch.mean(encoded.var(dim=1, unbiased=False) + 1e-4)

    reg = (var_p ** beta_r) + (var_e ** beta_r)

    energy = distance * lambda_d + lambda_r * reg

    return energy, distance, reg