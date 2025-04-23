import torch
import torch.nn as nn
import torch.nn.functional as F


def energy_function(criterion: nn.Module,
                    predicted: torch.Tensor,
                    encoded: torch.Tensor,
                    lambda_d: float,
                    lambda_r: float,
                    beta_r: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    distance = criterion(predicted, encoded)
    std_p = torch.sqrt(predicted.var(dim=1, unbiased=False) + 1e-4)
    std_e = torch.sqrt(encoded.var(dim=1, unbiased=False) + 1e-4)

    reg_p = torch.mean(F.leaky_relu(1.0 - std_p)) ** beta_r
    reg_e = torch.mean(F.leaky_relu(1.0 - std_e)) ** beta_r
    reg = reg_p + reg_e

    energy = distance * lambda_d + lambda_r * reg

    return energy, distance, reg