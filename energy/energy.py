import torch
import torch.nn as nn


def main_loss(criterion: nn.Module,
              predicted: torch.Tensor,
              encoded: torch.Tensor,
              lambda_d: float) -> torch.Tensor:
    distance = lambda_d * criterion(predicted, encoded)
    return distance


def variance_loss(predicted: torch.Tensor,
                  lambda_r: float,
                  target_std: float = 1.0) -> torch.Tensor:
    p = predicted.view(-1, predicted.shape[-1])  # [b*t, d]

    std = torch.sqrt(p.var(dim=0) + 1e-4)  # [d]
    hinge = torch.max(torch.zeros_like(std), target_std - std)

    loss = hinge.mean() * lambda_r
    return loss


def covariance_loss(predicted: torch.Tensor,
                    lambda_c: float) -> torch.Tensor:
    p = predicted.view(-1, predicted.shape[-1])  # [b*t, d]

    norm_p = p - p.mean(dim=0, keepdim=True)
    N, D = norm_p.shape
    cov_p = (norm_p.T @ norm_p) / (N - 1 + 1e-8)
    cov_p_val = ((cov_p ** 2).sum() - (cov_p.diag() ** 2).sum()) / D

    cov_loss = cov_p_val * lambda_c
    return cov_loss


def energy_function(criterion: nn.Module,
                    predicted: torch.Tensor,
                    encoded: torch.Tensor,
                    lambda_d: float,
                    lambda_r: float,
                    lambda_c: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    distance = main_loss(criterion, predicted, encoded, lambda_d)
    var_reg = variance_loss(predicted, lambda_r)
    cov_reg = covariance_loss(predicted, lambda_c)

    energy = distance + var_reg + cov_reg
    return energy, distance, var_reg, cov_reg