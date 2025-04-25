import torch
import torch.nn as nn


def main_loss(criterion: nn.Module,
              predicted: torch.Tensor,
              encoded: torch.Tensor,
              lambda_d: float) -> torch.Tensor:
    distance = lambda_d * criterion(predicted, encoded)
    return distance


def variance_loss(predicted: torch.Tensor,
                  encoded: torch.Tensor,
                  lambda_r: float) -> torch.Tensor:
    p = predicted.view(-1, predicted.shape[-1])
    e = encoded.view(-1, encoded.shape[-1])
    std_p = torch.mean(p.var(dim=0, unbiased=False) + 1e-4)
    std_e = torch.mean(e.var(dim=0, unbiased=False) + 1e-4)

    variance = lambda_r * (std_p ** -1.0 + std_e ** -1.0)
    return variance


def covariance_loss(predicted: torch.Tensor,
                    encoded: torch.Tensor,
                    lambda_c: float) -> torch.Tensor:
    p = predicted.view(-1, predicted.shape[-1])
    e = encoded.view(-1, encoded.shape[-1])

    norm_p = p - p.mean(dim=0, keepdim=True)
    norm_e = e - e.mean(dim=0, keepdim=True)

    N, D = norm_p.shape
    cov_p = (norm_p.T @ norm_p) / (N - 1)
    cov_p_val = (cov_p ** 2).sum() - (cov_p.diag() ** 2).sum()
    cov_e = (norm_e.T @ norm_e) / (N - 1)
    cov_e_val = (cov_e ** 2).sum() - (cov_e.diag() ** 2).sum()
    return lambda_c * (cov_p_val + cov_e_val)


def energy_function(criterion: nn.Module,
                    predicted: torch.Tensor,
                    encoded: torch.Tensor,
                    lambda_d: float,
                    lambda_r: float,
                    lambda_c: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    distance = main_loss(criterion, predicted, encoded, lambda_d)
    var_reg = variance_loss(predicted, encoded, lambda_r)
    cov_reg = covariance_loss(predicted, encoded, lambda_c)

    energy = distance + var_reg + cov_reg
    return energy, distance, var_reg, cov_reg