import torch

def energy_function(criterion, predicted, encoded, lambda_d, lambda_r, beta_r):
    distance = criterion(predicted, encoded)
    var_p = torch.mean(predicted.var(dim=1, unbiased=False) + 1e-4)
    var_e = torch.mean(encoded.var(dim=1, unbiased=False) + 1e-4)

    reg = (var_p ** beta_r) + (var_e ** beta_r)

    energy = distance * lambda_d + lambda_r * reg

    return energy, distance, reg