import torch
import yaml
import os
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

from dataset import create_wall_dataloader
from model.JEPA import ExploreJEPA
from energy.energy import energy_function


def get_criterion() -> nn.Module:
    return nn.MSELoss()


def get_optimizer(model: nn.Module,
                  learning_rate: float,
                  weight_decay: float) -> optim.Optimizer:
    return optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


def get_scheduler(optimizer: optim.Optimizer,
                  t_max: int) -> optim.lr_scheduler.CosineAnnealingLR:
    return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)


def training_loop(model: nn.Module,
                  train_loader: torch.utils.data.DataLoader,
                  epoch: int,
                  learning_rate: float,
                  weight_decay: float,
                  lambda_d: float,
                  lambda_r: float,
                  beta_r: float,
                  fine_tune: bool=False) -> None:
    criterion = get_criterion()
    optimizer = get_optimizer(model, learning_rate, weight_decay)
    scheduler = get_scheduler(optimizer, t_max=epoch * 2300)

    if fine_tune:
        best_energy = float("inf")
    for i in range(epoch):
        pbar = tqdm(train_loader)
        for data in pbar:
            states = data.states
            actions = data.actions
            predicted_s, encoded_s = model(states, actions)

            energy, d, r = energy_function(criterion, predicted_s, encoded_s, lambda_d=lambda_d, lambda_r=lambda_r, beta_r=beta_r)

            if fine_tune:
                if energy.item() < best_energy:
                    best_energy = energy.item()
                    ckpt_path = "./ft_result_model_weights.pth"
                    torch.save(model.state_dict(), ckpt_path)

            optimizer.zero_grad()
            energy.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            pbar.set_postfix({"total energy: ": energy.item(), "mse loss:": d.item(), "reg loss:": r.item()})

        if not fine_tune:
            ckpt_path = f"./checkpoint/checkpoint_weights_{i + 1}.pth"
            torch.save(model.state_dict(), ckpt_path)
    return


if __name__ == "__main__":
    with open(os.path.join(os.path.dirname(__file__), "config.yaml"), "r") as f:
        config = yaml.safe_load(f)

    epoch = config["train"]["epoch"]
    batch_size = config["train"]["batch_size"]
    train_dataloader = create_wall_dataloader(data_path="/scratch/DL25SP/train", batch_size=batch_size)

    image_size = config["model"]["image_size"]
    hidden_dim = config["model"]["hidden_dim"]
    encoding_dim = config["model"]["output_dim"]
    layers = config["model"]["layers"]
    model = ExploreJEPA(encoding_hidden_dim=hidden_dim,
                        encoding_dim=encoding_dim,
                        encoding_layers=layers)
    model.to("cuda")

    learning_rate = config["train"]["learning_rate"]
    weight_decay = config["train"]["weight_decay"]
    lambda_d = config["train"]["lambda_d"]
    lambda_r = config["train"]["lambda_r"]
    beta_r = config["train"]["beta_r"]
    training_loop(model=model,
                  train_loader=train_dataloader,
                  epoch=epoch,
                  learning_rate=learning_rate,
                  weight_decay=weight_decay,
                  lambda_d=lambda_d,
                  lambda_r=lambda_r,
                  beta_r=beta_r)
    ckpt_path = "./model_weights.pth"
    torch.save(model.state_dict(), ckpt_path)
