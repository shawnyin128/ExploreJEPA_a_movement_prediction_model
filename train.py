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
                  config: dict,
                  train_loader: torch.utils.data.DataLoader,
                  epoch: int,
                  learning_rate: float,
                  weight_decay: float,
                  fine_tune: bool=False) -> None:
    criterion = get_criterion()
    optimizer = get_optimizer(model, learning_rate, weight_decay)
    scheduler = get_scheduler(optimizer, t_max=epoch * 1150)

    cur_stage = 0
    total_stages = epoch // config["train"]["stage_epoch"]
    best_energy = float("inf")
    last_stage_var_loss = 0.9

    for i in range(epoch):
        if i != 0 and i % config["train"]["stage_epoch"] == 0:
            cur_stage += 1
            if cur_stage == total_stages - 1:
                print("Reaching last stage, switching to fine-tune mode...")
                print(f"Last stage variance loss (upper bound for saving model): {last_stage_var_loss}")
                fine_tune = True

        pbar = tqdm(train_loader)
        epoch_var_loss = 0.0
        iteration = 0

        if not fine_tune:
            lambda_d = config["energy"][f"lambda_d_{cur_stage}"]
            lambda_r = config["energy"][f"lambda_r_{cur_stage}"]
            lambda_c = config["energy"][f"lambda_c_{cur_stage}"]
        else:
            lambda_d = config["energy"]["lambda_d_ft"]
            lambda_r = config["energy"]["lambda_r_ft"]
            lambda_c = config["energy"]["lambda_c_ft"]
        print(f"stage: {cur_stage}, epoch: {i}, lambda_d: {lambda_d}, lambda_r: {lambda_r}, lambda_c: {lambda_c}, fine_tune: {fine_tune}")
        
        for data in pbar:
            states = data.states
            actions = data.actions
            predicted_s, encoded_s = model(states, actions)

            energy, dis, var, cov = energy_function(criterion, predicted_s, encoded_s, lambda_d=lambda_d, lambda_r=lambda_r, lambda_c=lambda_c)

            epoch_var_loss += var.item()
            iteration += 1

            if fine_tune:
                if energy.item() < best_energy and var.item() <= last_stage_var_loss:
                    best_energy = energy.item()
                    ckpt_path = "./model_weights.pth"
                    torch.save(model.state_dict(), ckpt_path)

            optimizer.zero_grad()
            energy.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)

            optimizer.step()
            scheduler.step()

            pbar.set_postfix({"total energy: ": energy.item(),
                              "mse loss:": dis.item(),
                              "var loss:": var.item(),
                              "cov loss:": cov.item()})

        ckpt_path = f"./checkpoint/checkpoint_weights_{(i + 1) % 10}.pth"
        torch.save(model.state_dict(), ckpt_path)

        if not fine_tune:
            last_stage_var_loss = epoch_var_loss / iteration


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
    if config["train"]["base_model"] or config["train"]["fine_tune"]:
        print("using base model")
        ckpt_path = "./base_model_weights.pth"
        model.load_state_dict(torch.load(ckpt_path))
    model.to("cuda")

    learning_rate = config["train"]["learning_rate"]
    weight_decay = config["train"]["weight_decay"]

    training_loop(model=model,
                  config=config,
                  train_loader=train_dataloader,
                  epoch=epoch,
                  learning_rate=learning_rate,
                  weight_decay=weight_decay,
                  fine_tune=config["train"]["fine_tune"])
