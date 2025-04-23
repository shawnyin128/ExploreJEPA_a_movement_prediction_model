import torch
import yaml
import os

from dataset import create_wall_dataloader
from model.JEPA import ExploreJEPA
from train import training_loop


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
    ckpt_path = "./ft_model_weights.pth"
    model.load_state_dict(torch.load(ckpt_path))
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
    ckpt_path = "./model_weights_ft.pth"
    torch.save(model.state_dict(), ckpt_path)