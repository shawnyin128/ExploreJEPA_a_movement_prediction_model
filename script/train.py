import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

from dataset import create_wall_dataloader
from model.JEPA import ExploreJEPA
from energy.energy import energy_function

def get_criterion():
    return nn.MSELoss()

def get_optimizer(model, learning_rate, weight_decay):
    return optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

def get_scheduler(optimizer):
    return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10000)

def training_loop(model, epoch, learning_rate, weight_decay, train_loader):
    criterion = get_criterion()
    optimizer = get_optimizer(model, learning_rate, weight_decay)
    scheduler = get_scheduler(optimizer)

    for i in range(epoch):
        pbar = tqdm(train_loader)
        for data in pbar:
            states = data.states
            actions = data.actions
            predicted_s, encoded_s = model(states, actions)

            energy, d, r = energy_function(criterion, predicted_s, encoded_s, lambda_d=0.5, lambda_r=0.5, beta_r=-1.0)

            optimizer.zero_grad()
            energy.backward()
            optimizer.step()
            scheduler.step()

            pbar.set_postfix({"total energy: ": energy.item(), "mse loss:": d.item(), "reg loss:": r.item()})

        ckpt_path = f"/scratch/xy2053/2025SP/2572_DeepLearning/codes/checkpoint/checkpoint_epoch_{i + 1}.pt"
        torch.save({
            'epoch': i + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }, ckpt_path)
    return model

if __name__ == "__main__":
    epoch = 10
    batch_size = 64

    learning_rate = 1e-4
    weight_decay = 1e-5

    train_dataloader = create_wall_dataloader(data_path="/scratch/DL25SP/train", batch_size=batch_size)

    model = ExploreJEPA(image_size=65,
                        encoding_hidden_dim=16,
                        encoding_dim=256,
                        encoding_layers=3)
    model.to("cuda")

    train_model = training_loop(model, epoch, learning_rate, weight_decay, train_dataloader)
