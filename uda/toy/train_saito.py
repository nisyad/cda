from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim

from datasets import mnist, mnist_m
from models.ganin import GaninModel
from trainer import train_ganin, test_ganin
from utils import transform, helper

# Random Seed
helper.set_random_seed(seed=123)

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Hyperparameters
config = dict(epochs=15,
              batch_size=64,
              learning_rate=2e-4,
              classes=10,
              img_size=28,
              experiment='minst-minist_m')


def main():

    model = GaninModel().to(device)

    # transforms
    transform_m = transform.get_transform(dataset="mnist")
    transform_mm = transform.get_transform(dataset="mnist_m")

    # dataloaders
    loaders_args = dict(
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=1,
        pin_memory=True,
    )

    trainloader_m = mnist.fetch(data_dir="data/mnist/processed/train.pt",
                                transform=transform_m,
                                **loaders_args)

    # fetching testloader_m for symmetry but it is not needed in the code
    testloader_m = mnist.fetch(data_dir="data/mnist/processed/test.pt",
                               transform=transform_m,
                               **loaders_args)

    trainloader_mm = mnist_m.fetch(data_dir="data/mnist_m/processed/train.pt",
                                   transform=transform_mm,
                                   **loaders_args)

    testloader_mm = mnist_m.fetch(data_dir="data/mnist_m/processed/test.pt",
                                  transform=transform_mm,
                                  **loaders_args)

    start_time = datetime.now()
    for epoch in range(config["epochs"]):

        train_ganin(model, epoch, config, trainloader_m, trainloader_mm,
                    testloader_mm, device)

        test_ganin(model, testloader_mm, device)

    end_time = datetime.now()
    print(f"Train Time for {config['epochs']} epochs: {end_time - start_time}")

    return model


if __name__ == "__main__":
    model = main()
