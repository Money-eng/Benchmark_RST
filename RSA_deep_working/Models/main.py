# main.py
import yaml
from Model import get_model
from Losses import get_loss
from Training.trainer import Trainer
from DataLoaders.dataloaders import create_dataloader
from torchvision import transforms

if __name__ == "__main__":
    with open("RSA_deep_working/Models/config.yml", "r") as f:
        config = yaml.safe_load(f)

    img_transform = transforms.Compose([
        transforms.ToTensor(), 
    ])
    mask_transform_image = transforms.Compose([
        transforms.ToTensor(),
    ])
    mask_transform_series = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_loader, val_loader, test_loader, series_val_loader, series_test_loader = create_dataloader(
        base_directory=config["data"]["base_dir"],
        img_transform=img_transform,
        mask_transform_image=mask_transform_image,
        mask_transform_series=mask_transform_series
    ) # 1 Gb of RAM

    model = get_model(config["model"])
    criterion = get_loss(config["loss"])

