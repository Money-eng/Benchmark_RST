import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from .dataset import RSASeg2DDataset
from RSA_deep_working.Data_loader.class_data_loaders import DirectoryRSAClass

# Verrouillage des seeds et de l’environnement
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def worker_init_fn(worker_id):
    # Assure que chaque worker a une seed différente mais déterministe
    worker_seed = SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def create_dataloader(base_directory: str,
                      img_transform: transforms.Compose,
                      mask_transform_image: transforms.Compose,
                      mask_transform_series: transforms.Compose,
                      batch_size: int = 32,
                      shuffle: bool = True,
                      num_workers: int = 8):
    
    dir_loader = DirectoryRSAClass(base_directory, load_date_map=True, lazy=True)
    rsa_dataset_image = RSASeg2DDataset(
        dir_loader,
        mode='image',
        img_transform=img_transform,
        mask_transform_image=mask_transform_image,
        image_with_mtg=True
    )

    print("Nombre d'échantillons :", len(rsa_dataset_image), "images\n")

    # Split avec generator fixe
    generator = torch.Generator().manual_seed(SEED)
    lengths = [
        int(len(rsa_dataset_image) * 0.7),
        int(len(rsa_dataset_image) * 0.2),
        len(rsa_dataset_image) - int(len(rsa_dataset_image) * 0.7) - int(len(rsa_dataset_image) * 0.2)
    ]
    train_set, val_set, test_set = torch.utils.data.random_split(
        rsa_dataset_image,
        lengths,
        generator=generator
    )

    print("Train :", len(train_set), "  Val :", len(val_set), "  Test :", len(test_set))

    # DataLoaders avec generator et worker_init_fn
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        generator=generator
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        generator=generator
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        generator=generator
    )

    return train_loader, val_loader, test_loader
