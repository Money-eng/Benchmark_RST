import torch
from torch.utils.data import DataLoader
from utils.logger import log_dataset_stats
from utils.misc import set_seed, SEED, worker_init_fn

from .dataset import RSADataset
from .directory_RSA_class import DirectoryRSAClass

set_seed(SEED)
DEFAULT_BATCH_SIZE = 29


def create_dataloader(
        base_directory: str,
        img_transforms: list,
        batch_size: int = 32,
        num_workers: int = 4,
        generator: torch.Generator = None,
):
    # Load datasets
    train_dir = base_directory + "/Train"
    val_dir = base_directory + "/Val"
    test_dir = base_directory + "/Test"

    train_base = DirectoryRSAClass(
        train_dir, load_date_map=True, lazy=True)
    val_base = DirectoryRSAClass(
        val_dir, load_date_map=True, lazy=True)
    test_base = DirectoryRSAClass(
        test_dir, load_date_map=True, lazy=True)

    train_dataset_0 = RSADataset(
        rsa_dir_loader=train_base,
        mode="image",
        img_transform=img_transforms[0],
        image_with_mtg=True,
        as_RGB=False,
    )
    train_dataset_1 = RSADataset(
        rsa_dir_loader=train_base,
        mode="image",
        img_transform=img_transforms[1],
        image_with_mtg=True,
        as_RGB=False,
    )
    train_dataset_2 = RSADataset(
        rsa_dir_loader=train_base,
        mode="image",
        img_transform=img_transforms[2],
        image_with_mtg=True,
        as_RGB=False,
    )

    val_dataset = RSADataset(
        rsa_dir_loader=val_base,
        mode="image",
        img_transform=img_transforms[3],
        image_with_mtg=True,
        as_RGB=False,
    )
    test_dataset = RSADataset(
        rsa_dir_loader=test_base,
        mode="image",
        img_transform=img_transforms[3],
        image_with_mtg=True,
        as_RGB=False,
    )

    # Create dataloaders
    g = generator if generator is not None else torch.Generator()

    train_dataset = torch.utils.data.ConcatDataset(
        [train_dataset_0, train_dataset_1, train_dataset_2]
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
        pin_memory=True,
        generator=g,
    )

    # img, mask, time, mtg = next(iter(train_loader))
    # print(f"Image shape: {img.shape}, Mask shape: {mask.shape}, Time: {time}, MTG: {mtg}")
    # print(f"Image dtype: {img.dtype}, Mask dtype: {mask.dtype}, Time dtype: {type(time)}, MTG type: {type(mtg)}")
    # print(f"Image min: {img.min()}, max: {img.max()}, Mask min: {mask.min()}, max: {mask.max()}")

    global DEFAULT_BATCH_SIZE
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,  # DEFAULT_BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
        pin_memory=True,
        generator=g,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,  # DEFAULT_BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
        pin_memory=True,
        generator=g,
    )

    return (
        train_loader,
        val_loader,
        test_loader
    )
