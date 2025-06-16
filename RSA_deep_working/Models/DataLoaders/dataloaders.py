import torch
import os
from random import Random
from torch.utils.data import DataLoader, Subset, Sampler
from torchvision import transforms
from utils.logger import log_dataset_stats
from utils.misc import set_seed, worker_init_fn

from .dataset import RSADataset
from .directory_RSA_class import DirectoryRSAClass

SEED = 42
set_seed(SEED)  # Call only once at script startup!


class SeriesBatchSampler(Sampler):
    """
    Custom sampler to group samples by series.
    """

    def __init__(self, list_of_series_idx_lists, shuffle=False, seed=42):
        self.series = list_of_series_idx_lists
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        order = list(range(len(self.series)))
        if self.shuffle:
            rng = Random(self.seed)
            rng.shuffle(order)
        for i in order:
            yield self.series[i]

    def __len__(self):
        return len(self.series)


def create_dataloader(
    base_directory: str,
    img_transforms: list,
    default_batch_size: int = 32,
    num_workers: int = 16,
    seed: int = 42,
):
    # Load datasets
    dir_loader = DirectoryRSAClass(base_directory, load_date_map=True, lazy=True)

    series_dataset = RSADataset(
        dir_loader, mode="series", img_transform=img_transforms[0], image_with_mtg=True
    )

    image_dataset_1 = RSADataset(
        dir_loader, mode="image", img_transform=img_transforms[0], image_with_mtg=True
    )

    image_dataset_2 = RSADataset(
        dir_loader, mode="image", img_transform=img_transforms[1], image_with_mtg=True
    )

    image_dataset_3 = RSADataset(
        dir_loader, mode="image", img_transform=img_transforms[2], image_with_mtg=True
    )
    
    image_dataset_val_test = RSADataset(
        dir_loader, mode="image", img_transform=img_transforms[3], image_with_mtg=True
    )

    n_series = len(series_dataset)
    n_images = len(image_dataset_1)

    # Split series indices into train/val/test
    generator = torch.Generator().manual_seed(seed)
    n_train = int(0.7 * n_series)
    n_val = int(0.2 * n_series)
    n_test = n_series - n_train - n_val

    # Group image indices by series (using mtg_path as series identifier)
    series = {}

    for i in range(n_images):
        mtg_path = image_dataset_1.samples[i]["mtg_path"]
        if mtg_path not in series:
            series[mtg_path] = [i, i, 1]
        else:
            series[mtg_path][1] = i
            series[mtg_path][2] += 1

    series = list(series.items())
    series_split = torch.utils.data.random_split(
        series, [n_train, n_val, n_test], generator=generator
    )
    series_train, series_val, series_test = series_split

    # Create image indices for each split
    def get_indices(split):
        indices, per_series = [], []
        for _, (start, end, _) in split:
            idx = list(range(start, end + 1))
            indices += idx
            per_series.append(idx)
        return indices, per_series

    train_indices, _ = get_indices(series_train)
    val_indices, val_indices_per_series = get_indices(series_val)
    test_indices, test_indices_per_series = get_indices(series_test)

    # Log stats
    log_dataset_stats(
        n_series,
        n_images,
        len(series_train),
        len(series_val),
        len(series_test),
        len(train_indices),
        len(val_indices),
        len(test_indices),
    )
    print(
        (
            f"Number of transformed images: "
            f"{len(image_dataset_1)} (dataset 1), "
            f"{len(image_dataset_2)} (dataset 2), "
            f"{len(image_dataset_3)} (dataset 3)"
        )
    )

    # Datasets and samplers
    train_dataset_1 = Subset(image_dataset_1, train_indices)
    train_dataset_2 = Subset(image_dataset_2, train_indices)
    train_dataset_3 = Subset(image_dataset_3, train_indices)
    train_dataset = torch.utils.data.ConcatDataset(
        [train_dataset_1, train_dataset_2, train_dataset_3]
    )
    val_dataset = Subset(image_dataset_val_test, val_indices)
    test_dataset = Subset(image_dataset_val_test, test_indices)

    val_batch_sampler = SeriesBatchSampler(val_indices_per_series, shuffle=False)
    test_batch_sampler = SeriesBatchSampler(test_indices_per_series, shuffle=False)

    # Dataloaders
    train_loader = DataLoader(  # only for training images - not series
        train_dataset,
        batch_size=default_batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=lambda wid: worker_init_fn(wid, base_seed=seed),
        pin_memory=True,
    )
    val_loader = DataLoader(  # only for validation images - not series
        val_dataset,
        batch_size=default_batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=lambda wid: worker_init_fn(wid, base_seed=seed),
        pin_memory=True,
    )
    test_loader = DataLoader(  # only for test images - not series
        test_dataset,
        batch_size=default_batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=lambda wid: worker_init_fn(wid, base_seed=seed),
        pin_memory=True,
    )
    val_loader_series = DataLoader(  # work with series
        image_dataset_1,
        batch_sampler=val_batch_sampler,
        num_workers=num_workers,
        worker_init_fn=lambda wid: worker_init_fn(wid, base_seed=seed),
        pin_memory=True,
    )
    test_loader_series = DataLoader(  # work with series
        image_dataset_1,
        batch_sampler=test_batch_sampler,
        num_workers=num_workers,
        worker_init_fn=lambda wid: worker_init_fn(wid, base_seed=seed),
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, val_loader_series, test_loader_series
