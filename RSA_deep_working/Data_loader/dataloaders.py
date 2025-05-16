# DANGER 
## We are using a random seed to split the dataset into train, val and test sets. 
## AND we make sure to have full time series in the train, val and test sets respectively. -> Dangerous
import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, Sampler
from torchvision import transforms
from .dataset import RSADataset
from .directory_RSA_class import DirectoryRSAClass
import rsml

# Verrouillage des seeds et de l’environnement
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class SeriesBatchSampler(Sampler[list[int]]):
    def __init__(self, list_of_series_idx_lists, shuffle=False, seed=42):
        """
        list_of_series_idx_lists: ex. [[0,1,2],[3,4,5,6], ...]
        """
        self.series = list_of_series_idx_lists
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        order = list(range(len(self.series)))
        if self.shuffle:
            rng = random.Random(self.seed)
            rng.shuffle(order)
        for i in order:
            yield self.series[i]           # liste d'indices pour un batch

    def __len__(self):
        return len(self.series)

def worker_init_fn(worker_id): # for reproducibility
    worker_seed = SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def create_dataloader(base_directory: str,
                      img_transform: transforms.Compose,
                      mask_transform_image: transforms.Compose,
                      mask_transform_series: transforms.Compose,
                      default_batch_size: int = 32,
                      num_workers: int = 8):
    """
    Creates and returns PyTorch DataLoaders for training, validation, and testing datasets
    from a given base directory. The function splits the dataset into train, validation,
    and test sets, applies transformations, and configures DataLoader parameters.
    Args:
        base_directory (str): The base directory containing the dataset.
        img_transform (transforms.Compose): Transformations to apply to the input images.
        mask_transform_image (transforms.Compose): Transformations to apply to the image masks.
        mask_transform_series (transforms.Compose): Transformations to apply to the series masks.
        batch_size (int, optional): The number of samples per batch. Defaults to 32.
        shuffle (bool, optional): Whether to shuffle the training dataset. Defaults to True.
        num_workers (int, optional): The number of subprocesses to use for data loading. Defaults to 8.
    Returns:
        tuple: A tuple containing three DataLoader objects:
            - train_loader: DataLoader for the training dataset.
            - val_loader: DataLoader for the validation dataset.
            - test_loader: DataLoader for the testing dataset.
    """
    
    # --- Chargement des datasets ---
    dir_loader = DirectoryRSAClass(base_directory, load_date_map=True, lazy=True)
    series_dataset = RSADataset(
        dir_loader,
        mode='series',
        img_transform=img_transform,
        mask_transform_series=mask_transform_series,
        image_with_mtg=True
    )
    
    image_dataset = RSADataset(
        dir_loader,
        mode='image',
        img_transform=img_transform,
        mask_transform_image=mask_transform_image,
        image_with_mtg=True
    )

    # --- Affichage total ---
    print(f"Nombre de séries : {len(series_dataset)}")
    print(f"Nombre d'images (toutes séries confondues) : {len(image_dataset)}\n")

    # --- Split 70/20/10 au niveau séries ---
    generator = torch.Generator().manual_seed(SEED)
    n = len(series_dataset)
    n_train = int(0.7 * n)
    n_val   = int(0.2 * n)
    n_test  = n - n_train - n_val
    
    # So we want n_train number of series in the training set, n_val number of series in the validation set and n_test number of series in the test set
    ## and we want to generate an image dataset with the corresponding number of seires in the training, validation and test sets
    
    # Identify all the series in the image dataset (all the images in the serie have the dame mtg_path)
    series = {} # mtg_path -> [index of first image, index of last image]
    for i in range(len(image_dataset)):
        mtg_path = image_dataset.samples[i][3]
        if mtg_path not in series:
            series[mtg_path] = [i, i, 1]
        else:
            series[mtg_path][1] = i
            series[mtg_path][2] += 1
    
    # split randomly (with seed) the series into train, val and test sets
    series = list(series.items())
    series_split = torch.utils.data.random_split(series, [n_train, n_val, n_test], generator=generator)
    series_train = series_split[0]
    series_val = series_split[1]
    series_test = series_split[2]
    
    print(f"Number of Training series : {len(series_train)}")
    print(f"Number of Validation series : {len(series_val)}")
    print(f"Number of Testing series : {len(series_test)}\n")
    
    # print all the series in the train, val and test sets
    #print("Training series :")
    #for i in range(len(series_train)):
     #   print(f"{i+1} : {series_train[i][0]} with {series_train[i][1][2]} images at indices {series_train[i][1][0]} to {series_train[i][1][1]}")
    #print("\nValidation series :")
    #for i in range(len(series_val)):
     #   print(f"{i+1} : {series_val[i][0]} with {series_val[i][1][2]} images")
    #print("\nTesting series :")
    #for i in range(len(series_test)):
     #   print(f"{i+1} : {series_test[i][0]} with {series_test[i][1][2]} images")
    
    # create the image dataloaders with corresponding indices
    train_indices = []
    val_indices = []
    test_indices = []
    for i in range(len(series_train)):
        train_indices += list(range(series_train[i][1][0], series_train[i][1][1]+1))
    for i in range(len(series_val)):
        val_indices += list(range(series_val[i][1][0], series_val[i][1][1]+1))
    for i in range(len(series_test)):
        test_indices += list(range(series_test[i][1][0], series_test[i][1][1]+1))
    print(f"Number of Training images : {len(train_indices)}")
    print(f"Number of Validation images : {len(val_indices)}")
    print(f"Number of Testing images : {len(test_indices)}\n")
    
    train_indices_per_series = []
    val_indices_per_series = []
    test_indices_per_series = []
    for i in range(len(series_train)):
        train_indices_per_series.append(list(range(series_train[i][1][0], series_train[i][1][1]+1)))
    for i in range(len(series_val)):
        val_indices_per_series.append(list(range(series_val[i][1][0], series_val[i][1][1]+1)))
    for i in range(len(series_test)):
        test_indices_per_series.append(list(range(series_test[i][1][0], series_test[i][1][1]+1)))
    # --- création des batch samplers ---
    val_batch_sampler   = SeriesBatchSampler(val_indices_per_series,   shuffle=False)
    test_batch_sampler  = SeriesBatchSampler(test_indices_per_series,  shuffle=False)

    train_dataset = Subset(image_dataset, train_indices)
    val_dataset   = Subset(image_dataset, val_indices)
    test_dataset  = Subset(image_dataset, test_indices)
    
    # --- DataLoaders ---
    train_loader = DataLoader( # data loader made for training 
        train_dataset, # train set 
        batch_size=default_batch_size, 
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, # validation set
        batch_size=default_batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, # test set
        batch_size=default_batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
        pin_memory=True
    )
    
    val_loader_series = DataLoader( # loader that loads all the images in a series at a time -> same images as the val_loader but load the whole serie (in order) at once
        image_dataset, # dataset (+sampler)
        batch_sampler=val_batch_sampler, # sampler that gives the indices of the images of a serie
        shuffle=False, # no need to shuffle the series
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
        pin_memory=True
    )
    
    test_loader_series = DataLoader(
        image_dataset,
        batch_sampler=test_batch_sampler, # sampler that gives the indices of the images of a serie
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
        pin_memory=True
    )
    return train_loader, val_loader, test_loader, val_loader_series, test_loader_series