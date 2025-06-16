import numpy as np
import os
import tifffile
import torch
from PIL import Image
from torch.utils.data import Dataset

from .tiff_reader import TiffReader


class RSADataset(Dataset):
    """
    RSADataset is a PyTorch Dataset class designed to handle RSA (Root System Architecture) data.
    It supports loading image stacks, masks, and metadata for either series or individual image slices.

    Attributes:
        mode (str): Specifies the mode of operation, either 'series' or 'image'.
        samples (list): A list of tuples containing paths and metadata for each sample.
        img_transform (callable, optional): Transformation function to apply to the images.
        mask_transform_series (callable, optional): Transformation function to apply to the mask in 'series' mode.
        mask_transform_image (callable, optional): Transformation function to apply to the mask in 'image' mode.
        image_with_mtg (bool): Whether to include metadata (MTG) in the output.
        as_RGB (bool): Whether to convert grayscale images to RGB by repeating channels.
        tiff_reader (TiffReader): A cached TIFF reader for efficient image loading.

    Methods:
        __init__(rsa_dir_loader, mode='series', img_transform=None, mask_transform_series=None,
                 mask_transform_image=None, image_with_mtg=False, as_RGB=False):
            Initializes the dataset with the given parameters and loads the samples.

        __len__():
            Returns the number of samples in the dataset.

        __getitem__(idx):
            Retrieves the sample at the specified index, including the image, mask, time/slice,
            and optionally the metadata (MTG).

    Args:
        rsa_dir_loader (object): Loader object containing paths to image stacks, masks, and metadata.
        mode (str, optional): Mode of operation, either 'series' or 'image'. Defaults to 'series'.
        img_transform (callable, optional): Transformation function for images. Defaults to None.
        mask_transform_series (callable, optional): Transformation function for masks in 'series' mode. Defaults to None.
        mask_transform_image (callable, optional): Transformation function for masks in 'image' mode. Defaults to None.
        image_with_mtg (bool, optional): Whether to include metadata (MTG) in the output. Defaults to False.
        as_RGB (bool, optional): Whether to convert grayscale images to RGB. Defaults to False.

    Raises:
        ValueError: If the mode is not 'series' or 'image'.
    """

    def __init__(
        self,
        rsa_dir_loader,
        mode: str = "series",
        img_transform=None,
        image_with_mtg: bool = False,
        as_RGB: bool = False,
    ):
        assert mode in ('series', 'image'), "Mode must be 'series' or 'image'"
        self.mode = mode
        self.samples = []
        self.img_transform = img_transform
        self.image_with_mtg = image_with_mtg
        self.as_RGB = as_RGB
        self.tiff_reader = TiffReader()

        for loader in rsa_dir_loader.loaders:
            img_path = loader.image_stack_path
            mask_path = loader.date_map_path
            mtg_path = (
                loader.rsml_default_file
                if os.path.exists(loader.rsml_default_file)
                else loader.rsml_expert_file
            )
            with tifffile.TiffFile(img_path) as tif:
                num_slices = len(tif.pages)

            if mode == "series":
                self.samples.append({
                    'img_path': img_path,
                    'mask_path': mask_path,
                    'slice_idx': None,
                    'num_slices': num_slices,
                    'mtg_path': mtg_path,
                })
            elif mode == "image":
                for z in range(num_slices):
                    self.samples.append({
                        'img_path': img_path,
                        'mask_path': mask_path,
                        'slice_idx': z,
                        'num_slices': num_slices,
                        'mtg_path': mtg_path,
                    })
            else:
                raise ValueError("Mode non reconnu, choisissez 'series' ou 'image'")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = sample['img_path']
        mask_path = sample['mask_path']
        z = sample['slice_idx']
        num_slices = sample['num_slices']
        mtg = sample['mtg_path'] if self.image_with_mtg else torch.tensor(0)

        # Load and build mask
        mask_full = tifffile.imread(mask_path)
        if self.mode == 'series':
            mask_np = (mask_full > 0).astype(np.uint8)
            img_np = self.tiff_reader.get_series(img_path)
            time = num_slices
        else:
            mask_np = ((mask_full != 0) & (mask_full <= z + 1)).astype(np.uint8)
            img_np = self.tiff_reader.get_page(img_path, z)
            time = z

        # Apply image transform if provided
        if self.img_transform:
            if img_np.ndim == 2:
                img_np_aug = img_np[..., None]
            else:
                img_np_aug = img_np
                
            # image and mask to float32
            img_np_aug = img_np_aug.astype(np.float32)
            mask_np = mask_np.astype(np.float32)
            # image and mask augmentation
            augmented = self.img_transform(image=img_np_aug, mask=mask_np)
            img = augmented['image']   # tensor [C,H,W]
            mask = augmented['mask']  
            mask = mask.unsqueeze(0)  # tensor [1,H,W]
        else:
            img = torch.from_numpy(img_np)
            if img.ndim == 2:
                img = img.unsqueeze(0)
            else:
                img = img.permute(2, 0, 1)
            img = img.float()
            mask = torch.from_numpy(mask_np).float().unsqueeze(0)

        return img, mask.clone(), time, mtg