import os
import numpy as np
import torch
from torch.utils.data import Dataset
import tifffile
from torchvision import transforms
from .tiff_reader import CachedTiffReader


class RSASeg2DDataset(Dataset):
    def __init__(
        self,
        rsa_dir_loader: None,
        mode='series',
        img_transform=None,
        mask_transform_series=None,
        mask_transform_image=None,
        image_with_mtg=False,
        as_RGB=False
    ):
        self.mode = mode
        self.samples = []
        self.img_transform = img_transform
        self.mask_transform_series = mask_transform_series
        self.mask_transform_image = mask_transform_image
        self.image_with_mtg = image_with_mtg
        self.as_RGB = as_RGB
        self.tiff_reader = CachedTiffReader()

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

            if mode == 'series':
                self.samples.append(
                    (img_path, mask_path, num_slices, mtg_path))
            elif mode == 'image':
                for z in range(num_slices):
                    self.samples.append((img_path, mask_path, z, mtg_path))
            else:
                raise ValueError(
                    "Mode non reconnu, choisissez 'series' ou 'image'")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path, slice_info, mtg_path = self.samples[idx]
        if self.mode == 'series':
            img = tifffile.imread(img_path)
            mask_raw = tifffile.imread(mask_path)
            info = slice_info
            if self.img_transform:
                img = self.img_transform(img)
            mask = (
                self.mask_transform_series(mask_raw)
                if self.mask_transform_series
                else mask_raw
            )
        else:
            z = slice_info
            img = self.tiff_reader.get_page(img_path, z)
            mask_raw = tifffile.imread(mask_path)
            mask = np.where((mask_raw != 0) & (mask_raw <= z+1), 1, 0)
            if self.img_transform:
                img = self.img_transform(img)
            mask = (
                self.mask_transform_image(mask)
                if self.mask_transform_image
                else mask
            )
            info = z

        if self.as_RGB:
            img = img.repeat(3, 1, 1)
        additional = mtg_path if self.image_with_mtg else torch.tensor(0)
        return img, mask.float(), info, additional
