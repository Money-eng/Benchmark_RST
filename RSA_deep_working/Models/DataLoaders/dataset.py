import os
import numpy as np
import torch
from torch.utils.data import Dataset
import tifffile
from PIL import Image
from .tiff_reader import CachedTiffReader


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
        tiff_reader (CachedTiffReader): A cached TIFF reader for efficient image loading.

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
        rsa_dir_loader=None,
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
        img_path, mask_path, num_slices, mtg_path = self.samples[idx]

        if self.mode == 'series':
            img_np   = tifffile.imread(img_path)         # NumPy array, maybe uint16 or int
            mask_raw = tifffile.imread(mask_path)        # Likely a date‐map int64

            # Convert image → PIL so that img_transform (Pad, etc.) works:
            img = Image.fromarray(img_np)

            # Cast mask_raw to uint8 before giving to PIL:
            mask_uint8 = mask_raw.astype(np.uint8)
            mask = Image.fromarray(mask_uint8)

            time = num_slices
            if self.img_transform:
                img = self.img_transform(img)

            # If you have a transform for masks (series-level), remember to give it a PIL mask:
            if self.mask_transform_series:
                # (mask_transform_series may expect a PIL Image)
                mask = self.mask_transform_series(mask)
            else:
                # If you want to keep it as a raw NumPy fallback, you could do:
                mask = torch.from_numpy(mask_uint8).float()

        else:  # mode == 'image'
            z = num_slices
            img_np   = self.tiff_reader.get_page(img_path, z)  # NumPy array
            mask_raw = tifffile.imread(mask_path)              # date‐map int64

            # Build binary mask up to slice z:
            mask_np = np.where((mask_raw != 0) & (mask_raw <= z + 1), 1, 0)

            # Convert image → PIL:
            img = Image.fromarray(img_np)

            # Cast the binary mask to uint8:
            mask_uint8 = mask_np.astype(np.uint8)
            mask = Image.fromarray(mask_uint8)

            if self.img_transform:
                img = self.img_transform(img)

            if self.mask_transform_image:
                mask = self.mask_transform_image(mask)
            else:
                mask = torch.from_numpy(mask_uint8).float()

            time = z

        # If you really need RGB channels (rare for RSA), you can do this later on a tensor.
        if self.as_RGB:
            # But if img is still PIL here, you’d do something like:
            # img = transforms.Grayscale(num_output_channels=3)(img)
            # or convert back to tensor then repeat channels. 
            pass

        mtg = mtg_path if self.image_with_mtg else torch.tensor(0)

        # At this point, ensure `mask` is a tensor (if you want to do `mask.clone()`):
        if isinstance(mask, Image.Image):
            # Convert PIL mask back to tensor if your training loop expects that:
            mask = torch.from_numpy(np.array(mask)).float()

        # Now you can safely call `mask.clone().detach()` in your training loop.
        return img, mask, time, mtg