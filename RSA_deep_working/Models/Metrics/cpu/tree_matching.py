# Metrics/cpu/ari_index.py

import numpy as np
import torch
import tempfile
from ..base import BaseMetric
from utils.root_System_class import RootSystem


class TreeMatching(BaseMetric):
    type = "cpu"
    need = "serie"

    def __init__(self):
        super().__init__()

    def __call__(self, prediction: torch.Tensor, mask: torch.Tensor, time, mtg: str) -> float:
        # Create 2 temp directories for the input and output using the library tempfile
        random_suffix = str(np.random.randint(1000000, 9999999))
        random_suffix2 = str(np.random.randint(1000000, 9999999))
        # parent directory of mtg file
        data_input_dir = mtg.rsplit('/', 1)[0] if '/' in mtg else '.'
        input_dir = tempfile.mkdtemp(prefix="tree_matching_input_" + random_suffix) # save prediction to
        output_dir = tempfile.mkdtemp(prefix="tree_matching_output_" + random_suffix2)
        print(f"Input directory: {input_dir}")
        print(f"Output directory: {output_dir}")
        print(f"Data input directory: {data_input_dir}")
        
        # Assemble date_map from batch_size 2D grayscale images
        prediction = prediction.squeeze(0).cpu().numpy()
        two_d_prediction = np.zeros_like(prediction, dtype=np.uint8)
        for i in range(prediction.shape[0]):
            two_d_prediction[prediction[i] > 0 & two_d_prediction[i] == 0] = i + 1
        
        # save tif files in the input directory
        input_file = f"{input_dir}/40_date_map.tif"
        import tifffile as tiff
        tiff.imwrite(input_file, two_d_prediction.astype(np.uint8))        