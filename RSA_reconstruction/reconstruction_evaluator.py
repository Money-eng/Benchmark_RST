from __future__ import annotations

import gc
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from dask.distributed import Client, LocalCluster, Future
from monai.inferers import SlidingWindowInfererAdapt
from torch.nn import Module
from torch.utils.data import DataLoader
from tqdm import tqdm
from rsml import rsml2mtg
from utils.misc import SEED, set_seed

set_seed(SEED)


class ReconstructionEvaluator:
    def __init__(
            self,
            test_list_folders: Sequence[str],
            val_list_folders: Sequence[str],
            predictions: Dict[str, pd.DataFrame],
            metrics: Optional[Dict[str, Module]] = None,
            use_dask: bool = True,
    ) -> None:
        # Misc. -----------------------------------------------------------
        self.use_dask = use_dask
        self.predictions = predictions
        self.metrics = metrics

        # Dask cluster for parallel CPU‑metric evaluation -----------------
        self.cluster: Optional[LocalCluster] = None
        self.client: Optional[Client] = None

        if self.use_dask:
            self.cluster = LocalCluster(
                n_workers=max(1, int(0.8 * os.cpu_count())),
                threads_per_worker=1,
                processes=True,
                memory_limit="8GB",
            )
            self.client = Client(self.cluster)

        # We assume that in every folder there is a "61_graph.rsml" file (the expertized RSML) and a "61_before_expertized_graph.rsml" file (the before expertized RSML)
        self.dict_gt = {
            "test": {
                folder: {
                    "expertized": rsml2mtg(os.path.join(folder, "61_graph.rsml")),
                    "before_expertized": rsml2mtg(os.path.join(folder, "61_before_expertized_graph.rsml"))
                }
                for folder in test_list_folders
            },
            "val": {
                folder: {
                    "expertized": rsml2mtg(os.path.join(folder, "61_graph.rsml")),
                    "before_expertized": rsml2mtg(os.path.join(folder, "61_before_expertized_graph.rsml"))
                }
                for folder in val_list_folders
            },
        }
        
        
    def evaluate(self, ) -> Dict[str, Dict[str, pd.DataFrame]]:
        
        return