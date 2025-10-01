from __future__ import annotations

import os
import gc
from collections import defaultdict
from typing import Dict, Optional

import tifffile
import torch
from monai.inferers import SlidingWindowInfererAdapt
from openalea.mtg import MTG
from torch.nn import Module
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.launch_RST import process_date_map
from utils.misc import SEED, set_seed

set_seed(SEED)

TARGET_SIZE = (1348, 1166)


class Reconstructor:
    def __init__(
            self,
            model: Module,
            val_dataloader: DataLoader,
            test_dataloader: DataLoader,
            device: torch.device,
            model_name: str = "Model_X",
            threshold: float = 0.5,
            patch_size: Optional[int] = None,
            jar_path: Optional[str] = None,
            save_path: Optional[str] = None
    ) -> None:

        # --------------------------- Public fields ----------------------
        self.model = model.to(device).eval()  # ensure eval mode
        self.device = device
        self.val_loader = val_dataloader
        self.test_loader = test_dataloader
        self.threshold = threshold
        self.model_name = model_name

        # Sliding‑window inference ---------------------------------------
        self.sw_inferer: Optional[SlidingWindowInfererAdapt] = None
        if patch_size is not None:
            self.sw_inferer = SlidingWindowInfererAdapt(
                roi_size=(int(patch_size), int(patch_size)),
                sw_batch_size=4,
                overlap=0.25,
                mode="constant",
            )

        # jar path for Root System Tracker (RST) ----------------------
        self.jar_path = jar_path
        self.save_path = save_path

    # =====================================================================
    # API
    # =====================================================================

    # {"Test", "Val" : {path: MTG}}
    def reconstruct_all(self) -> Dict[str, Dict[str, MTG]]:
        self.model.eval()
        # Containers ------------------------------------------------------
        predicted_mtgs: Dict[str, Dict[str, MTG]] = defaultdict(dict)
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Evaluating",
                        leave=False, dynamic_ncols=True)
            for imgs, masks, _, mtg_list in pbar:
                # get box name of mtg path (before last slash)
                mtg_path = mtg_list[0]
                mtg_box_name = mtg_path.split("/")[-2]
                # Process MTG
                try:
                    pred_mtg = self.reconstruct(imgs, masks, mtg_list, save_path=os.path.join(
                        self.save_path, "Val", mtg_box_name))
                except Exception as e:
                    print(f"Error processing {mtg_box_name}: {e}")
                    continue
                val_or_test_str = "Val"
                predicted_mtgs[val_or_test_str][mtg_list[0]] = pred_mtg
            pbar.close()
            # Clear memory
            gc.collect()
            #torch.cuda.empty_cache()

            pbar = tqdm(self.test_loader, desc="Evaluating",
                        leave=False, dynamic_ncols=True)
            for imgs, masks, _, mtg_list in pbar:
                # get box name of mtg path (before last slash)
                mtg_path = mtg_list[0]
                mtg_box_name = mtg_path.split("/")[-2]
                # Process MTG
                try:
                    pred_mtg = self.reconstruct(imgs, masks, mtg_list, save_path=os.path.join(
                        self.save_path, "Test", mtg_box_name))
                except Exception as e:
                    print(f"Error processing {mtg_box_name}: {e}")
                    continue
                val_or_test_str = "Test"
                predicted_mtgs[val_or_test_str][mtg_list[0]] = pred_mtg
            pbar.close()
            # Clear memory
            gc.collect()
            #torch.cuda.empty_cache()

        return predicted_mtgs

    def reconstruct(self, imgs: torch.Tensor, masks: torch.Tensor, mtgs: list, save_path: str) -> MTG:
        # a batch is composed (for UC1 of 29 images) -> direct call to process_date_map
        imgs = imgs.to(self.device)
        masks = masks.to(self.device).float()

        # (B, C, H, W) - already sigmoid
        predictions = self._infer(imgs)

        # save probability heatmap in save_path
        if False:
            import os
            os.makedirs(save_path, exist_ok=True)
            for i in range(predictions.shape[0]):
                pred_img = predictions[i].cpu().numpy()
                tifffile.imwrite(os.path.join(save_path, f"pred_heatmap_{i}.tif"), pred_img)

        preds = predictions.float()  # (predictions > self.threshold).float()

        # original image size is 1348 × 1166 but = 1376 × 1184 after padding operation : A.PadIfNeeded(min_height=ajusted_width, min_width=ajusted_height, border_mode=0, position='top_left'),
        # removing padding to get the original size
        preds = preds[:, :, :TARGET_SIZE[1], :TARGET_SIZE[0]]
        masks = masks[:, :, :TARGET_SIZE[1], :TARGET_SIZE[0]]
        _, mtg_pred = process_date_map(mtgs,
                                       preds,
                                       save_path=save_path,
                                       jar_path=self.jar_path)

        del preds, predictions
        #torch.cuda.empty_cache()
        return mtg_pred

    # =====================================================================
    # Internal helpers
    # =====================================================================

    def _infer(self, imgs: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional sliding-window inference."""
        if self.sw_inferer is None:
            return self.model(imgs)
        return self.sw_inferer(inputs=imgs, network=self.model)
