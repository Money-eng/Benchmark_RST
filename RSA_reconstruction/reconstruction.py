from __future__ import annotations

import gc
import logging
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import torch
from monai.inferers import SlidingWindowInfererAdapt
from torch.nn import Module
from torch.utils.data import DataLoader
from tqdm import tqdm

from openalea.mtg import MTG

from utils.misc import SEED, set_seed
from utils.launch_RST import process_date_map

set_seed(SEED)


class Reconstructor:
    def __init__(
            self,
            model: Module,
            val_dataloader: DataLoader,
            test_dataloader: DataLoader,
            device: torch.device,
            threshold: float = 0.5,
            patch_size: Optional[int] = None,
            jar_path: Optional[str] = None,
            logger: Optional[logging.Logger] = None,
    ) -> None:
        
        # --------------------------- Public fields ----------------------
        self.model = model.to(device).eval()  # ensure eval mode
        self.val_loader = val_dataloader
        self.test_loader = test_dataloader
        self.threshold = threshold
        
        # Misc. -----------------------------------------------------------
        self.logger = logger or logging.getLogger(__name__)
        
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


    # =====================================================================
    # API
    # =====================================================================
    def reconstruct_all(self, on_test: bool = False) -> Dict[str, Dict[str, MTG]]: # {"test / val" : {path: MTG}}
        """Run a forward pass on *val* or *test* set and compute metrics."""
        self.model.eval()
        loader = self.test_loader if on_test else self.val_loader

        # Containers ------------------------------------------------------
        raw: Dict[str, List[float]] = defaultdict(list)
        for m in self.gpu_metrics + self.cpu_metrics:
            raw[m.__class__.__name__] = []
        if self.criterion is not None:
            raw[f"val_loss_{self.criterion.__class__.__name__}"] = []

        # ----------------------------------------------------------------
        save_first_pred = True  # one‑off TB image logging flag
        preds_cpu: List[np.ndarray] = []
        masks_cpu: List[np.ndarray] = []

        with torch.no_grad():
            pbar = tqdm(loader, desc="Evaluating", leave=False, dynamic_ncols=True)
            for imgs, masks, *_ in pbar:
                imgs = imgs.to(self.device)
                masks = masks.to(self.device).float()

                # (B, C, H, W) - already sigmoid
                predictions = self._infer(imgs)
                preds = (predictions > self.threshold).float()

                # -------------------------- Loss -----------------------
                if self.criterion is not None:
                    loss_val = self.criterion(predictions, masks).item()
                    raw[f"val_loss_{self.criterion.__class__.__name__}"].append(
                        loss_val)
                    pbar.set_postfix(loss=f"{loss_val:.4f}")

                # ---------------------- GPU metrics --------------------
                for metric in self.gpu_metrics:
                    name = metric.__class__.__name__
                    raw[name].append(metric(preds, masks))

                # ---------------------- CPU metrics -----------------------
                if self.compute_cpu_metrics:
                    preds_cpu.extend(preds.detach().cpu().numpy())
                    masks_cpu.extend(masks.detach().cpu().numpy())

                # ---------------- TensorBoard images -------------------
                if save_first_pred and self.tb_logger is not None:
                    self.tb_logger.log_image(
                        "Mask", masks * 255.0, global_step=self.epoch)
                    self.tb_logger.log_image(
                        "Prediction", preds * 255.0, global_step=self.epoch)
                    save_first_pred = False

                # Manual clean‑up --------------------------------------
                del preds, predictions
                torch.cuda.empty_cache()

        # ----------------------------------------------------------------
        if self.compute_cpu_metrics:
            self._compute_cpu_metrics(preds_cpu, masks_cpu, raw)

        # ----------------------------------------------------------------
        torch.cuda.empty_cache()
        gc.collect()

        # Aggregate means -------------------------------------------------
        mean_results = {
            k: float(np.mean(v)) if v else 0.0 for k, v in raw.items()}
        self._log(logging.INFO, "Epoch %d - Results: %s",
                  self.epoch, mean_results)

        self._persist_raw_metrics(raw)
        return mean_results

    def reconstruct(self, imgs: torch.Tensor, masks: torch.Tensor, mtgs: list) -> MTG:
        # a batch is composed (for UC1 of 29 images) -> direct call 
        imgs = imgs.to(self.device)
        masks = masks.to(self.device).float()

        # (B, C, H, W) - already sigmoid
        predictions = self._infer(imgs)
        preds = (predictions > self.threshold).float()
        
        _, mtg_pred = process_date_map(mtgs, preds, jar_path=self.jar_path)
        return mtg_pred


    # =====================================================================
    # Internal helpers
    # =====================================================================
    def _infer(self, imgs: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional sliding-window inference."""
        if self.sw_inferer is None:
            return self.model(imgs)
        return self.sw_inferer(inputs=imgs, network=self.model)

    # ------------------------------------------------------------------
    def _log(self, lvl: int, msg: str, *args, **kwargs) -> None:
        if self.logger is not None:
            self.logger.log(lvl, msg, *args, **kwargs)
        else:
            print(msg % args)
