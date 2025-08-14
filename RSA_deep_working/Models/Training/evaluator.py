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
#from torch.profiler import profile, tensorboard_trace_handler, ProfilerActivity, schedule
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.logger import TensorboardLogger
from utils.misc import SEED, set_seed

# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------
set_seed(SEED)


# -----------------------------------------------------------------------------
# Evaluator class
# -----------------------------------------------------------------------------
class Evaluator:
    def __init__(
            self,
            model: Module,
            criterion: Optional[Module],
            val_dataloader: DataLoader,
            test_dataloader: DataLoader,
            metrics: Dict[str, Sequence[Module]],
            device: torch.device,
            logger: Optional[logging.Logger] = None,
            tb_logger: Optional[TensorboardLogger] = None,
            log_metric_path: str | os.PathLike | None = None,
            threshold: float = 0.5,
            epoch: int = 0,
            patch_size: Optional[int] = None,
            *,
            roi_fnc: Optional[callable] = None,
            compute_cpu_metrics: bool = True,
            use_dask: bool = True,
            profile_dir: Optional[str] = None,
    ) -> None:

        # --------------------------- Public fields ---------------------
        self.model = model.to(device).eval()  # ensure eval mode
        self.criterion = criterion.to(device) if criterion else None
        self.val_loader = val_dataloader
        self.test_loader = test_dataloader

        # Metric groups ---------------------------------------------------
        self.gpu_metrics: List[Module] = list(metrics.get("gpu", []))
        self.cpu_metrics: List[Module] = list(metrics.get("cpu", []))

        # Misc. -----------------------------------------------------------
        self.device = device
        self.logger = logger or logging.getLogger(__name__)
        self.tb_logger = tb_logger
        self.epoch = epoch
        self.threshold = threshold
        self.compute_cpu_metrics = compute_cpu_metrics
        self.use_dask = use_dask
        self.profile_dir = Path(profile_dir) if profile_dir else None

        # Output path for long‑term metric storage ------------------------
        self.metrics_dir = Path(log_metric_path) if log_metric_path else None
        self.metrics_file = None
        if self.metrics_dir is not None:
            self.metrics_dir.mkdir(parents=True, exist_ok=True)
            self.metrics_file = os.path.join(
                self.metrics_dir, "metrics_results.parquet")
            self.metrics_file = Path(self.metrics_file)

        # Sliding‑window inference ---------------------------------------
        self.sw_inferer: Optional[SlidingWindowInfererAdapt] = None
        if patch_size is not None:
            self.sw_inferer = SlidingWindowInfererAdapt(
                roi_size=(int(patch_size), int(patch_size)),
                sw_batch_size=4,
                overlap=0.25,
                mode="constant",
            )

        # Dask cluster for parallel CPU‑metric evaluation -----------------
        self.cluster: Optional[LocalCluster] = None
        self.client: Optional[Client] = None

        if self.compute_cpu_metrics and self.use_dask:
            self.cluster = LocalCluster(
                n_workers=max(1, int(0.8 * os.cpu_count())),
                threads_per_worker=1,
                processes=True,
                memory_limit="8GB",
            )
            self.client = Client(self.cluster)

        self._log(logging.INFO, "Evaluator initialised (epoch %d).", self.epoch)
        self.roi_fnc = roi_fnc

    # =====================================================================
    # API
    # =====================================================================
    def evaluate(self, on_test: bool = False) -> Dict[str, float]:
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
            for imgs, masks, time, mtgs in pbar:
                imgs = imgs.to(self.device)
                masks = masks.to(self.device).float()

                # (B, C, H, W) - already sigmoid
                predictions = self._infer(imgs)
                preds = (predictions > self.threshold).float()

                if self.roi_fnc is not None:
                    # imgs is a tensor, time and mtgs are lists
                    roi_masks = self.roi_fnc(imgs, time, mtgs)
                    roi_masks = roi_masks.to(self.device).float()
                else:
                    roi_masks = torch.ones_like(masks)

                # Apply ROI mask to predictions and ground truth masks
                preds_of_interest = preds * roi_masks
                masks_of_interest = masks * roi_masks

                # -------------------------- Loss -----------------------
                if self.criterion is not None:
                    # loss_val = self.criterion(predictions, masks).item()
                    loss_val = self.criterion(
                        preds_of_interest, masks_of_interest).item()
                    raw[f"val_loss_{self.criterion.__class__.__name__}"].append(
                        loss_val)
                    pbar.set_postfix(loss=f"{loss_val:.4f}")

                # ---------------------- GPU metrics --------------------
                for metric in self.gpu_metrics:
                    name = metric.__class__.__name__
                    # pbar.set_postfix(**{"Metric": name})
                    # raw[name].append(metric(preds, masks))
                    pbar.set_postfix(**{"Metric": f"{name}"})
                    value = metric(preds_of_interest, masks_of_interest)
                    if isinstance(value, (float, int, np.floating)):
                        raw[name].append(float(value))
                    elif isinstance(value, dict):
                        for k, v in value.items():
                            key_str = f"{name}_{k}"
                            if isinstance(v, (float, int, np.floating)):
                                raw[key_str].append(float(v))
                            elif isinstance(v, (list, tuple, np.ndarray)):
                                raw[key_str].append(float(np.mean(v)))
                            else:
                                self._log(
                                    logging.WARNING, f"Unhandled value type {type(v)} for metric {name}:{k}")
                    elif isinstance(value, (list, tuple, np.ndarray)):
                        raw[name].append(float(np.mean(value)))
                    else:
                        self._log(
                            logging.WARNING, f"Unhandled GPU metric return type {type(value)} for {name}")

                # ---------------------- CPU metrics -----------------------
                if self.compute_cpu_metrics:
                    pbar.set_postfix(
                        **{"CPU Metrics": "adding to memory..."})
                    # preds_cpu.extend(preds.detach().cpu().numpy())
                    # masks_cpu.extend(masks.detach().cpu().numpy())
                    if self.roi_fnc is not None:
                        preds_cpu.extend(
                            preds_of_interest.detach().cpu().numpy())
                        masks_cpu.extend(
                            masks_of_interest.detach().cpu().numpy())

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

    # ---------------------------------------------------------------------
    def done_evaluating(self) -> None:
        """Release Dask resources explicitly."""
        if self.client is not None:
            self.client.close()
        if self.cluster is not None:
            self.cluster.close()
        self._log(logging.INFO, "Dask resources cleaned up.")

    # =====================================================================
    # Internal helpers
    # =====================================================================

    def _infer(self, imgs: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional sliding-window inference."""
        if self.sw_inferer is None:
            return self.model(imgs)
        return self.sw_inferer(inputs=imgs, network=self.model)

    # ------------------------------------------------------------------
    def _compute_cpu_metrics(
            self,
            preds: List[np.ndarray],
            masks: List[np.ndarray],
            raw: Dict[str, List[float]],
    ) -> None:
        """Compute CPU & MTG metrics, potentially in parallel across all metrics."""
        metrics = self.cpu_metrics  # + self.mtg_metrics
        if not metrics:
            return

        # If Dask is enabled, schedule all metrics at once
        if self.use_dask and self.client:
            # Map each metric to its list of futures
            tasks: Dict[str, List[Future]] = {
                metric.__class__.__name__: self.client.map(
                    metric, preds, masks)
                for metric in metrics
            }

            # Show progress over metric‐names
            pbar = tqdm(tasks.keys(), desc="CPU metrics",
                        leave=False, dynamic_ncols=True)

            # Gather all results in one go; returns a dict name→List[values]
            results: Dict[str, List[float]] = self.client.gather(tasks)

            # Aggregate per‐metric
            for name in pbar:
                pbar.set_postfix(metric=name)
                _aggregate_metric_results(name, results[name], raw)

        else:
            # Fallback: compute each metric sequentially
            pbar = tqdm(metrics, desc="CPU metrics",
                        leave=False, dynamic_ncols=True)
            for metric in pbar:
                name = metric.__class__.__name__
                pbar.set_postfix(metric=name)
                values = [metric(p, m) for p, m in zip(preds, masks)]
                _aggregate_metric_results(name, values, raw)

    # ------------------------------------------------------------------
    def _persist_raw_metrics(self, raw: Dict[str, List[float]]) -> None:
        if self.metrics_file is None:
            return
        records = [{"epoch": self.epoch, "metric": k, "value": float(
            v)} for k, vals in raw.items() for v in vals]
        df = pd.DataFrame.from_records(records)

        if self.metrics_file.exists():
            df_old = pd.read_parquet(self.metrics_file)
            df = pd.concat([df_old, df], ignore_index=True)
        df.to_parquet(self.metrics_file, index=False,
                      engine="pyarrow", compression="snappy")

    # ------------------------------------------------------------------
    def _log(self, lvl: int, msg: str, *args, **kwargs) -> None:
        if self.logger is not None:
            self.logger.log(lvl, msg, *args, **kwargs)
        else:
            print(msg % args)


# -----------------------------------------------------------------------------
# Stand‑alone helper functions
# -----------------------------------------------------------------------------
def _aggregate_metric_results(name: str, results: Sequence, raw: Dict[str, List[float]]) -> None:
    """Normalise heterogeneous *results* into flat, numeric lists inside *raw*."""
    for r in results:
        if isinstance(r, (float, int, np.floating)):
            raw[name].append(float(r))
        elif isinstance(r, list):
            raw[name].append(float(np.mean(r)))
        elif isinstance(r, np.ndarray):
            raw[name].append(float(np.mean(r)))
        elif isinstance(r, dict):
            for k, v in r.items():
                raw[f"{name}_{k}"].append(float(v))
        else:
            print(
                f"[Evaluator] Warning: unhandled type {type(r)} for metric {name} - skipped.")
