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
from utils.misc import SEED, set_seed

set_seed(SEED)


class ReconstructionEvaluator:
    def __init__(
            self,
            logger: Optional[logging.Logger] = None,
            use_dask: bool = True,
    ) -> None:
        # Misc. -----------------------------------------------------------
        self.logger = logger or logging.getLogger(__name__)
        self.use_dask = use_dask
        
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

        self._log(logging.INFO, "Evaluator initialised (epoch %d).", self.epoch)

    # =====================================================================
    # API
    # =====================================================================
    def reconstruct(self, on_test: bool = False) -> Dict[str, float]:
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
            for imgs, masks, _, mtgs in pbar:
                

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
        metrics = self.cpu_metrics # + self.mtg_metrics
        if not metrics:
            return

        # If Dask is enabled, schedule all metrics at once
        if self.use_dask and self.client:
            # Map each metric to its list of futures
            tasks: Dict[str, List[Future]] = {
                metric.__class__.__name__: self.client.map(metric, preds, masks)
                for metric in metrics
            }

            # Show progress over metric‐names
            pbar = tqdm(tasks.keys(), desc="CPU metrics", leave=False, dynamic_ncols=True)

            # Gather all results in one go; returns a dict name→List[values]
            results: Dict[str, List[float]] = self.client.gather(tasks)

            # Aggregate per‐metric
            for name in pbar:
                pbar.set_postfix(metric=name)
                _aggregate_metric_results(name, results[name], raw)

        else:
            # Fallback: compute each metric sequentially
            pbar = tqdm(metrics, desc="CPU metrics", leave=False, dynamic_ncols=True)
            for metric in pbar:
                name = metric.__class__.__name__
                pbar.set_postfix(metric=name)
                values = [metric(p, m) for p, m in zip(preds, masks)]
                _aggregate_metric_results(name, values, raw)

    # ------------------------------------------------------------------
    def _persist_raw_metrics(self, raw: Dict[str, List[float]]) -> None:
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
