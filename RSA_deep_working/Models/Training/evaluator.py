# Training/evaluator.py

import os
import numpy as np
import torch
from logging import Logger
from tqdm import tqdm
from utils.launch_RST import process_date_map
from utils.logger import TensorboardLogger
from monai.inferers import SlidingWindowInfererAdapt
from dask.distributed import Client, LocalCluster
from dask import delayed, compute
from collections import defaultdict
import gc

from utils.misc import set_seed, SEED

set_seed(SEED)  # Ensure reproducibility


class Evaluator:
    def __init__(
            self,
            model: torch.nn.Module,
            val_dataloader: torch.utils.data.DataLoader,
            test_dataloader: torch.utils.data.DataLoader,
            # {"gpu": [Dice(), ...], "cpu": [Connectivity(), ..., "mtg": [AreaBetweenIntercepts(), ...]}
            metrics: dict,
            device: torch.device,
            logger: Logger = None,
            tb_logger: TensorboardLogger = None,
            log_metric_path: str = None,
            jar_path: str = None,
            threshold: float = 0.5,
            epoch: int = 0,
            patch_size: int = None,
    ):
        """
        - model : réseau PyTorch (déjà instancié et .to(device))
        - val_dataloader, test_dataloader, etc.
        - metrics : dict avec deux listes : "gpu" et "cpu"
        - device : torch.device("cuda") ou ("cpu")
        - logger / tb_logger : pour journaliser si besoin
        """
        self.model = model.to(device)
        self.device = device

        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader

        self.gpu_metrics = metrics.get("gpu", [])
        self.cpu_metrics = metrics.get("cpu", [])
        self.mtg_metrics = metrics.get("mtg", [])

        self.jar_path = jar_path

        self.logger = logger
        self.tb_logger = tb_logger
        self.epoch = epoch
        self.threshold = threshold

        self.metrics_dir = log_metric_path or "metrics"
        os.makedirs(self.metrics_dir, exist_ok=True)
        self.metrics_file = os.path.join(
            self.metrics_dir, "metrics_results.parquet")

        self.do_not_reconstruct = False
        self.cpu_metrics_evaluation = True

        if self.logger:
            self.logger.info("[Evaluator] Initialisation terminée.")
        else:
            print("[Evaluator] Initialisation terminée.")

        if patch_size is not None:
            self.sw_inferer = SlidingWindowInfererAdapt(
                roi_size=(int(patch_size), int(patch_size)),
                sw_batch_size=4,
                overlap=0.25,
                mode="constant",
            )
        else:
            self.sw_inferer = None

        self.cluster = LocalCluster(
            # min(len(self.cpu_metrics) + len(self.mtg_metrics), os.cpu_count()),  # limit workers
            n_workers=int(0.8 * os.cpu_count()),
            threads_per_worker=1,
            processes=True,
            memory_limit="8GB",
        )
        self.client = Client(self.cluster)

        self.logger.info(
            '[Evaluator] Initialisation de l\'Evaluator terminée avec Dask Client.')

    def evaluate(self, on_test: bool = False) -> dict:
        """
        Evaluate the model on validation or test data.
        - on_test : bool, if True, evaluate on test data, else on validation data
        - Returns a dictionary with metric names as keys and their mean values as values.
        """
        self.model.eval()
        dataloader = self.test_dataloader if on_test else self.val_dataloader

        results_raw = defaultdict(list)
        for metric in self.gpu_metrics + self.cpu_metrics:
            name = metric.__class__.__name__
            results_raw[name] = []
        save_first_pred = True
        with torch.no_grad():
            with tqdm(dataloader, desc="Evaluating batch per batch", leave=False, dynamic_ncols=True) as pbar:
                all_preds, all_masks = [], []
                for imgs, masks, _, _ in pbar:
                    imgs = imgs.to(self.device)
                    masks = masks.to(self.device)
                    if self.sw_inferer is None:
                        preds_logits_sigmoidee = self.model(imgs)
                    else:
                        preds_logits_sigmoidee = self.sw_inferer(
                            inputs=imgs, network=self.model)
                    # SIGMOID + BINARY THRESHOLDING NEEDED IN METRICS
                    preds = (preds_logits_sigmoidee >
                             self.threshold).float()
                    ##### GPU METRICS #####
                    for metric in self.gpu_metrics:
                        name = metric.__class__.__name__
                        pbar.set_postfix_str(f"Metric: {name}")
                        results_raw[name].append(metric(preds, masks))

                    ##### CPU METRICS #####
                    if self.cpu_metrics_evaluation:
                        # [B, C, H, W] -> X x [C, H, W]
                        all_preds.extend(preds.detach().cpu().numpy())
                        all_masks.extend(masks.detach().cpu().numpy())

                    # Log first prediction and mask
                    if save_first_pred and self.tb_logger:
                        img_to_log = imgs * 255
                        self.tb_logger.log_image("Image", img_to_log, global_step=self.epoch)
                        self.tb_logger.log_image("Mask", masks * 255, global_step=self.epoch)
                        self.tb_logger.log_image("Prediction", preds * 255, global_step=self.epoch)
                        save_first_pred = False

                    del preds, preds_logits_sigmoidee
                    torch.cuda.empty_cache()

                if self.cpu_metrics_evaluation:
                    preds_future = self.client.scatter(
                        all_preds, broadcast=True)
                    masks_future = self.client.scatter(
                        all_masks, broadcast=True)

                    tasks = []  # list of delayed calls
                    names = []  # parallel list of metric names
                    for metric in self.cpu_metrics:
                        name = metric.__class__.__name__
                        for pred_fut, mask_fut in zip(preds_future, masks_future):
                            task = delayed(metric)(pred_fut, mask_fut)
                            tasks.append(task)
                            names.append(name)

                    # returns a list of results
                    results_compute = compute(*tasks, scheduler=self.client)

                    # aggregate back into results
                    agg = defaultdict(list)
                    for name, val in zip(names, results_compute):
                        if isinstance(val, int):
                            agg[name].append(val)
                        elif isinstance(val, float):
                            agg[name].append(val)
                        elif isinstance(val, list):
                            agg[name].append(np.mean(val))
                        elif isinstance(val, np.ndarray):
                            agg[name].append(np.mean(val))
                        elif isinstance(val, dict):
                            for k, v in val.items():
                                agg[name + "_" + str(k)].append(v)
                        else:
                            print
                            continue

                    for name, values in agg.items():
                        if name not in results_raw:
                            results_raw[name] = []
                        results_raw[name].extend(values)

        # empty cache to free memory
        torch.cuda.empty_cache()
        # free cpu memory
        gc.collect()

        mean_results = {name: np.mean(
            values) if values else 0.0 for name, values in results_raw.items()}
        return mean_results

    def done_evaluating(self):
        """
        Clean up Dask resources.
        """
        self.client.close()
        self.cluster.close()
        if self.logger:
            self.logger.info("[Evaluator] Dask resources cleaned up.")
        else:
            print("[Evaluator] Dask resources cleaned up.")
