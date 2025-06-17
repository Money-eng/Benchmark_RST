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
from collections import defaultdict


class Evaluator:
    def __init__(
        self,
        model: torch.nn.Module,
        val_dataloader: torch.utils.data.DataLoader,
        test_dataloader: torch.utils.data.DataLoader,
        val_series_dataloader: torch.utils.data.DataLoader,
        test_series_dataloader: torch.utils.data.DataLoader,
        metrics: dict,  # {"gpu": [Dice(), ...], "cpu": [Connectivity(), ..., "mtg": [AreaBetweenIntercepts(), ...]}
        device: torch.device,
        logger: Logger = None,
        tb_logger: TensorboardLogger = None,
        jar_path: str = None,
        threshold: float = 0.5,
        epoch: int = 0
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
        self.val_series_dataloader = val_series_dataloader
        self.test_series_dataloader = test_series_dataloader

        self.gpu_metrics = metrics.get("gpu", [])
        self.cpu_metrics = metrics.get("cpu", [])
        self.mtg_metrics = metrics.get("mtg", [])

        self.jar_path = jar_path

        self.logger = logger
        self.tb_logger = tb_logger
        self.epoch = epoch
        self.threshold = threshold

        if self.logger:
            self.logger.info("[Evaluator] Initialisation terminée.")
        else:
            print("[Evaluator] Initialisation terminée.")

        self.sw_inferer = SlidingWindowInfererAdapt(
            roi_size=(512, 512),
            sw_batch_size=4,
            overlap=0.25,
            mode="constant",
        )

        self.cluster = LocalCluster(
            n_workers=os.cpu_count(),
            threads_per_worker=1,
            processes=True,
            memory_limit="16GB",
        )
        self.client = Client(self.cluster)
        self.logger.info(
            '[Evaluator] Initialisation de l\'Evaluator terminée avec Dask Client.')

    def evaluate(self, last_loss_value, on_test: bool = False) -> dict:
        """
        Calcule toutes les métriques pour l'ensemble des batches de validation (ou test si on_test=True).
        Retourne un dict { metric_name: moyenne_sur_tous_les_batches }.
        """
        self.model.eval()

        dataloader = self.test_dataloader if on_test else self.val_dataloader
        data_loader_series = (
            self.test_series_dataloader if on_test else self.val_series_dataloader
        )

        results = {}
        for metric in self.gpu_metrics + self.cpu_metrics:
            name = metric.__class__.__name__
            results[name] = []

        save_first_pred = True
        with torch.no_grad():
            try:
                # if accuracy loss gives us more than 60% accuracy, we can process the whole series
                if last_loss_value < 0.3:
                    with tqdm(data_loader_series, desc="Evaluating serie per serie", leave=False, dynamic_ncols=True) as pbar:
                        for ts_imgs, masks, _, mtgs in pbar:
                            ts_imgs = ts_imgs.to(self.device)
                            masks = masks.to(self.device)

                            preds_logits_sigmoidee = self.sw_inferer(inputs=ts_imgs, network=self.model)

                            mtg_gt, mtg_pred = process_date_map(
                                mtgs, preds_logits_sigmoidee, jar_path=self.jar_path)
                            # mtg_gt, mtg_pred = process_date_map(mtgs, preds, jar_path=self.jar_path)

                            ##### MTG metrics #####
                            for metric in self.mtg_metrics:
                                name = metric.__class__.__name__
                                pbar.set_postfix_str(f"Metric: {name}")
                                value = metric(mtg_pred, mtg_gt)
                                results[name].append(value)

                with tqdm(dataloader, desc="Evaluating batch per batch", leave=False, dynamic_ncols=True) as pbar:
                    for imgs, masks, _, _ in pbar:
                        imgs = imgs.to(self.device)
                        masks = masks.to(self.device)

                        preds_logits_sigmoidee = self.sw_inferer(inputs=imgs, network=self.model)
                        
                        # SIGMOID + BINARY THRESHOLDING NEEDED IN METRICS
                        preds = (preds_logits_sigmoidee >
                                 self.threshold).float()

                        ##### GPU METRICS #####
                        for metric in self.gpu_metrics:
                            name = metric.__class__.__name__
                            pbar.set_postfix_str(f"Metric: {name}")
                            value = metric(preds, masks)
                            results[name].append(value)

                        ##### CPU METRICS #####
                        preds_cpu = preds.detach().cpu().numpy()
                        masks_cpu = masks.detach().cpu().numpy()

                        pred_list = [preds_cpu[i]
                                     for i in range(len(preds_cpu))]
                        mask_list = [masks_cpu[i]
                                     for i in range(len(masks_cpu))]

                        for i in range(len(pred_list)):
                            pred_list[i] = np.expand_dims(pred_list[i], axis=0)
                            mask_list[i] = np.expand_dims(mask_list[i], axis=0)

                        all_futs = []
                        for metric in self.cpu_metrics:
                            name = metric.__class__.__name__
                            futs = self.client.map(
                                metric, pred_list, mask_list)
                            all_futs.extend([(name, f) for f in futs])

                        fut_only = [f for _, f in all_futs]
                        pbar.set_postfix_str(
                            f"Gathering {len(fut_only)} futures")
                        vals = self.client.gather(fut_only)

                        results_by_metric = defaultdict(list)
                        for (name, _), v in zip(all_futs, vals):
                            results_by_metric[name].append(v)

                        for name, vs in results_by_metric.items():
                            if isinstance(vs[0], (int, float, np.floating)):
                                mean_val = float(sum(vs) / len(vs))
                                results[name].append(mean_val)
                            elif isinstance(vs[0], dict):
                                keys = vs[0].keys()
                                mean_dict = {
                                    k: float(np.mean([v[k] for v in vs])) for k in keys}
                                for k in keys:
                                    colname = f"{name}_{k}"
                                    results.setdefault(
                                        colname, []).append(mean_dict[k])
                            else:
                                raise ValueError(
                                    f"Type de retour inattendu pour la métrique {name}: {type(vs[0])}")

                        if save_first_pred and self.tb_logger:
                            img_to_log = imgs
                            if imgs.shape[1] == 1:
                                img_to_log = imgs.repeat(1, 3, 1, 1)                                
                            self.tb_logger.log_image(
                                "Image", img_to_log, global_step=self.epoch)
                            self.tb_logger.log_image(
                                "Mask", masks * 255, global_step=self.epoch
                            )  # Multiplier par 255 pour visualiser en noir et blanc
                            self.tb_logger.log_image(
                                "Prediction", preds * 255, global_step=self.epoch
                            )  # Multiplier par 255 pour visualiser en noir et blanc
                            save_first_pred = False
            except Exception as e:
                if self.logger:
                    self.logger.error(
                        f"[Evaluator] Error during evaluation: {e}")
                else:
                    print(f"[Evaluator] Error during evaluation: {e}")
                return {}

        print(results)
        mean_results = {}
        for name in results:
            mean_results[name] = torch.mean(torch.tensor(results[name])).item()
        return mean_results
