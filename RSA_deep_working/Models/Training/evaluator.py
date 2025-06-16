# Training/evaluator.py

import os
import torch
from logging import Logger
from tqdm import tqdm
from utils.launch_RST import process_date_map
from utils.logger import TensorboardLogger
from monai.inferers import SlidingWindowInfererAdapt
import numpy as np
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
        metrics: dict,  # {"gpu": [Dice(), ...], "cpu": [Connectivity(), ...]}
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
            roi_size=(512, 512),  # La taille du patch à utiliser pour l'inférence
            sw_batch_size=4,  # Nombre de patchs à traiter en même temps (ajuste selon la VRAM)
            overlap=0.25,  # Recouvrement entre les fenêtres, 0.25=25%
            mode="constant",  # Moyenne sur les recouvrements (classique)
        )
        
        self.cluster = LocalCluster(
            n_workers=os.cpu_count(),
            threads_per_worker=1,
            processes=True,
            memory_limit="8GB",
        )
        self.client = Client(self.cluster)

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
                if last_loss_value < 0.4:
                    print("Processing the whole series")
                    for ts_imgs, masks, _, mtgs in tqdm(
                        data_loader_series,
                        desc="Evaluating whole series",
                        leave=False,
                        dynamic_ncols=True,
                    ):
                        ts_imgs = ts_imgs.to(self.device)
                        masks = masks.to(self.device)

                        preds_logits_sigmoidee = self.sw_inferer(
                            inputs=imgs, network=self.model
                        )

                        mtg_gt, mtg_pred = process_date_map(
                            mtgs, preds, jar_path=self.jar_path
                        )
                
                with tqdm(dataloader, desc="Evaluating batch per batch", leave=False, dynamic_ncols=True) as pbar:
                    for imgs, masks, _, _ in pbar:
                        imgs = imgs.to(self.device)
                        masks = masks.to(self.device)

                        preds_logits_sigmoidee = self.sw_inferer(inputs=imgs, network=self.model)  
                        # SIGMOID + BINARY THRESHOLDING NEEDED IN METRICS
                        preds = (preds_logits_sigmoidee > self.threshold).float()

                        ##### GPU METRICS #####
                        for metric in self.gpu_metrics:
                            name = metric.__class__.__name__
                            pbar.set_postfix_str(f"Metric: {name}")
                            value = metric(preds, masks)
                            results[name].append(value)

                        ##### CPU METRICS #####
                        preds_cpu = preds.detach().cpu().numpy()
                        masks_cpu = masks.detach().cpu().numpy()

                        pred_list = [preds_cpu[i] for i in range(len(preds_cpu))]
                        mask_list = [masks_cpu[i] for i in range(len(masks_cpu))]
                        
                        # unsqueeze pour avoir la forme (B, 1, H, W) si nécessaire
                        for i in range(len(pred_list)):
                            pred_list[i] = np.expand_dims(pred_list[i], axis=0)
                            mask_list[i] = np.expand_dims(mask_list[i], axis=0)

                        # 1) créer tous les futures sans jamais gather
                        all_futs = []                     # liste de tuples (metric_name, Future)
                        for metric in self.cpu_metrics:
                            name = metric.__class__.__name__
                            # map renvoie une liste de Future
                            futs = self.client.map(metric, pred_list, mask_list)
                            all_futs.extend([(name, f) for f in futs])

                        # 2) rassembler tous les résultats en une passe
                        fut_only = [f for _, f in all_futs]
                        pbar.set_postfix_str(f"Gathering {len(fut_only)} futures")
                        vals = self.client.gather(fut_only)   # liste de valeurs dans le même ordre que fut_only

                        # 3) regrouper par métrique
                        results_by_metric = defaultdict(list)
                        for (name, _), v in zip(all_futs, vals):
                            results_by_metric[name].append(v)

                        # 4) pour chaque métrique calculer la moyenne et l’ajouter à results
                        for name, vs in results_by_metric.items():
                            if isinstance(vs[0], (int, float, np.floating)):
                                mean_val = float(sum(vs) / len(vs))
                                results[name].append(mean_val)
                            elif isinstance(vs[0], dict):
                                # même logique que vous aviez pour les métriques retournant des dicts
                                keys = vs[0].keys()
                                mean_dict = {k: float(np.mean([v[k] for v in vs])) for k in keys}
                                for k in keys:
                                    colname = f"{name}_{k}"
                                    results.setdefault(colname, []).append(mean_dict[k])
                            else:
                                raise ValueError(f"Type de retour inattendu pour la métrique {name}: {type(vs[0])}")


                        if save_first_pred and self.tb_logger:
                            # Sauvegarde la première image, masque et prédiction dans TensorBoard
                            # duplicate chanel on img
                            self.tb_logger.log_image("Image", imgs, global_step=self.epoch)
                            self.tb_logger.log_image(
                                "Mask", masks * 255, global_step=self.epoch
                            )  # Multiplier par 255 pour visualiser en noir et blanc
                            self.tb_logger.log_image(
                                "Prediction", preds * 255, global_step=self.epoch
                            )  # Multiplier par 255 pour visualiser en noir et blanc
                            save_first_pred = False
            except Exception as e:
                if self.logger:
                    self.logger.error(f"[Evaluator] Error during evaluation: {e}")
                else:
                    print(f"[Evaluator] Error during evaluation: {e}")
                return {}

        print(results)
        mean_results = {}
        for name in results:
            mean_results[name] = torch.mean(torch.tensor(results[name])).item()
        return mean_results
