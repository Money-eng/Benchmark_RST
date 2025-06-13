# Training/evaluator.py

import os
import torch
from logging import Logger
from tqdm import tqdm
from utils.launch_RST import process_date_map
from utils.logger import TensorboardLogger
from monai.inferers import SlidingWindowInfererAdapt
import concurrent.futures


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
        epoch: int = 0,
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
        executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=len(self.cpu_metrics) or 32
        )
        with torch.no_grad():
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

                    preds_logits_sigmoidee = self.sw_inferer(
                        inputs=imgs, network=self.model
                    )  # SIGMOID + BINARY THRESHOLDING NEEDED IN METRICS
                    preds = (preds_logits_sigmoidee > self.threshold).float()

                    for metric in self.gpu_metrics:
                        name = metric.__class__.__name__
                        pbar.set_postfix_str(f"Metric: {name}")
                        value = metric(preds, masks)
                        results[name].append(value)

                    preds_cpu = preds.detach().cpu().numpy()
                    masks_cpu = masks.detach().cpu().numpy()
                    for metric in self.cpu_metrics:
                        name = metric.__class__.__name__
                        pbar.set_postfix_str(f"Metric: {name}")
                        if (
                            name == "PeristenceBottleneck"
                            or name == "PeristenceWasserstein"
                        ):
                            value_cc, value_holes = metric(preds_cpu, masks_cpu)
                            name_cc = f"{name}_cc"
                            name_holes = f"{name}_holes"
                            if name_cc not in results:
                                results.setdefault(name_cc, [])
                                results.setdefault(name_holes, [])
                            results[name_cc].append(value_cc)
                            results[name_holes].append(value_holes)
                        else:
                            value = metric(preds_cpu, masks_cpu)
                            results[name].append(value)

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

        mean_results = {}
        for name in results:
            mean_results[name] = torch.mean(torch.tensor(results[name])).item()
        return mean_results
