# Training/evaluator.py

import torch
from tqdm import tqdm
from logging import Logger


class Evaluator:
    def __init__(
        self,
        model: torch.nn.Module,
        val_dataloader: torch.utils.data.DataLoader,
        test_dataloader: torch.utils.data.DataLoader,
        val_series_dataloader: torch.utils.data.DataLoader,
        test_series_dataloader: torch.utils.data.DataLoader,
        metrics: dict,   # {"gpu": [Dice(), ...], "cpu": [Connectivity(), ...]}
        device: torch.device,
        logger: Logger = None,
        tb_logger=None,
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

        self.logger = logger
        self.tb_logger = tb_logger

        if self.logger:
            self.logger.info("[Evaluator] Initialisation terminée.")
        else:
            print("[Evaluator] Initialisation terminée.")

    def evaluate(self, on_test: bool = False, on_serie: bool = False) -> dict:
        """
        Calcule toutes les métriques pour l'ensemble des batches de validation (ou test si on_test=True).
        Retourne un dict { metric_name: moyenne_sur_tous_les_batches }.
        """
        self.model.eval()

        # Choix du dataloader (on ne gère pas 'on_serie' ici, mais vous pouvez l'ajouter si besoin)
        dataloader = self.test_dataloader if on_test else self.val_dataloader

        # Initialisation des vecteurs d'acumulation
        sums = {}
        counts = {}
        for metric in (self.gpu_metrics + self.cpu_metrics):
            name = metric.__class__.__name__
            sums[name] = 0.0
            counts[name] = 0

        with torch.no_grad():
            for imgs, masks, time, mtg in tqdm(dataloader, desc="Evaluating", leave=False, dynamic_ncols=True):
                imgs = imgs.to(self.device)
                masks = masks.to(self.device)

                # 1) Forward sur tout le batch
                preds = self.model(imgs)  # shape [B, 1, H, W]

                # 2) Calcul des métriques GPU (elles sont vectorisées)
                for metric in self.gpu_metrics:
                    name = metric.__class__.__name__
                    value = metric(preds, masks)  # un float = moyenne sur le batch
                    sums[name] += value
                    counts[name] += 1

                # 3) Transfert des tenseurs sur CPU pour les métriques CPU
                preds_cpu = preds.detach().cpu()
                masks_cpu = masks.detach().cpu()
                for metric in self.cpu_metrics:
                    name = metric.__class__.__name__
                    value = metric(preds_cpu, masks_cpu)
                    sums[name] += value
                    counts[name] += 1

        # 4) Moyennage
        results = {}
        for name in sums:
            if counts[name] > 0:
                results[name] = sums[name] / counts[name]
            else:
                results[name] = 0.0

        return results
