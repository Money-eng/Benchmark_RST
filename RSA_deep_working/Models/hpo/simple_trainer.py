"""
SimpleTrainer : une classe d'entraînement *minimaliste* et bien commentée
pour un modèle de segmentation (ou n'importe quel modèle PyTorch).

Objectifs :
- rester lisible et facile à maintenir ;
- exposer les étapes clés (train / validation) ;
- optionnellement, rapporter la métrique à Optuna pour permettre le pruning.

Ce fichier NE dépend que de PyTorch et n'exige aucune structure projet.
Tu peux l'utiliser tel quel avec tes DataLoaders et ton modèle.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

try:  # Import facultatif (uniquement si utilisé avec Optuna)
    import optuna
except Exception:  # pragma: no cover
    optuna = None  # type: ignore


@dataclass
class TrainResult:
    """Résultat récapitulatif d'un entraînement court ou long."""
    best_val_loss: float
    best_epoch: int
    history: Dict[str, list]  # {"train_loss": [...], "val_loss": [...]} (éventuel)


class SimpleTrainer:
    """Boucle d'entraînement PyTorch simple et claire.

    Paramètres
    ----------
    model : nn.Module
        Le modèle à entraîner.
    optimizer : Optimizer
        L'optimiseur déjà construit (Adam/SGD...).
    criterion : nn.Module
        La fonction de perte (ex : BCEWithLogitsLoss, Dice+CE...).
    device : torch.device
        CPU ou CUDA.
    train_loader : DataLoader
        Batches d'entraînement.
    val_loader : DataLoader
        Batches de validation (pour mesurer val_loss et faire du pruning si souhaité).
    num_epochs : int
        Nombre d'époques à effectuer.
    eval_every : int
        Fréquence d'évaluation (en époques). 1 = évaluer à chaque époque.
    trial : optuna.Trial | None
        Si fourni, on reporte la val_loss à Optuna pour activer le pruning.
    patience : int | None
        Early stopping basique : stop après 'patience' évaluations sans amélioration.
    """

    def __init__(
            self,
            model: nn.Module,
            optimizer: Optimizer,
            criterion: nn.Module,
            metric: nn.Module,
            device: torch.device,
            train_loader: DataLoader,
            val_loader: DataLoader,
            *,
            num_epochs: int = 10,
            eval_every: int = 1,
            trial: Optional["optuna.Trial"] = None,
            patience: Optional[int] = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.eval_metric = metric
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = int(num_epochs)
        self.eval_every = max(1, int(eval_every))
        self.trial = trial
        self.patience = patience

        self.model.to(self.device)
        try:
            self.criterion.to(self.device)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # API publique
    # ------------------------------------------------------------------
    def train(self) -> TrainResult:
        """Exécute la boucle d'entraînement + validation.

        Retourne la meilleure val_loss observée et un petit historique.
        """
        history = {"train_loss": [], "val_loss": []}
        best_val = -float("inf")
        best_epoch = -1
        no_improve = 0  # compteur early-stopping

        for epoch in range(1, self.num_epochs + 1):
            # ------------------- phase TRAIN -------------------
            self.model.train()
            running = 0.0
            nbatches = 0

            for imgs, masks, *_ in self.train_loader:
                imgs = imgs.to(self.device)
                masks = masks.to(self.device).float()

                self.optimizer.zero_grad(set_to_none=True)
                preds = self.model(imgs).float()
                loss = self.criterion(preds, masks)
                loss.backward()
                self.optimizer.step()

                running += float(loss.item())
                nbatches += 1

            train_loss = running / max(1, nbatches)
            history["train_loss"].append(train_loss)

            # ------------------- phase VAL (selon fréquence) ----
            if epoch % self.eval_every == 0:
                val_value = self._validate()
                history["val_loss"].append(val_value)

                # Si utilisé dans une étude Optuna → report + prune
                if self.trial is not None and optuna is not None:
                    self.trial.report(val_value, step=epoch)
                    if self.trial.should_prune():
                        raise optuna.TrialPruned(f"Pruned at epoch {epoch}")

                # Suivi du meilleur modèle
                if val_value > best_val + 1e-8:
                    best_val = float(val_value)
                    best_epoch = epoch
                    no_improve = 0
                else:
                    no_improve += 1
                    if self.patience is not None and no_improve >= self.patience:
                        break  # early stop

        return TrainResult(best_val_loss=best_val, best_epoch=best_epoch, history=history)

    # ------------------------------------------------------------------
    # Détails internes
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _validate(self) -> float:
        """Calcule la loss de validation moyenne (aucune métrique avancée ici)."""
        self.model.eval()
        running = 0.0
        nbatches = 0
        for imgs, masks, *_ in self.val_loader:
            imgs = imgs.to(self.device)
            masks = masks.to(self.device).float()
            preds = self.model(imgs).float()
            value = self.eval_metric(preds, masks)
            if isinstance(value, torch.Tensor):
                value = float(value.detach().mean().item())
            else:
                value = float(value)
            running += float(value)
            nbatches += 1
        return running / max(1, nbatches)
