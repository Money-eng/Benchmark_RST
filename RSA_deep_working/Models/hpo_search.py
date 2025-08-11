"""
HPOSearcher : une classe *séparée* et simple pour la recherche d'hyperparamètres
avec Optuna (TPE + ASHA), découplée de l'entraînement.

Idée :
- HPOSearcher ne sait pas entraîner par lui-même : il crée le modèle/optimiseur,
  puis délègue l'entraînement à SimpleTrainer (fichier séparé) en lui passant
  éventuellement le `trial` pour activer le pruning.
- On garde le **strict minimum** : learning_rate, weight_decay, optimizer.

Utilisation (exemple) :

    from hpo_search import HPOSearcher
    from simple_trainer import SimpleTrainer

    searcher = HPOSearcher(build_model, build_criterion)
    best = searcher.search(
        train_loader, val_loader,
        n_trials=100, epochs_per_trial=8,
        lr_bounds=(1e-4, 3e-2), wd_bounds=(1e-7, 1e-3),
        optimizers=("adamw", "adam", "sgd"),
        device=torch.device("cuda"),
    )
    print(best)

Ensuite, tu peux relancer un **entraînement final** long en recréant un modèle
avec ces hyperparamètres et en utilisant SimpleTrainer (sans Optuna cette fois).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Optional, Tuple

import optuna
import torch
from optuna.samplers import TPESampler
from optuna.pruners import SuccessiveHalvingPruner

from simple_trainer import SimpleTrainer


# -----------------------------------------------------------------------------
# Fonctions utilitaires minimales
# -----------------------------------------------------------------------------

def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class BestTrial:
    value: float
    params: Dict[str, object]


# -----------------------------------------------------------------------------
# Classe HPOSearcher
# -----------------------------------------------------------------------------
class HPOSearcher:
    """Recherche d'hyperparamètres minimaliste avec Optuna.

    Paramètres
    ----------
    build_model : Callable[[Dict], torch.nn.Module]
        Fabrique un modèle à partir d'une config *déjà fusionnée* (si nécessaire).
        Tu peux aussi ignorer la config et retourner un modèle fixe.
    build_criterion : Callable[[], torch.nn.Module]
        Fabrique la loss (ex : BCEWithLogitsLoss) ; sans arguments pour simplifier.
    make_optimizer : Callable[[Iterable, str, float, float], torch.optim.Optimizer] | None
        Fabrique l'optimiseur ; si None, on utilise une version interne simple.
    seed : int
        Graine pour reproductibilité de l'échantillonnage Optuna.
    study_storage : str | None
        Chemin SQLite de la study (ex: "sqlite:///study.db"). None => en mémoire.
    study_name : str | None
        Nom de l'étude (utile avec storage pour reprendre plus tard).
    """

    def __init__(
        self,
        build_model: Callable[[Dict], torch.nn.Module],
        build_criterion: Callable[[], torch.nn.Module],
        *,
        make_optimizer: Optional[Callable[[Iterable, str, float, float], torch.optim.Optimizer]] = None,
        seed: int = 42,
        study_storage: Optional[str] = None,
        study_name: Optional[str] = None,
    ) -> None:
        self.build_model = build_model
        self.build_criterion = build_criterion
        self.make_optimizer = make_optimizer
        self.seed = seed
        self.study_storage = study_storage
        self.study_name = study_name

    # ------------------------------- API publique ----------------------------
    def search(
        self,
        train_loader,
        val_loader,
        *,
        n_trials: int = 50,
        epochs_per_trial: int = 8,
        eval_every: int = 1,
        lr_bounds: Tuple[float, float] = (1e-4, 3e-2),
        wd_bounds: Tuple[float, float] = (1e-7, 1e-3),
        optimizers: Tuple[str, ...] = ("adamw", "adam"),
        device: Optional[torch.device] = None,
    ) -> BestTrial:
        """Lance une étude Optuna et retourne le meilleur essai (val_loss minimale)."""
        self.device = device or default_device()

        sampler = TPESampler(seed=self.seed, multivariate=True, group=True)
        pruner = SuccessiveHalvingPruner(reduction_factor=3, min_resource=1)

        study = optuna.create_study(
            direction="minimize",
            sampler=sampler,
            pruner=pruner,
            study_name=self.study_name,
            storage=self.study_storage,
            load_if_exists=bool(self.study_storage is not None),
        )

        # Prépare et fige l'objective avec les ressources externes (loaders)
        objective = self._make_objective(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs_per_trial=epochs_per_trial,
            eval_every=eval_every,
            lr_bounds=lr_bounds,
            wd_bounds=wd_bounds,
            optimizers=optimizers,
        )

        study.optimize(objective, n_trials=n_trials, gc_after_trial=True)

        return BestTrial(value=float(study.best_value), params=dict(study.best_params))

    # ----------------------------- Détails internes --------------------------
    def _make_objective(
        self,
        *,
        train_loader,
        val_loader,
        epochs_per_trial: int,
        eval_every: int,
        lr_bounds: Tuple[float, float],
        wd_bounds: Tuple[float, float],
        optimizers: Tuple[str, ...],
    ):
        def objective(trial: optuna.Trial) -> float:
            # 1) Échantillonnage des hyperparamètres (log pour LR et WD)
            opt_name = trial.suggest_categorical("optimizer", list(optimizers))
            lr = trial.suggest_float("learning_rate", lr_bounds[0], lr_bounds[1], log=True)
            wd = trial.suggest_float("weight_decay", wd_bounds[0], wd_bounds[1], log=True)

            # 2) Modèle + loss
            model = self.build_model({})  # passe une dict vide si non utilisée
            model = model.to(self.device)
            criterion = self.build_criterion().to(self.device)

            # 3) Optimiseur (basique et lisible)
            optimizer = self._build_optimizer(model.parameters(), opt_name, lr, wd)

            # 4) Entraînement court avec SimpleTrainer + pruning Optuna
            trainer = SimpleTrainer(
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                device=self.device,
                train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=epochs_per_trial,
                eval_every=eval_every,
                trial=trial,
                patience=None,
            )

            try:
                result = trainer.train()
                best_val = float(result.best_val_loss)
            except optuna.TrialPruned:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    raise optuna.TrialPruned("Pruned on CUDA OOM")
                raise
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            return best_val

        return objective

    # ------------------------------ Helpers ----------------------------------
    def _build_optimizer(self, params: Iterable, name: str, lr: float, wd: float):
        if self.make_optimizer is not None:
            return self.make_optimizer(params, name, lr, wd)
        name = name.lower()
        if name == "adamw":
            return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
        if name == "sgd":
            return torch.optim.SGD(params, lr=lr, weight_decay=wd, momentum=0.9, nesterov=True)
        return torch.optim.Adam(params, lr=lr, weight_decay=wd) 


if __name__ == "__main__":
    """Exemple minimal pour tester rapidement (à adapter à ton projet).

    Tu dois fournir :
    - un build_model() qui retourne un nn.Module
    - un build_criterion() qui retourne une loss
    - des DataLoaders train/val
    """
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    # Données jouet (binaire) : 1000 échantillons, 16 features
    X = torch.randn(1000, 16)
    y = (X.sum(dim=1) > 0).float().unsqueeze(1)
    ds = TensorDataset(X, y)
    train_loader = DataLoader(ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(ds, batch_size=64)

    def build_model(_cfg: Dict) -> nn.Module:
        return nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid())

    def build_criterion() -> nn.Module:
        return nn.BCELoss()

    searcher = HPOSearcher(build_model, build_criterion, study_storage=None)
    best = searcher.search(train_loader, val_loader, n_trials=20, epochs_per_trial=3)
    print("Best:", best)
