"""training/trainer.py
Cleaned‑up, fully‑typed implementation of the training loop.
All public behaviour is preserved; only readability has been improved and
comments are now in English.
"""
from __future__ import annotations

import logging
import os
from gc import collect
from pathlib import Path
from typing import Dict, Optional

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, _LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.logger import TensorboardLogger
from utils.misc import SEED, get_device, set_seed

from .evaluator import Evaluator

# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------
set_seed(SEED)  # sets python, numpy and torch seeds


# -----------------------------------------------------------------------------
# Main Trainer
# -----------------------------------------------------------------------------


class Trainer:
    def __init__(
            self,
            model: nn.Module,
            train_loader: DataLoader,
            epochs: int,
            epochs_btw_eval: int,
            criterion: nn.Module,
            optimizer: Optimizer,
            config: dict,
            evaluator: Evaluator,
            logger: Optional[logging.Logger] = None,
            tb_logger: Optional[TensorboardLogger] = None,
            checkpoint_dir: str | os.PathLike | None = "checkpoint_dir",
            device: Optional[torch.device] = None,
            *,
            save_each_epoch: bool = True,
            do_evaluation: bool = True,
    ) -> None:

        # ------------------------------------------------------------------
        # Public attributes
        # ------------------------------------------------------------------
        self.model = model
        self.train_loader = train_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.evaluator = evaluator
        self.logger = logger or logging.getLogger(__name__)
        self.tb_logger = tb_logger

        # ------------------------------------------------------------------
        # Training schedule configuration
        # ------------------------------------------------------------------
        self.epochs = int(epochs or config["training"]["epochs"])
        self.epochs_btw_eval = int(
            epochs_btw_eval or config["training"].get("epochs_btw_eval", 10))
        self.save_each_epoch = save_each_epoch
        self.do_evaluation = do_evaluation

        # ------------------------------------------------------------------
        # Device & seed
        # ------------------------------------------------------------------
        self.device = device or get_device(
            preferred=config["training"].get("device", "cuda"))
        self.model.to(self.device)
        try:
            self.criterion.to(self.device)
        except Exception:
            # e.g. criterion does not implement .to()
            pass

        # ------------------------------------------------------------------
        # Learning‑rate scheduler
        # ------------------------------------------------------------------
        self.scheduler: Optional[_LRScheduler] = self._init_scheduler(config)

        # ------------------------------------------------------------------
        # Checkpointing
        # ------------------------------------------------------------------
        self.checkpoint_dir: Optional[Path] = None
        if checkpoint_dir is not None:
            self.checkpoint_dir = Path(checkpoint_dir)
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # ------------------------------------------------------------------
        # Early‑stopping configuration
        # ------------------------------------------------------------------
        es_cfg = config["training"]["early_stopping"]
        self.early_stopper = EarlyStopping(
            patience=int(es_cfg["patience"]),
            metric_name=es_cfg["metric"],
            delta=float(es_cfg["delta"]),
        )

        self._log(logging.INFO, "Device in use: %s", self.device)
        self._log(logging.INFO, "Trainer initialised successfully.")

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    def train(self) -> None:
        """Run the full training loop."""
        self._log(logging.INFO, "Starting training for %d epochs", self.epochs)

        best_metrics: Dict[str, float] = {}
        global_step = 0  # counts *batches*

        for epoch in range(1, self.epochs + 1):
            # ---------------------------- TRAIN PHASE ---------------------
            self.model.train()
            epoch_loss = 0.0

            pbar = tqdm(
                self.train_loader,
                desc=f"Epoch {epoch}/{self.epochs} [Train]",
                leave=False,
                dynamic_ncols=True,
            )

            for imgs, masks, *_ in pbar:
                imgs = imgs.to(self.device)
                masks = masks.to(self.device).float()

                # already passed through a sigmoid inside the model
                preds = self.model(imgs).float()
                loss = self.criterion(preds, masks)

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                pbar.set_postfix({"batch_loss": f"{loss.item():.6f}"})

                if self.tb_logger is not None:
                    self.tb_logger.log_scalar(
                        "train/batch_loss", loss.item(), global_step)
                global_step += 1

            # ------------------------ END OF EPOCH ------------------------
            avg_epoch_loss = epoch_loss / len(self.train_loader)
            self._after_epoch(avg_epoch_loss, epoch)

            # ------------------------ EVALUATION --------------------------
            if self.do_evaluation and epoch % self.epochs_btw_eval == 0:
                improved = self._run_validation(epoch, best_metrics)

                # Handle early‑stopping only after we have evaluation results.
                if self.early_stopper(improved):
                    self._log(logging.INFO,
                              "Early stopping triggered at epoch %d.", epoch)
                    break

        # Final clean‑up ----------------------------------------------------
        if self.do_evaluation:
            self.evaluator.done_evaluating()
        self._log(logging.INFO, "Training finished.")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _after_epoch(self, avg_epoch_loss: float, epoch: int) -> None:
        """Book-keeping tasks executed at the end of every *training* epoch."""
        # Step the scheduler *before* logging the learning rate (PyTorch style‑guide)
        if self.scheduler is not None:
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(avg_epoch_loss)
            else:
                self.scheduler.step()

        current_lr = self.optimizer.param_groups[0]["lr"]

        self._log(logging.INFO, "Epoch %d/%d | Train Loss: %.4f | LR: %.3e",
                  epoch, self.epochs, avg_epoch_loss, current_lr)

        if self.tb_logger is not None:
            self.tb_logger.log_scalar(
                "train/epoch_loss", avg_epoch_loss, epoch)
            self.tb_logger.log_scalar("train/lr", current_lr, epoch)

        if self.save_each_epoch and self.checkpoint_dir is not None:
            self._save_checkpoint_at_epoch(epoch)

    # ------------------------------------------------------------------
    def _run_validation(self, epoch: int, best_metrics: Dict[str, float]) -> Dict[str, float]:
        """Run evaluation on the *validation* set and manage checkpoints."""
        self.evaluator.epoch = epoch
        val_metrics = self.evaluator.evaluate(on_test=False)

        # Free GPU memory right after evaluation to reduce fragmentation.
        collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        metrics_str = ", ".join(
            f"{k}: {v:.4f}" for k, v in val_metrics.items())
        self._log(
            logging.INFO, "[EVALUATOR] Epoch %d/%d | %s", epoch, self.epochs, metrics_str)

        # TensorBoard ------------------------------------------------------
        if self.tb_logger is not None:
            for name, value in val_metrics.items():
                self.tb_logger.log_scalar(f"val/{name}", value, epoch)

        # Best‑model checkpointing ----------------------------------------
        if self.checkpoint_dir is not None:
            for name, val in val_metrics.items():
                is_better = _is_metric_better(
                    name, val, best_metrics.get(name), self.evaluator)
                if is_better:
                    best_metrics[name] = val
                    self._save_best_checkpoint(epoch, name)

        self._log(logging.INFO, "Best metrics so far: %s", best_metrics)
        return val_metrics

    # ------------------------------------------------------------------
    def _init_scheduler(self, cfg: dict) -> Optional[_LRScheduler]:
        """Return an LR scheduler instance configured from *cfg* (or *None*)."""
        sch_cfg = cfg["training"]["lr_scheduler"]
        name = sch_cfg.get("name")

        if name == "ReduceLROnPlateau":
            return ReduceLROnPlateau(
                self.optimizer,
                mode=sch_cfg["mode"],
                factor=sch_cfg["factor"],
                patience=sch_cfg["patience"],
            )
        if name == "StepLR":
            return StepLR(
                self.optimizer,
                step_size=sch_cfg["step_size"],
                gamma=sch_cfg["gamma"],
            )
        return None

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def _save_best_checkpoint(self, epoch: int, metric: str) -> None:
        """Save *best-so-far* model (one per metric)."""
        assert self.checkpoint_dir is not None  # mypy: ignore [assert‑never]

        # Remove older best checkpoint for this metric (if any)
        pattern = f"{self.model.__class__.__name__}_{metric}_epoch"
        for file in self.checkpoint_dir.glob(f"{pattern}*.pth"):
            file.unlink(missing_ok=True)

        filename = f"{self.model.__class__.__name__}_{metric}_epoch{epoch:03d}.pth"
        torch.save(self.model.state_dict(), self.checkpoint_dir / filename)
        self._log(logging.INFO,"[Trainer] Best-checkpoint saved: %s", filename)

    # ------------------------------------------------------------------
    def _save_checkpoint_at_epoch(self, epoch: int) -> None:
        """Save the model *state_dict* at the end of every epoch."""
        assert self.checkpoint_dir is not None  # mypy: ignore [assert‑never]

        epoch_dir = self.checkpoint_dir / "by_epochs"
        epoch_dir.mkdir(exist_ok=True)
        filename = f"{self.model.__class__.__name__}_epoch{epoch:03d}.pth"
        torch.save(self.model.state_dict(), epoch_dir / filename)
        self._log(logging.INFO,
                  "[Trainer] Epoch-checkpoint saved: %s", filename)

    # ------------------------------------------------------------------
    # Misc.
    # ------------------------------------------------------------------

    def _log(self, level: int, msg: str, *args, **kwargs) -> None:
        if self.logger is not None:
            self.logger.log(level, msg, *args, **kwargs)
        else:
            print(msg % args)


# -----------------------------------------------------------------------------
# Helper classes
# -----------------------------------------------------------------------------
class EarlyStopping:
    """Stop training when a monitored validation metric has stopped improving.

    Parameters
    ----------
    patience : int
        Number of *consecutive* evaluation rounds without improvement before we
        stop.
    metric_name : str, default="f1_score"
        Key in the metric dict returned by :py:meth:`Evaluator.evaluate` to
        monitor.
    delta : float, default=0.0
        Minimum **absolute** improvement required to reset the patience
        counter.  A smaller change is considered *no* improvement.
    """

    def __init__(self, patience: int, metric_name: str = "f1_score", delta: float = 0.0) -> None:
        self.patience = patience
        self.metric_name = metric_name
        self.delta = delta

        self._best: Optional[float] = None
        self._counter: int = 0
        self.early_stop: bool = False

    # ---------------------------------------------------------------------
    # Call interface – behaves like a function: ``early_stopper(metrics)``
    # ---------------------------------------------------------------------
    def __call__(self, metrics: Dict[str, float]) -> bool:  # noqa: D401 – imperative form is clearer.
        """Update internal state and decide whether to stop early.

        Returns
        -------
        bool
            *True* if the patience has been exceeded and training should stop.
        """
        current = metrics.get(self.metric_name)
        if current is None:
            return False  # metric not provided

        if self._best is None or current > self._best + self.delta:
            self._best = current
            self._counter = 0
            return False

        # No (significant) improvement → increment the counter
        self._counter += 1
        if self._counter >= self.patience:
            self.early_stop = True

        return self.early_stop


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
def _is_metric_better(name: str, new_val: float, best_val: Optional[float], evaluator: Evaluator) -> bool:
    """Return *True* if *new_val* is better than *best_val* for the given metric."""
    if best_val is None:
        return True

    if name in evaluator.cpu_metrics:
        return evaluator.cpu_metrics[name].is_better(best_val, new_val)
    
    if name in evaluator.gpu_metrics:
        return evaluator.gpu_metrics[name].is_better(best_val, new_val)


    # Fallback: maximise the metric.
    return new_val > best_val
