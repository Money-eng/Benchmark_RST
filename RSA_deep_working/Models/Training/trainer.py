# Training/trainer.py

import os
import torch
from logging import Logger
from tqdm import tqdm
from utils.logger import TensorboardLogger
from utils.misc import get_device

from .evaluator import Evaluator


class Trainer:
    def __init__(
            self,
            model: torch.nn.Module,
            train_loader: torch.utils.data.DataLoader,
            criterion: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            config: dict,
            evaluator: Evaluator,
            logger: Logger = None,
            tb_logger: TensorboardLogger = None,
            device: torch.device = None,
    ):
        """
        - model : le réseau de neurones à entraîner
        - train_loader : DataLoader du jeu d'entraînement
        - criterion : objet torch.nn.Module (ex. DiceLoss, BCEDiceLoss, etc.)
        - optimizer : ex. torch.optim.Adam
        - config : dictionnaire complet issu de config.yml
        - evaluator : instance de Training.evaluator.Evaluator
        - logger : logger Python standard (info, warning, etc.)
        - tb_logger : wrapper TensorBoard (log_scalar, log_image, etc.)
        - device : torch.device("cuda") ou ("cpu")
        """
        self.model = model
        self.train_loader = train_loader
        self.criterion = criterion
        self.optimizer = optimizer

        self.epochs = int(config["training"]["epochs"])
        self.checkpoint_dir = config["training"]["checkpoint_dir"]
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.evaluator = evaluator
        self.device = (
            device
            if device is not None
            else get_device(preferred=config["training"].get("device", "cuda"))
        )

        self.logger = logger
        self.tb_logger = tb_logger

        if self.logger:
            self.logger.info(f"[Trainer] Device : {self.device}")
        else:
            print(f"[Trainer] Device : {self.device}")

        # On envoie le modèle et la loss sur l'appareil
        self.model.to(self.device)
        try:
            self.criterion.to(self.device)
        except:
            pass

    def train(self):
        """
        Boucle d'entraînement principale, avec :
        - forward + backward / mise à jour des poids
        - calcul de la loss au *batch* et à la fin de chaque epoch
        - évaluation sur validation + TensorBoard + sauvegarde du meilleur modèle
        """
        if self.logger:
            self.logger.info(
                f"[Trainer] Démarrage de l'entraînement pour {self.epochs} epochs"
            )
        else:
            print(f"[Trainer] Démarrage de l'entraînement pour {self.epochs} epochs")

        best_metric_val = {}
        global_step = 0

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            epoch_loss = 0.0

            pbar = tqdm(
                self.train_loader,
                desc=f"Epoch {epoch}/{self.epochs} [Train]",
                leave=False,
                dynamic_ncols=True,
            )
            for imgs, masks, _, _ in pbar:
                imgs, masks = imgs.to(self.device), masks.to(self.device)
                preds_logits_sigmoidee = self.model(imgs)
                loss = self.criterion(preds_logits_sigmoidee, masks) # need for sigmoid !!! (in loss implementation)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                pbar.set_postfix({"batch_loss": f"{loss.item():.4f}"})

                if self.tb_logger:
                    self.tb_logger.log_scalar(
                        "train/batch_loss", loss.item(), global_step
                    )

                global_step += 1

            avg_epoch_loss = epoch_loss / len(self.train_loader)

            if self.logger:
                self.logger.info(
                    f"[Trainer] Epoch {epoch}/{self.epochs} | Train Loss: {avg_epoch_loss:.4f}"
                )
            else:
                print(
                    f"[Trainer] Epoch {epoch}/{self.epochs} | Train Loss: {avg_epoch_loss:.4f}"
                )

            if self.tb_logger:
                self.tb_logger.log_scalar("train/epoch_loss", avg_epoch_loss, epoch)

            val_results = self.evaluator.evaluate(on_test=False, last_loss_value=avg_epoch_loss)

            val_str = ", ".join(f"{k}: {v:.4f}" for k, v in val_results.items())
            if self.logger:
                self.logger.info(
                    f"[EVALUATOR] Epoch {epoch}/{self.epochs} | Values : {val_str}"
                )
            else:
                print(f"[EVALUATOR] Epoch {epoch}/{self.epochs} | Values : {val_str}")

            if self.tb_logger:
                for metric_name, metric_val in val_results.items():
                    self.tb_logger.log_scalar(f"val/{metric_name}", metric_val, epoch)

            if not best_metric_val:
                for metric_name, metric_val in val_results.items():
                    best_metric_val[metric_name] = metric_val
                    self._save_checkpoint(epoch, metric_name, metric_val)
            else:
                for metric_name, metric_val in val_results.items():
                    if (
                            metric_name not in best_metric_val
                            or metric_val > best_metric_val[metric_name]
                    ):
                        best_metric_val[metric_name] = metric_val
                        self._save_checkpoint(epoch, metric_name, metric_val)

        if self.logger:
            self.logger.info("[Trainer] Entraînement terminé.")
        else:
            print("[Trainer] Entraînement terminé.")

    def _save_checkpoint(self, epoch: int, metric_name: str, metric_val: float):
        """
        Sauvergarde du state_dict du modèle dans checkpoint_dir.
        On ajoute epoch et valeur de métrique dans le nom de fichier.
        """
        # remove older checkpoint for the same metric
        
        filename = f"{self.model.__class__.__name__}_{metric_name}_epoch{epoch:03d}_{metric_val:.4f}.pth"
        filepath = os.path.join(self.checkpoint_dir, filename)
        torch.save(self.model.state_dict(), filepath)
        if self.logger:
            self.logger.info(f"[Trainer] Checkpoint sauvegardé : {filepath}")
        else:
            print(f"[Trainer] Checkpoint sauvegardé : {filepath}")
