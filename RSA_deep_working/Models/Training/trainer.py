# Training/trainer.py

import os
import torch
from logging import Logger
from tqdm import tqdm
from utils.logger import TensorboardLogger
from utils.misc import get_device
from gc import collect

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
            checkpoint_dir: str = "checkpoint_dir",
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
        self.epochs_btw_eval = int(
            config["training"].get("epochs_btw_eval", 10))
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.evaluator = evaluator
        self.early_stopper = EarlyStopping(patience=int(config['training']["early_stopping"]['patience']), 
                                           metric_name=config['training']["early_stopping"]['metric'],
                                           delta=float(config['training']["early_stopping"]['delta']))
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

        if config["training"]["lr_scheduler"]["name"] == "ReduceLROnPlateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=config['training']['lr_scheduler']['mode'],
                factor=config['training']['lr_scheduler']['factor'],
                patience=config['training']['lr_scheduler']['patience']
            )
        elif config["training"]["lr_scheduler"]["name"] == "StepLR":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=config['training']['lr_scheduler']['step_size'],
                gamma=config['training']['lr_scheduler']['gamma']
            )
        else:
            self.scheduler = None

        self.model.to(self.device)
        try:
            self.criterion.to(self.device)
        except:
            pass

        self.scaler = torch.GradScaler(device=self.device)
        self.logger.info('[Trainer] Initialisation du Trainer terminée.')
        
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
            print(
                f"[Trainer] Démarrage de l'entraînement pour {self.epochs} epochs")

        best_metric_val = {}
        batch_step = 0
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            epoch_loss = 0.0
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.epochs} [Train]", leave=False, dynamic_ncols=True)
            for imgs, masks, *_ in pbar:
                imgs  = imgs.to(self.device)
                masks = masks.to(self.device)
                self.optimizer.zero_grad()
                with torch.autocast(device_type=self.device.type):
                    preds = self.model(imgs)
                    loss = self.criterion(preds, masks)

                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward() # 1. scale & backward
                self.scaler.unscale_(self.optimizer)  # 2. unscale gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0, norm_type=2) # 3. clip gradients
                self.scaler.step(self.optimizer) # 4. step optimizer
                self.scaler.update() # 5. update scaler

                epoch_loss += loss.item()
                pbar.set_postfix({"batch_loss": f"{loss.item():.4f}"})

                if self.tb_logger:
                    self.tb_logger.log_scalar("train/batch_loss", loss.item(), batch_step)
                batch_step += 1

            avg_epoch_loss = epoch_loss / len(self.train_loader)

            if self.logger:
                self.logger.info(
                    f"[Trainer] Epoch {epoch}/{self.epochs} | Train Loss: {avg_epoch_loss:.4f} | Learning Rate: {self.optimizer.param_groups[0]['lr']}"
                )
            else:
                print(
                    f"[Trainer] Epoch {epoch}/{self.epochs} | Train Loss: {avg_epoch_loss:.4f} | Learning Rate: {self.optimizer.param_groups[0]['lr']}"
                )

            if self.tb_logger:
                self.tb_logger.log_scalar(
                    "train/epoch_loss", avg_epoch_loss, epoch)

            # free memory for next epoch
            collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            
            ### EVALUATION ###
            if epoch % self.epochs_btw_eval == 0:
                val_results = self.evaluator.evaluate(on_test=False, last_loss_value=avg_epoch_loss)
                # free memory after evaluation
                collect()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                val_str = ", ".join(f"{k}: {v:.4f}" for k,
                                    v in val_results.items())
                if self.logger:
                    self.logger.info(
                        f"[EVALUATOR] Epoch {epoch}/{self.epochs} | Values : {val_str}"
                    )
                else:
                    print(
                        f"[EVALUATOR] Epoch {epoch}/{self.epochs} | Values : {val_str}")

                if self.tb_logger:
                    for metric_name, metric_val in val_results.items():
                        self.tb_logger.log_scalar(
                            f"val/{metric_name}", metric_val, epoch)

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
                    self.logger.info(
                        f"[Trainer] Epoch {epoch}/{self.epochs} | Best Metric Values: {best_metric_val}"
                    )
                else:
                    print(
                        f"[Trainer] Epoch {epoch}/{self.epochs} | Best Metric Values: {best_metric_val}"
                    )

                self.early_stopper(val_results)
                if self.early_stopper.early_stop:
                    if self.logger:
                        self.logger.info(
                            f"[Trainer] Early stopping triggered at epoch {epoch}."
                        )
                    else:
                        print(
                            f"[Trainer] Early stopping triggered at epoch {epoch}."
                        )
                    break
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
        for file in os.listdir(self.checkpoint_dir):
            if file.startswith(f"{self.model.__class__.__name__}_{metric_name}_epoch"):
                os.remove(os.path.join(self.checkpoint_dir, file))
        filename = f"{self.model.__class__.__name__}_{metric_name}_epoch{epoch:03d}.pth"
        filepath = os.path.join(self.checkpoint_dir, filename)
        torch.save(self.model.state_dict(), filepath)
        if self.logger:
            self.logger.info(f"[Trainer] Checkpoint sauvegardé : {filepath}")
        else:
            print(f"[Trainer] Checkpoint sauvegardé : {filepath}")
            
class EarlyStopping:
    def __init__(self, patience, metric_name="f1_score", delta=0.0):
        self.patience = patience
        self.metric_name = metric_name
        self.delta = delta
        self.last_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_score: dict):
        current_score = val_score.get(self.metric_name, None)
        if current_score is None:
            return False
        elif self.last_score is None:
            self.last_score = current_score
            return False
        
        # if L2 distance is less than delta, consider it as no improvement
        elif abs(current_score - self.last_score) < self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        else:
            self.last_score = current_score
            self.counter = 0

        return False