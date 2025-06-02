import os
import torch
from tqdm import tqdm
from .evaluator import Evaluator  

class Trainer:
    def __init__(self, model: torch.nn.Module, train_loader: torch.utils.data.DataLoader, criterion: torch.nn.Module, optimizer: torch.optim.Optimizer, config: dict, evaluator: Evaluator):
        self.model = model
        self.train_loader = train_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.epochs = config['training']['epochs']
        self.checkpoint_dir = config['training']['checkpoint_dir']
        self.evaluator = evaluator  # instance d'Evaluator (val/test)
        self.device = config['training']['device']

    def train(self):
        best_metric = -float('inf')
        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs} [Training]", dynamic_ncols=True, leave=False)
            for imgs, masks in pbar:
                imgs, masks = imgs.to(self.device), masks.to(self.device)
                preds = self.model(imgs)
                loss = self.criterion(preds, masks)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
                # Display the current loss
                pbar.set_postfix(loss=loss.item())
            avg_loss = epoch_loss / len(self.train_loader)
            
            # Evaluate on validation set
            val_results = self.evaluator.evaluate(epoch=epoch+1, total_epochs=self.epochs)
            val_metrics_str = ", ".join(f"{k}: {v:.4f}" for k, v in val_results.items())
            print(f"\nEpoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.4f}, Val: {val_metrics_str}")

            first_metric = list(val_results.values())[0]
            if first_metric > best_metric:
                best_metric = first_metric
                self.save_checkpoint()

    def save_checkpoint(self):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        path = os.path.join(self.checkpoint_dir, f"{self.model.__class__.__name__}_checkpoint.pth")
        torch.save(self.model.state_dict(), path)
        print(f"Checkpoint saved to {path}")
