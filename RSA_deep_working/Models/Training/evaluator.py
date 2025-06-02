import torch
from tqdm import tqdm

class Evaluator:
    def __init__(self, model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, metrics: list, device: str = 'cuda'):
        self.model = model
        self.dataloader = dataloader
        self.metrics = metrics  # liste de fonctions métriques
        self.device = device

    def evaluate(self):
        self.model.eval()
        results = {m.__name__: 0.0 for m in self.metrics}
        count = 0

        with torch.no_grad():
            for imgs, masks in tqdm(self.dataloader, desc="Evaluating"):
                imgs = imgs.to(self.device)
                masks = masks.to(self.device)
                preds = self.model(imgs)
                for metric in self.metrics:
                    val = metric(preds, masks)
                    results[metric.__name__] += val.item()
                count += 1
        for key in results:
            results[key] /= count
        return results