import os
import copy
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from Evaluation.evaluation import evaluate_segmentation_on_loader
import tqdm

# instanciation dataset, loaders, modèles, optimizers, etc.

import copy

def train_model(model: torch.nn.Module, 
                train_loader: torch.utils.data.DataLoader, 
                val_loader: torch.utils.data.DataLoader, 
                criterion: torch.nn.Module,
                optimizer: torch.optim.Optimizer,
                all_metrics: dict, # metrics -> {cpu: [metric1, metric2], gpu: [metric3, metric4]}
                num_epochs: int = 100,
                scheduler: torch.optim.lr_scheduler._LRScheduler = None,
                device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                writer: SummaryWriter = None,
                save_path: str = '/home/loai/Documents/code/RSMLExtraction/RSA_deep_working/Models/Checkpoints'
                ):
    
    torch.cuda.empty_cache()
    model.train()

    best_loss = np.inf
    best_epoch = 0
    # dict : {cpu: {metric1: score1, metric2: score2}, gpu: {metric3: score3, metric4: score4}}
    best_metric_scores = {'cpu': {}, 'gpu': {}} # store the best metric scores for each metric

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, masks, _, _ in tqdm(train_loader, 
                                        desc=f"Epoch {epoch+1}/{num_epochs}"
                                        ):
            
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)

        # write the loss to tensorboard
        if writer:
            writer.add_scalar('Loss/train', epoch_loss, epoch)

        # save current model state
        if epoch % 50 == 0:
            torch.save(model.state_dict(), os.path.join(save_path, f'model_{epoch+1}.pth'))
        
        model.eval()
        val_scores = {}
        with torch.no_grad():
            val_scores = evaluate_segmentation_on_loader( # dict : {cpu: {metric1: score1, metric2: score2}, gpu: {metric3: score3, metric4: score4}}
                model,
                val_loader,
                metrics=all_metrics,
                threshold=0.5,
                writer=writer,
                global_step=epoch,
                device=device
            )
            # for each metric, we save the best model and the best score
            for device in val_scores.keys():
                for metric_name, score in val_scores[device].items():
                    if metric_name not in best_metric_scores[device]:
                        best_metric_scores[device][metric_name] = score
                    elif score > best_metric_scores[device][metric_name]:
                        best_metric_scores[device][metric_name] = score
                        # save the model
                        torch.save(model.state_dict(), os.path.join(save_path, f'best_model_{metric_name}.pth'))
                        print(f"Best model saved for {metric_name} on {device} with score {score:.4f}")
                        
    print(f"Entraînement terminé à l'époque {epoch+1}, meilleure loss : {best_loss:.4f} à l'époque {best_epoch+1}")
    return model

def train_and_evaluate(model: torch.nn.Module,
                       loss_function: torch.nn.Module,
                       optimizer: torch.optim.Optimizer,
                       train_loader: torch.utils.data.DataLoader,
                       val_loader: torch.utils.data.DataLoader,
                       test_loader: torch.utils.data.DataLoader,
                       writer: SummaryWriter,
                       writer_name: str,
                       metrics: dict, # metrics -> {cpu: [metric1, metric2], gpu: [metric3, metric4]}
                       num_epochs: int = 100, 
                       scheduler: torch.optim.lr_scheduler._LRScheduler = None,
                       device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    # TensorBoard writer
    writer = SummaryWriter(log_dir=f"runs/Unet_{writer_name}")
    
    # Training
    model = train_model(
        model,
        train_loader,
        val_loader,
        loss_function,
        optimizer,
        num_epochs=num_epochs,
        scheduler=scheduler,
        device=device,
        writer=writer
    )
    writer.close()
    
    writer_test = SummaryWriter(log_dir=f"runs/Test_Unet_{writer_name}")
    # Evaluation on validation set
    evaluate_segmentation_on_loader(
        model,
        test_loader, # Test loader
        metrics=metrics,
        threshold=0.5,
        writer=writer,
        global_step=num_epochs,
        device=device
    )
    writer_test.close()
    
    return model

