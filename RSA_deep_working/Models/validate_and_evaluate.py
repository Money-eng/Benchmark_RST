import os
import argparse
import yaml
import torch
import re

from pathlib import Path
from torch.nn import DataParallel
from tqdm import tqdm

# --- Imports locaux ---
from DataLoaders.dataloaders import create_dataloader
from DataLoaders.transforms import (
    get_train_img_transform_1,
    get_train_img_transform_2,
    get_train_img_transform_3,
    get__val_test_img_transform,
)
from Model import get_model
from Losses import get_loss
from Metrics import get_metrics
from Training.evaluator import Evaluator
from utils.logger import get_logger, TensorboardLogger
from utils.mask_of_interest import roi_fnc
from utils.misc import SEED, get_device, set_seed

import wandb

set_seed(SEED)

def extract_epoch_num(path):
    match = re.search(r'epoch(\d+)', os.path.basename(path))
    return int(match.group(1)) if match else -1

def load_config(cfg_path: Path | str) -> dict:
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)

def build_dataloaders(cfg: dict):
    patch_size = cfg["data"]["patch_size"]
    transforms = [
        get_train_img_transform_1(patch_size=patch_size),
        get_train_img_transform_2(patch_size=patch_size),
        get_train_img_transform_3(patch_size=patch_size),
        get__val_test_img_transform(),
    ]
    
    _, val_loader, test_loader = create_dataloader(
        base_directory=cfg["data"]["base_dir"],
        img_transforms=transforms,
        batch_size=cfg["data"].get("batch_size", 32),
    )
    return val_loader, test_loader

def load_weights_safely(model, weights_path, device):

    print(f"Chargement des poids depuis {weights_path}")
    checkpoint = torch.load(weights_path, map_location="cpu") 
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            name = k[7:] 
            new_state_dict[name] = v
        else:
            new_state_dict[k] = v
            
    try:
        model.load_state_dict(new_state_dict, strict=True)
    except RuntimeError as e:
        print(f"ERREUR de chargement : \n{e}")
        raise e
    
    return model

def run_inference_pass(cfg, weights_path, dataloader, device, run_name="Run 1"):
    print(f"\n--- Démarrage {run_name} ---")
    
    model = get_model(cfg["model"])
    model = load_weights_safely(model, weights_path, device)
    
    if torch.cuda.device_count() > 1:
        # On récupère la liste explicite des IDs disponibles [0, 1, 2...]
        device_ids = list(range(torch.cuda.device_count()))
        model = DataParallel(model, device_ids=device_ids)

    model = model.to(device)
    model.eval()

    all_preds = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=run_name):
            imgs = batch[0].to(device)
            preds = model(imgs)
            all_preds.append(preds.detach().cpu())
            
    full_output = torch.cat(all_preds, dim=0)
    
    del imgs
    del preds
    del model
    torch.cuda.empty_cache()
    
    return full_output

def run_full_evaluation(cfg, weights_path, evaluator, tb_logger, device):
    print("\n" + "="*50)
    print(f"Starting Evaluation for {Path(weights_path).name}")
    print("="*50)

    filename = Path(weights_path).name
    match = re.search(r'epoch(\d+)', filename)
    epoch = int(match.group(1)) if match else 0

    model = get_model(cfg["model"])
    model = load_weights_safely(model, weights_path, device)
    
    if torch.cuda.device_count() > 1:
        device_ids = list(range(torch.cuda.device_count()))
        model = DataParallel(model, device_ids=device_ids)
    
    model = model.to(device)
    model.eval()
    
    evaluator.model = model
    evaluator.epoch = epoch

    print(f"--- Evaluation Validation Set (Epoch {epoch}) ---")
    val_metrics = evaluator.evaluate(on_test=False)
    
    for name, value in val_metrics.items():
        tb_logger.log_scalar(f"val/{name}", value, epoch)
        
    wandb_val_metrics = {f"val/{k}": v for k, v in val_metrics.items()}
    wandb_val_metrics["epoch"] = epoch
    wandb.log(wandb_val_metrics)

    print(f"--- Evaluation Test Set (Epoch {epoch}) ---")
    test_metrics = evaluator.evaluate(on_test=True)
    
    for name, value in test_metrics.items():
        tb_logger.log_scalar(f"test/{name}", value, epoch)
    
    wandb_test_metrics = {f"test/{k}": v for k, v in test_metrics.items()}
    wandb_test_metrics["epoch"] = epoch
    wandb.log(wandb_test_metrics)

def main():
    parser = argparse.ArgumentParser(description="Validation and Evaluation Script")
    parser.add_argument("--config", type=str, default="config.yml", help="Chemin vers le config.yml")
    parser.add_argument("--weights", type=str, required=True, help="Chemin vers le fichier .pth")
    
    args = parser.parse_args()
    
    cfg_path = Path(args.config)
    cfg = load_config(cfg_path)
    print(f"GPUs disponibles : {torch.cuda.device_count()}")
    device = get_device(preferred=cfg["training"].get("device", "cuda"))

    val_loader, test_loader = build_dataloaders(cfg)

    weigths_paths = os.listdir(args.weights)
    weigths_paths = [os.path.join(args.weights, f) for f in weigths_paths if os.path.basename(f).endswith('.pth')]
    weigths_paths.sort(key=extract_epoch_num)
    
    wandb.init(
        project="First_Test", 
        config=cfg,
        name=f"EPM_{cfg['model'].get('name', 'model')}_{cfg['loss'].get('name', 'loss')}",
        mode="offline"  
    )
    
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    log_dir = Path(cfg["training"]["log_dir"]) / f"Post_Training_Validation_{cfg['model'].get('name', 'model')}_{cfg['loss'].get('name', 'loss')}_{job_id}"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logger = get_logger(log_dir / "validation_log.txt")
    tb_logger = TensorboardLogger(log_dir / "tensorboard_logs")
    metric_logger_path = os.path.join(log_dir, "metrics_csv")
    os.makedirs(metric_logger_path, exist_ok=True)

    criterion = get_loss(cfg["loss"]).to(device)

    duh_model = get_model(cfg["model"]).to(device)

    evaluator = Evaluator(
        model=duh_model,
        val_dataloader=val_loader,
        test_dataloader=test_loader,
        criterion=criterion,
        metrics=get_metrics(cfg["metrics"]),
        device=device,
        logger=logger,
        threshold=cfg["metrics"].get("threshold_4_binarize", 0.5),
        tb_logger=tb_logger,
        patch_size=cfg["data"].get("patch_size", 512),
        log_metric_path=metric_logger_path,
        profile_dir=None,
        roi_fnc=roi_fnc,
        compute_cpu_metrics=True,
        use_dask=False, 
    )
    

    for weights_file in weigths_paths:
        print("\n" + "#"*60)
        print("Check reproductibility")
        print("#"*60)
        
        epoch_num = extract_epoch_num(weights_file)
        if epoch_num % 50 == 0 or epoch_num == extract_epoch_num(weigths_paths[0]) or epoch_num == extract_epoch_num(weigths_paths[-1]):
            output_1 = run_inference_pass(cfg, weights_file, val_loader, device, run_name="Run A")
            output_2 = run_inference_pass(cfg, weights_file, val_loader, device, run_name="Run B")

            diff = torch.abs(output_1 - output_2)
            max_diff = diff.max().item()
            
            print(f"Max diff: {max_diff}")

            if max_diff > 1e-6:
                print("❌ Failed reproducibility check! Outputs differ between runs.")
                return
            else:
                print("\n✅ SUCCESS")

        run_full_evaluation(cfg, weights_file, evaluator, tb_logger, device)

    evaluator.done_evaluating()
    wandb.finish()

if __name__ == "__main__":
    # from dask_mpi import initialize
    # initialize(interface='lo')
    main()