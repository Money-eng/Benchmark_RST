import os
import yaml
import torch
from Model import get_model
from Losses import get_loss
from Metrics import get_metrics
from Training.trainer import Trainer
from Training.evaluator import Evaluator
from DataLoaders.dataloaders import create_dataloader
from torchvision import transforms
from utils.logger import get_logger, TensorboardLogger
from utils.misc import get_device

if __name__ == "__main__":
    # ----------------------------------------
    # 1) Chargement du fichier de config YAML
    # ----------------------------------------
    cfg_path = "RSA_deep_working/Models/config.yml"
    assert os.path.exists(cfg_path), f"Le fichier de config n'existe pas : {cfg_path}"
    with open(cfg_path, "r") as f:
        config = yaml.safe_load(f)

    # ----------------------------------------
    # 2) Préparation des transformations
    # ----------------------------------------
    pad = transforms.Pad(padding=(0, 0, 28, 18), fill=0)  # grayscale → fill=0

    img_transform = transforms.Compose([
            pad,
            transforms.ToTensor(),
    ])
    mask_transform = transforms.Compose([
        pad,
        transforms.ToTensor(),
    ])

    # ----------------------------------------
    # 3) Création des DataLoaders
    # ----------------------------------------
    train_loader, val_loader, test_loader, series_val_loader, series_test_loader = create_dataloader(
        base_directory=config["data"]["base_dir"],
        img_transform=img_transform,
        mask_transform_image=mask_transform,
        mask_transform_series=mask_transform,
        default_batch_size=int(config["data"].get("batch_size", 32)),
    )

    # ----------------------------------------
    # 4) Détection du device
    # ----------------------------------------
    device = get_device(preferred=config["training"].get("device", "cuda"))

    # Vider le cache CUDA si nécessaire
    torch.cuda.empty_cache()

    # ----------------------------------------
    # 5) Instanciation du modèle, de la loss, de l'optimiseur
    # ----------------------------------------
    model = get_model(config["model"])
    model = model.to(device)

    criterion = get_loss(config["loss"])
    # Si la loss contient des paramètres internes torch.Tensor, on la place aussi sur device
    try:
        criterion = criterion.to(device)
    except:
        pass

    optimizer_name = config["optimizer"]["name"]
    lr = float(config["optimizer"]["learning_rate"])
    weight_decay = float(config["optimizer"].get("weight_decay", 0))
    if optimizer_name.lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        # À compléter si vous souhaitez d'autres optimizers
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # ----------------------------------------
    # 6) Instanciation des métriques via la factory
    # ----------------------------------------
    metrics_dict = get_metrics(config["metrics"])
    # metrics_dict est un dict {"gpu": [Dice(), IoU(), ...], "cpu": [Connectivity(), ...]}

    # ----------------------------------------
    # 7) Création des loggers
    # ----------------------------------------
    # Logger Python (console / fichier si configuré)
    logger = get_logger()  # Ex. renvoie un objet logging.Logger déjà configuré

    # TensorBoard logger : on crée un dossier au sein de checkpoint_dir/tensorboard
    tb_log_dir = os.path.join(config["training"]["checkpoint_dir"], "tensorboard_logs")
    os.makedirs(tb_log_dir, exist_ok=True)
    tb_logger = TensorboardLogger(log_dir=tb_log_dir)

    # ----------------------------------------
    # 8) Instanciation de l'Evaluator
    # ----------------------------------------
    evaluator = Evaluator(
        model=model,
        val_dataloader=val_loader,
        test_dataloader=test_loader,
        val_series_dataloader=series_val_loader,
        test_series_dataloader=series_test_loader,
        metrics=metrics_dict,
        device=device,
        logger=logger,
        tb_logger=tb_logger,
    )

    # ----------------------------------------
    # 9) Instanciation du Trainer
    # ----------------------------------------
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        config=config,
        evaluator=evaluator,
        logger=logger,
        tb_logger=tb_logger,
        device=device,
    )

    # ----------------------------------------
    # 10) Lancement de l'entraînement
    # ----------------------------------------
    trainer.train()
