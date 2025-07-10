import os

import torch
import yaml
from torch.nn import DataParallel

from DataLoaders.dataloaders import create_dataloader
from DataLoaders.transforms import get_train_img_transform_1, get_train_img_transform_2, get_train_img_transform_3, \
    get__val_test_img_transform
from Losses import get_loss
from Metrics import get_metrics
from Model import get_model
from Training.evaluator import Evaluator
from Training.trainer import Trainer
from utils.logger import get_logger, TensorboardLogger
from utils.misc import get_device
from utils.misc import set_seed, SEED

set_seed(SEED)

if __name__ == "__main__":
    g = torch.Generator()
    g.manual_seed(SEED)

    ##### Path to the config file #####
    # get current file directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    cfg_path = os.path.join(current_dir, "config.yml")
    assert os.path.exists(cfg_path), f"Le fichier de config n'existe pas : {cfg_path}"
    with open(cfg_path, "r") as f:
        config = yaml.safe_load(f)

    ##### Create dataloaders #####
    train_loader, val_loader, test_loader = create_dataloader(
        base_directory=config["data"]["base_dir"],
        img_transforms=[
            get_train_img_transform_1(patch_size=config["data"]["patch_size"]),
            get_train_img_transform_2(patch_size=config["data"]["patch_size"]),
            get_train_img_transform_3(patch_size=config["data"]["patch_size"]),
            get__val_test_img_transform()
        ],
        batch_size=int(config["data"].get("batch_size", 32)),
        generator=g
    )

    # print unique mtg value for every dataset - reminder "return img, mask.clone(), time, mtg"
    print("Unique mtg values in train dataset:",
          set([str(train_loader.dataset[i][-1]).split("/")[-2] for i in range(len(train_loader.dataset))]))
    print("Unique mtg values in val_loader dataset:",
          set([val_loader.dataset[i][-1].split("/")[-2] for i in range(len(val_loader.dataset))]))
    print("Unique mtg values in test_loader dataset:",
          set([test_loader.dataset[i][-1].split("/")[-2] for i in range(len(test_loader.dataset))]))

    ##### Initialize model, loss, optimizer, metrics, logger, and evaluator #####
    n_gpus = torch.cuda.device_count()
    print(f"{n_gpus} GPU(s) detected")
    device = get_device(preferred=config["training"].get("device", "cuda"))

    model = get_model(config["model"])
    model = DataParallel(model) if n_gpus > 1 else model
    model = model.to(device)

    criterion = get_loss(config["loss"])
    try:
        criterion = criterion.to(device)
    except:
        print("Warning: Loss function does not support moving to device. Continuing without moving it.")
        pass

    optimizer_name = config["optimizer"]["name"]
    lr = float(config["optimizer"]["learning_rate"])
    weight_decay = float(config["optimizer"].get("weight_decay", 0))

    match optimizer_name.lower():
        case "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        case "sgd":
            momentum = float(config["optimizer"].get("momentum", 0.9))
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        case "adamw":
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        case _:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Factory design pattern to get metrics from str name in config.yaml
    metrics_dict = get_metrics(config["metrics"])

    # Logs
    log_dir = config["training"]["log_dir"]
    os.makedirs(log_dir, exist_ok=True)
    log_model_path = log_dir + "/" + config["model"]["name"] + "_" + config["loss"]["name"]
    os.makedirs(log_model_path, exist_ok=True)
    if not os.path.exists(log_model_path + "/training.log"):
        open(log_model_path + "/training.log", "w").close()
    logger = get_logger(log_model_path + "/training.log")

    # TensorBoard logger
    tb_log_dir = os.path.join(log_model_path, "tensorboard_logs")
    os.makedirs(tb_log_dir, exist_ok=True)
    tb_logger = TensorboardLogger(log_dir=tb_log_dir)

    checkpoint_dir = config["training"]["checkpoint_dir"] + "/" + config["model"]["name"] + "_" + config["loss"]["name"]

    # log all configurations in the logger
    logger.info("Starting training with the following configurations:")
    logger.info(f"Model: {config['model']}")
    logger.info(f"Loss: {config['loss']}")
    logger.info(f"Optimizer: {config['optimizer']}")
    logger.info(f"Metrics: {config['metrics']}")
    logger.info(f"Training configurations: {config['training']}")

    # from torch_lr_finder import LRFinder
    # lr_finder = LRFinder(model, optimizer, criterion, device="cuda")
    # lr_finder.range_test(train_loader, end_lr=1, num_iter=200)
    # lr_finder.plot() # to inspect the loss-learning rate graph
    # lr_finder.reset() # to reset the model and optimizer to their initial state

    #### Evaluator and Trainer #####
    evaluator = Evaluator(
        model=model,
        val_dataloader=val_loader,
        test_dataloader=test_loader,
        metrics=metrics_dict,
        device=device,
        logger=logger,
        threshold=config["metrics"].get("threshold_4_binarize", 0.5),
        tb_logger=tb_logger,
        jar_path=config["rst"].get("jar_path",
                                   "/home/loai/Documents/code/RSMLExtraction/RootSystemTracker/target/rootsystemtracker-1.6.1-jar-with-dependencies.jar"),
        patch_size=config["data"].get("patch_size", 512),
        log_metric_path=log_model_path,
    )

    evaluator.evaluate()

    # evaluator.evaluate()  # Initial evaluation before training

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        config=config,
        evaluator=evaluator,
        logger=logger,
        tb_logger=tb_logger,
        checkpoint_dir=checkpoint_dir,
        device=device,
    )

    # Start training
    trainer.train()
