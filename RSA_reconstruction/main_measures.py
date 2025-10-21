import argparse
import os
from pathlib import Path

import yaml

from Measures import get_measures
from mesurator_para import ReconstructionMesurator
from utils.misc import SEED, set_seed

set_seed(SEED)
DEFAULT_CFG: Path = Path(__file__).with_name("config.yml")


def load_config(cfg_path: Path | str) -> dict:
    """Load and return the YAML configuration dictionary."""
    cfg_path = Path(cfg_path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with cfg_path.open("r") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train/Test a model for root system segmentation in 2D grayscale images."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "Path to the YAML configuration file. "
            "If omitted, 'config.yml' is searched in the same directory as this script."
        ),
    )
    
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help=(
            "Name of the model to use. "
            "If omitted, the model name is derived from the config file."
        ),
    )
    
    args = parser.parse_args()
    cfg_path = Path(args.config) if args.config else DEFAULT_CFG
    cfg = load_config(cfg_path)

    model_name = args.model_name if args.model_name else cfg.get("model_name", "default_model")

    print("Start")
    
    GT_VAL_FOLDERS = os.path.join(cfg["data"]["base_dir"], "Val")
    GT_TEST_FOLDERS = os.path.join(cfg["data"]["base_dir"], "Test")

    print(GT_VAL_FOLDERS)
    print(GT_TEST_FOLDERS)

    PRED_VAL_FOLDERS = os.path.join(cfg["data"]["save_path"], model_name, "Val")
    PRED_TEST_FOLDERS = os.path.join(cfg["data"]["save_path"], model_name, "Test")

    print(PRED_VAL_FOLDERS)
    print(PRED_TEST_FOLDERS)

    # list subfolder in all above directories and assert we can find the same number of folders in each
    gt_val_folders = sorted(os.listdir(GT_VAL_FOLDERS))
    gt_test_folders = sorted(os.listdir(GT_TEST_FOLDERS))
    pred_val_folders = sorted(os.listdir(PRED_VAL_FOLDERS))
    pred_test_folders = sorted(os.listdir(PRED_TEST_FOLDERS))

    assert len(gt_val_folders) == len(pred_val_folders), "Mismatch in number of validation folders"
    assert len(gt_test_folders) == len(pred_test_folders), "Mismatch in number of test folders"

    print("Evaluation")
    evaluator = ReconstructionMesurator(
        pred_folder="Results/Reconstruction_per_epoch/",
        gt_folder="RSA_deep_working/Data",
        measure=get_measures(cfg["measures"])
    )

    print("Start evaluation")
    evaluation_results = evaluator.evaluate()


if __name__ == "__main__":
    main()
