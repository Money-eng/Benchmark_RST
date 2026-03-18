import argparse
import os
from pathlib import Path

import yaml

from Measures import get_measures
from mesurator_para_rn2 import ReconstructionMesurator
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
        help=(
            "Path to the YAML configuration file. "
            "If omitted, 'config.yml' is searched in the same directory as this script."
        ),
    )
    
    parser.add_argument(
        "--path_to_results",
        type=str,
        help=(
            "Name of the model to use. "
            "If omitted, the model name is derived from the config file."
        ),
    )
    
    args = parser.parse_args()
    cfg_path = Path(args.config) if args.config else DEFAULT_CFG
    cfg = load_config(cfg_path)


    path2results = args.path_to_results # + "/" + cfg["model"]['name'] + "_" + cfg["loss"]['name'] + "/"
    mesurator = ReconstructionMesurator(
        pred_folder=path2results,
        gt_folder="./data_gt",
        measure=get_measures(cfg["measures"])
    )

    print("Start mesurator")
    mesurator_results = mesurator.evaluate()


if __name__ == "__main__":
    main()
