
import argparse
from rsml import rsml2mtg
from utils.misc import SEED, set_seed
from pathlib import Path
import yaml
import os

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
    args = parser.parse_args()
    cfg_path = Path(args.config) if args.config else DEFAULT_CFG
    cfg = load_config(cfg_path)

    # Load test list of folders and val list of folders
    test_list_folders = cfg.get("test_data", [])
    val_list_folders = cfg.get("val_data", [])

    # We assume that in every folder there is a "61_graph.rsml" file (the expertized RSML) and a "61_before_expertized_graph.rsml" file (the before expertized RSML)
    dict_rsml = {
        "test": {
            folder: {
                "expertized": rsml2mtg(os.path.join(folder, "61_graph.rsml")),
                "before_expertized": rsml2mtg(os.path.join(folder, "61_before_expertized_graph.rsml"))
            }
            for folder in test_list_folders
        },
        "val": {
            folder: {
                "expertized": rsml2mtg(os.path.join(folder, "61_graph.rsml")),
                "before_expertized": rsml2mtg(os.path.join(folder, "61_before_expertized_graph.rsml"))
            }
            for folder in val_list_folders
        },
    }

    for split, data in dict_rsml.items():
        print(f"Processing {split} data:")
        for folder, rsml_data in data.items():
            print(f"  Folder: {folder}")
            print(f"    Expertized RSML: {rsml_data['expertized']}")
            print(
                f"    Before Expertized RSML: {rsml_data['before_expertized']}")


if __name__ == "__main__":
    main()
