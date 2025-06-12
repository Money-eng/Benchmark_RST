# Metrics/__init__.py

from .base import BaseMetric
from .cpu.are_error import AREError
from .cpu.ari_index import ARIIndex
from .cpu.betti0_difference import Betti0Difference
from .cpu.connectivity import Connectivity
from .cpu.euler_charac_difference import EulerCharacDifference
from .cpu.vi_index import VIIndex
from .gpu.dice import Dice
from .gpu.f1_score import F1Score
from .gpu.iou import IoU
from .gpu.pixel_accuracy import PixelAccuracy
from .gpu.precision import Precision
from .gpu.recall import Recall
from .gpu.specificity import Specificity
from .gpu.surface_difference import Surface_distance

# Global dictionnary to map metric names to their corresponding classes
METRIC_FACTORIES = {
    # GPU
    "dice": Dice,
    "f1_score": F1Score,
    "iou": IoU,
    "pixel_accuracy": PixelAccuracy,
    "precision": Precision,
    "recall": Recall,
    "specificity": Specificity,
    "surface_distance": Surface_distance,

    # CPU
    "connectivity": Connectivity,
    "ari_index": ARIIndex,
    "are_error": AREError,
    "vi_index": VIIndex,
    "betti0_difference": Betti0Difference,
    "euler_charac_difference": EulerCharacDifference,
}


def get_metric(metric_config: dict) -> BaseMetric:
    """
    Instanciate a given metric based on its configuration.
    {
        "name": "dice",
        "params": {...} 
    }
    """
    name = metric_config["name"]
    params = metric_config.get("params", {})
    if name not in METRIC_FACTORIES:
        raise ValueError(f"Unknown metric: {name}. Known: {list(METRIC_FACTORIES.keys())}")
    try:
        return METRIC_FACTORIES[name](**params)
    except TypeError as e:
        raise TypeError(f"Error instantiating metric '{name}' with params {params}: {e}")


def get_metrics(metrics_config: dict) -> dict:
    """
    Instanciate all metrics based on the provided configuration.
    The configuration should be a dictionary with two keys: "cpu" and "gpu",
    each containing a list of metric configurations.    
    Example:
    {
        "cpu": [
            {"name": "connectivity", "params": {}},
            {"name": "ari_index", "params": {}}
        ],
        "gpu": [
            {"name": "dice", "params": {}},
            {"name": "iou", "params": {}}
        ]
    }
    """
    result = {"cpu": [], "gpu": []}
    for t in ["cpu", "gpu"]:
        for cfg in metrics_config.get(t, []):
            metric = get_metric(cfg)
            result[t].append(metric)
    return result
