# Metrics/__init__.py

from .base import BaseMetric
from .cpu.betti0_ratio import Betti0JaccardRatio
from .cpu.betti0_relative_error import Betti0RelativeError
from .cpu.betti0_variation_index import Betti0VariationIndex
from .cpu.euler_charac_abs_ratio import EulerCharaJaccardsRatio
from .cpu.euler_charac_relative_error import EulerCharacRelativeError
from .cpu.euler_charac_variation_index import EulerCharacVariationIndex
from .cpu.variation_of_information import VI
from .cpu.persistence_bottleneck import PeristenceBottleneck
from .cpu.persistence_wasserstein import PeristenceWasserstein

from .gpu.ari_index import ARIScore
from .gpu.dice import Dice
from .gpu.f1_score import F1Score
from .gpu.iou import MeanIoU
from .gpu.pixel_accuracy import PixelAccuracy
from .gpu.precision import Precision
from .gpu.recall import Recall
from .gpu.specificity import Specificity
from .gpu.surface_difference import Surface_distance
from .gpu.surface_dice import Surface_dice
from .gpu.haussdorff import HausdorffDistance
from .gpu.generalized_dice import GeneralizedDice
from .gpu.mutual_information import NormalizedMutualInformation

from .mtg.area_below_intercep import AreaBetweenIntercepts
# from .mtg.dtw_below_intercep import DTWBetweenIntercepts
from .mtg.euclidian_btw_intercepts import EuclidianDistancebtwIntercepts
from .mtg.number_of_organs_ratio import NumberOfOrgansRatio
from .mtg.number_of_plants_ratio import NumberOfPlantsRatio

from utils.misc import set_seed, SEED

set_seed(SEED)  # Ensure reproducibility

# Global dictionnary to map metric names to their corresponding classes
METRIC_FACTORIES = {
    # GPU
    "ari_score": ARIScore,
    "dice": Dice,
    "generalized_dice": GeneralizedDice,
    "f1_score": F1Score,
    "mean_iou": MeanIoU,
    "pixel_accuracy": PixelAccuracy,
    "precision": Precision,
    "recall": Recall,
    "specificity": Specificity,
    "surface_distance": Surface_distance,
    "surface_dice": Surface_dice,
    "hausdorff_distance": HausdorffDistance,
    "normalized_mutual_information": NormalizedMutualInformation,

    # CPU
    "variation_of_information": VI,
    "betti0_jaccard_ratio": Betti0JaccardRatio,
    "betti0_relative_error": Betti0RelativeError,
    "betti0_variation_index": Betti0VariationIndex,
    "euler_charac_jaccard_ratio": EulerCharaJaccardsRatio,
    "euler_charac_relative_error": EulerCharacRelativeError,
    "euler_charac_variation_index": EulerCharacVariationIndex,
    "persistence_bottleneck": PeristenceBottleneck,
    "persistence_wasserstein": PeristenceWasserstein,
    # CPU / MTG
    "area_below_intercep": AreaBetweenIntercepts,
    # "dtw_below_intercep": DTWBetweenIntercepts,
    "euclidian_btw_intercepts": EuclidianDistancebtwIntercepts,
    "number_of_organs_ratio": NumberOfOrgansRatio,
    "number_of_plants_ratio": NumberOfPlantsRatio,
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
        raise ValueError(
            f"Unknown metric: {name}. Known: {list(METRIC_FACTORIES.keys())}")
    try:
        return METRIC_FACTORIES[name](**params)
    except TypeError as e:
        raise TypeError(
            f"Error instantiating metric '{name}' with params {params}: {e}")


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
    result = {"cpu": [], "gpu": [], "mtg": []}
    for t in ["cpu", "gpu", "mtg"]:
        for cfg in metrics_config.get(t, []):
            metric = get_metric(cfg)
            result[t].append(metric)
    return result
