# Metrics/__init__.py

from utils.misc import set_seed, SEED

from .base import BaseMetric
from .cpu.betti0_ratio import Betti0JaccardRatio
from .cpu.betti0_relative_error import Betti0RelativeError
from .cpu.betti0_variation_index import Betti0VariationIndex
from .cpu.betti1_ratio import Betti1JaccardRatio
from .cpu.betti1_relative_error import Betti1RelativeError
from .cpu.betti1_variation_index import Betti1VariationIndex
from .cpu.euler_charac_abs_ratio import EulerCharaJaccardsRatio
from .cpu.euler_charac_relative_error import EulerCharacRelativeError
from .cpu.euler_charac_variation_index import EulerCharacVariationIndex
from .cpu.persistence_bottleneck import PeristenceBottleneck
from .cpu.persistence_wasserstein import PeristenceWasserstein
from .cpu.variation_of_information import VI
#from .gpu.apls import APLS
from .gpu.haussdorff_95 import HausdorffDistance95
from .gpu.nomalized_surface_distance import NormalizedSurfaceDistance
from .gpu.avg_centerline_distance import AverageSymetricCenterlineDistance
from .gpu.f_beta_score import FBetaScore
from .gpu.betti0_abs_err import Betti0AbsErrGPU
from .gpu.betti1_abs_err import Betti1AbsErrGPU
from .gpu.betti0_ratio_gpu import Betti0JaccardRatioGPU
from .gpu.betti0_relative_error_gpu import Betti0RelativeErrorGPU
from .gpu.betti0_variation_index_gpu import Betti0VariationIndexGPU
from .gpu.betti1_ratio_gpu import Betti1JaccardRatioGPU
from .gpu.betti1_relative_error_gpu import Betti1RelativeErrorGPU
from .gpu.betti1_variation_index_gpu import Betti1VariationIndexGPU
from .gpu.branch_break_rate import BranchBrakeRate
from .gpu.branch_length_rate import BranchLengthRate
from .gpu.cldice_metric import CLDICE_metric
from .gpu.cldice import CLDice
from .gpu.dice import Dice
from .gpu.f1_score import F1Score
from .gpu.haussdorff import HausdorffDistance
from .gpu.iou import IoU
from .gpu.mean_iou import MeanIoU
from .gpu.mutual_information import NormalizedMutualInformation
from .gpu.persistence_bottleneck_gpu import PersistenceBottleneckGPUParallel
from .gpu.persistence_wasserstein_gpu import PersistenceWassersteinGPUParallel
from .gpu.pixel_accuracy import PixelAccuracy
from .gpu.precision import Precision
from .gpu.recall import Recall
from .gpu.specificity import Specificity
from .gpu.surface_dice import Surface_dice
from .gpu.surface_difference import Surface_distance

# from .mtg.area_below_intercep import AreaBetweenIntercepts
# from .mtg.dtw_below_intercep import DTWBetweenIntercepts
# from .mtg.euclidian_btw_intercepts import EuclidianDistancebtwIntercepts
# from .mtg.number_of_organs_ratio import NumberOfOrgansRatio
# from .mtg.number_of_plants_ratio import NumberOfPlantsRatio

set_seed(SEED)  # Ensure reproducibility

# Global dictionnary to map metric names to their corresponding classes
METRIC_FACTORIES = {
    # GPU
    "dice": Dice,
    "cldice": CLDice,
    "f1_score": F1Score,
    "iou": IoU,
    "mean_iou": MeanIoU,
    "pixel_accuracy": PixelAccuracy,
    "precision": Precision,
    "recall": Recall,
    "specificity": Specificity,
    "surface_distance": Surface_distance,
    "surface_dice": Surface_dice,
    "hausdorff_distance": HausdorffDistance,
    "hausdorff_distance95": HausdorffDistance95,
    "normalized_mutual_information": NormalizedMutualInformation,
    "normalized_surface_distance": NormalizedSurfaceDistance,  
    "f_2_score": lambda **params: FBetaScore(beta=2.0, **params),
    "f_3_score": lambda **params: FBetaScore(beta=3.0, **params),
    "f_4_score": lambda **params: FBetaScore(beta=4.0, **params),
    "betti0_abs_err_gpu": Betti0AbsErrGPU,
    "betti1_abs_err_gpu": Betti1AbsErrGPU,
    "betti0_jaccard_ratio_gpu": Betti0JaccardRatioGPU,
    "betti0_relative_error_gpu": Betti0RelativeErrorGPU,
    "betti0_variation_index_gpu": Betti0VariationIndexGPU,
    "betti1_jaccard_ratio_gpu": Betti1JaccardRatioGPU,
    "betti1_relative_error_gpu": Betti1RelativeErrorGPU,
    "betti1_variation_index_gpu": Betti1VariationIndexGPU,
    "cldice_off_metric": CLDICE_metric,
    "average_symetric_centerline_distance": AverageSymetricCenterlineDistance,
    #"average_path_length_similarity": APLS,
    "persistence_bottleneck_gpu": PersistenceBottleneckGPUParallel,
    "persistence_wasserstein_gpu": PersistenceWassersteinGPUParallel,
    "branch_break_rate": BranchBrakeRate,
    "branch_length_rate": BranchLengthRate,

    # CPU
    "variation_of_information": VI,
    "betti0_jaccard_ratio": Betti0JaccardRatio,
    "betti0_relative_error": Betti0RelativeError,
    "betti0_variation_index": Betti0VariationIndex,
    "betti1_jaccard_ratio": Betti1JaccardRatio,
    "betti1_relative_error": Betti1RelativeError,
    "betti1_variation_index": Betti1VariationIndex,
    "euler_charac_jaccard_ratio": EulerCharaJaccardsRatio,
    "euler_charac_relative_error": EulerCharacRelativeError,
    "euler_charac_variation_index": EulerCharacVariationIndex,
    "persistence_bottleneck": PeristenceBottleneck,
    "persistence_wasserstein": PeristenceWasserstein
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
