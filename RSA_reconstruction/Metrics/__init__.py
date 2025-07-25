# Metrics/__init__.py

from utils.misc import set_seed, SEED

from .base import BaseMetric
from .box_and_plant.total_root_length_ratio import TotalRootLengthRatio
from .box_and_plant.area_below_intercep import AreaBetweenIntercepts
from .box_and_plant.dtw_below_intercep import DTWBetweenIntercepts
from .box_and_plant.euclidian_btw_intercepts import EuclidianDistancebtwIntercepts
from .box_and_plant.number_of_organs_ratio import NumberOfOrgansRatio

from .box.number_of_plants_ratio import NumberOfPlantsRatio

from .plant.area_convex_hull import Area_convex_Hull

# from .mtg.area_below_intercep import AreaBetweenIntercepts
# from .mtg.dtw_below_intercep import DTWBetweenIntercepts
# from .mtg.euclidian_btw_intercepts import EuclidianDistancebtwIntercepts
# from .mtg.number_of_organs_ratio import NumberOfOrgansRatio
# from .mtg.number_of_plants_ratio import NumberOfPlantsRatio

set_seed(SEED)  # Ensure reproducibility

# Global dictionnary to map metric names to their corresponding classes
METRIC_FACTORIES = {
    # Per box
    "number_of_plants_ratio": NumberOfPlantsRatio,
    # Per plant
    "area_convex_hull": Area_convex_Hull,
    # Both
    "number_of_organs_ratio": NumberOfOrgansRatio,
    "total_root_length_distances": TotalRootLengthRatio,
    "area_below_intercept": AreaBetweenIntercepts,
    "dtw_below_intercept": DTWBetweenIntercepts,
    "euclidian_btw_intercepts": EuclidianDistancebtwIntercepts,
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
    result = {"per_plant": [], "per_box": []}
    for t in ["per_plant", "per_box"]:
        for cfg in metrics_config.get(t, []):
            metric = get_metric(cfg)
            result[t].append(metric)
    return result
