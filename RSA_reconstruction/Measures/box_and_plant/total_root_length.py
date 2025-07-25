from openalea.mtg import MTG
from utils.mtg_operations import total_root_length

from ..base import BaseMeasure


class TotalRootLength(BaseMeasure):
    type = "cpu"
    need = "serie"

    def __init__(self):
        super().__init__()

    def __call__(self, mtg: MTG) -> float:
        return total_root_length(mtg)
