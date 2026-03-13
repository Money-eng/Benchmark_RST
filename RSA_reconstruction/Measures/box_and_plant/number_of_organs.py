from openalea.mtg import MTG
from openalea.rsml.misc import root_vertices

from ..base import BaseMeasure


class NumberOfOrgans(BaseMeasure):
    type = "cpu"
    need = "serie"

    def __init__(self):
        super().__init__()

    def __call__(self, mtg: MTG) -> float:
        return len(root_vertices(mtg))
