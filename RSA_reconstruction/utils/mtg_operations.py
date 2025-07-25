from openalea.mtg import MTG
from rsml.misc import root_vertices

from typing import List, Dict
from copy import deepcopy
from openalea.mtg import MTG


def _truncate_lists(prop: Dict[int, List], idx: int, v: int) -> None:
    val = prop.get(v)
    if isinstance(val, (list, tuple)) and len(val) > idx + 1:
        prop[v] = val[: idx + 1]                      # garde 0…idx


def extract_mtg_at_time_t(g: MTG, t: int) -> MTG:
    g_new = deepcopy(g)

    time_prop = g_new.property("time")
    time_h_prop = g_new.property("time_hours")
    diameter_prop = g_new.property("diameter")
    geometry_prop = g_new.property("geometry")
    
    to_remove = []
    for v, serie in time_prop.items():
        first_t = serie[0]
        if first_t > t:
            to_remove.append(v)
        else:
            idx = max(i for i, tau in enumerate(serie) if tau <= t)

            _truncate_lists(time_prop, idx, v)
            _truncate_lists(time_h_prop, idx, v)
            _truncate_lists(diameter_prop, idx, v)
            _truncate_lists(geometry_prop, idx, v)

    for v in to_remove:
        try:
            g_new.remove_tree(v)
        except Exception:
            g_new.remove_vertex(v, reparent_child=False)

    return g_new


def extract_plant_sub_mtg(mtg: MTG, plant_vertex: int) -> dict:
    """
    Extract a sub-MTG for a specific plant.
    """
    return mtg.sub_mtg(plant_vertex).copy()


def total_root_length(mtg: MTG) -> float:
    roots = root_vertices(mtg)
    total_length = 0.0
    for root in roots:
        geometry = mtg.property("geometry")
        polyline = geometry[root]
        for i in range(len(polyline) - 1):
            length = ((polyline[i][0] - polyline[i + 1][0]) ** 2 +
                      (polyline[i][1] - polyline[i + 1][1]) ** 2) ** 0.5
            total_length += length
    return total_length
