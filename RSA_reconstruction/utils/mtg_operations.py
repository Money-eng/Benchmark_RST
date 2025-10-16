from copy import deepcopy
from typing import List, Dict

import numpy as np
from openalea.mtg import MTG
from rsml.misc import root_vertices


def _truncate_lists(prop: Dict[int, List], idx: int, v: int) -> None:
    val = prop.get(v)
    if isinstance(val, (list, tuple)) and len(val) > idx + 1:
        prop[v] = val[: idx + 1]  # garde 0…idx


def extract_mtg_at_time_t(g: MTG, t: int) -> MTG:
    g_new = deepcopy(g)

    time_prop = g_new.property("time")
    time_h_prop = g_new.property("time_hours")
    diameter_prop = g_new.property("diameter")
    geometry_prop = g_new.property("geometry")
    
    if t == -1: 
        t = max(max(time_prop.values()))

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

            # if list is empty has 1 or less elements, remove vertex
            if len(geometry_prop[v]) <= 1:
                to_remove.append(v)

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


def remove_single_coordinate_vertices(mtg: MTG) -> MTG:
    """
    Remove vertices that have only one coordinate in their geometry.
    """
    to_remove = []
    for key, values in mtg.property("geometry").items():
        if len(values) <= 1:
            to_remove.append(key)

    for v in to_remove:
        mtg.remove_vertex(v, reparent_child=False)

    return mtg


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


def intercept_curve(mtg: MTG, nlengths=2500, step=1e-3):
    """
    Calcule la courbe intercepto pour une plante d'un mtg, éventuellement à un temps donné.
    """
    from hydroroot.analysis import intercept
    from hydroroot.hydro_io import import_rsml_to_discrete_mtg
    mtg_2 = deepcopy(mtg)
    mtg_2 = remove_single_coordinate_vertices(mtg_2)
    mtg_test = import_rsml_to_discrete_mtg(mtg_2)
    lengths = np.linspace(0, (nlengths - 1) * step, nlengths)
    intercepto = np.array(intercept(g=mtg_test, dists=lengths, dl=3e-3, max_order=None))
    return lengths, intercepto


def intercept_curve_at_all_time(
        mtg: MTG,
        nlengths: int = 2500,
        step: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray]:
    times_prop = mtg.properties().get("time", {})
    if not times_prop:  # garde‑fou supplémentaire
        lengths = np.linspace(0, (nlengths - 1) * step, nlengths)
        return lengths, np.zeros((1, nlengths))

    max_time = max(max(times_prop.values()))
    times = range(int(max_time))

    lengths = np.linspace(0, (nlengths - 1) * step, nlengths)
    intercepto_all = []

    from hydroroot.analysis import intercept
    from hydroroot.hydro_io import import_rsml_to_discrete_mtg

    for t in times:
        mtg_t = extract_mtg_at_time_t(mtg, t)
        try:
            mtg_disc = import_rsml_to_discrete_mtg(mtg_t)
            inter = intercept(g=mtg_disc, dists=lengths, dl=3e-3, max_order=None)
            intercepto_all.append(inter)
        except IndexError as e:
            print(f"IndexError at time {t}: {e}")
        except Exception as e:
            print(f"Error processing time {t}: {e}")

    return lengths, np.asarray(intercepto_all)
