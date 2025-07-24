from openalea.mtg import MTG
from rsml.misc import root_vertices, plant_vertices

def extract_mtg_at_time_t(mtg: MTG, temps_max: float) -> MTG:
    """
    Create a new MTG with only the vertices that are present before a given time.
    """
    new_g = mtg.copy()
    to_remove = []

    # scale=2, normalement, c’est chaque axe/racine
    for v in root_vertices(new_g):
        node = new_g.node(v)
        if hasattr(node, "time"):
            t = node.time
        else:
            continue

        first_t = min(t)
        if first_t > temps_max:
            to_remove.append(v)
            continue

        mask = [tt <= temps_max for tt in t]
        if hasattr(node, "geometry"):
            node.geometry = [p for p, m in zip(node.geometry, mask) if m]
        if hasattr(node, "diameter"):
            node.diameter = [d for d, m in zip(node.diameter, mask) if m]
        node.time = [tt for tt, m in zip(t, mask) if m]
        node.time_hours = [th for th, m in zip(node.time_hours, mask) if m]

        if not node.geometry or len(node.geometry) < 2:
            to_remove.append(v)

    for v in to_remove:
        new_g.remove_vertex(v)

    return new_g

def extract_plant_sub_mtg(mtg: MTG) -> dict:
    """
    Extract a sub-MTG for a specific plant.
    """
    new_g = mtg.copy()
    sub_mtgs = {}

    for v in plant_vertices(new_g):
        sub_mtg = new_g.sub_mtg(v, include_root=True)
        if sub_mtg.num_vertices() > 0:
            sub_mtgs[v] = sub_mtg

    return sub_mtgs