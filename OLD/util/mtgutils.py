from openalea.mtg import MTG


def mtg_at_time_t(mtg: MTG, temps_max: float) -> MTG:
    """
    Create a new MTG with only the vertices that are present at a given time.
    """
    new_g = mtg.copy()
    to_remove = []

    for v in new_g.vertices(new_g.max_scale()):  # scale=2, normalement, c’est chaque axe/racine
        node = new_g.node(v)
        if hasattr(node, "time"):
            t = node.time
        else:
            # DEBATABLE
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

    # On enlève toutes les racines/axes à supprimer
    for v in to_remove:
        new_g.remove_vertex(v)  # ou new_g.delete_vertex(v) selon la lib

    return new_g
