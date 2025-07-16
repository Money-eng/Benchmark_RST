import matplotlib.pyplot as plt
import numpy as np
from openalea.mtg import MTG


def load_mtg(rsml_path):
    from rsml import rsml2mtg
    return rsml2mtg(rsml_path)


def densify_mtg(mtg: MTG, step=1.0):
    """
    Densifie un graphe MTG en ajoutant des nœuds intermédiaires entre chaque paire de nœuds connectés,
    avec une distance fixe `step` entre eux.

    :param mtg: Le graphe MTG à densifier.
    :param step: La distance entre les nœuds intermédiaires (par défaut 1.0).
    :return: Un nouveau graphe MTG densifié.
    """
    new_mtg = mtg.copy()
    max_scale = new_mtg.max_scale()
    vid_mapping = {}  # Mapping des anciens IDs vers les nouveaux

    for v in list(new_mtg.vertices(scale=max_scale)):
        node = new_mtg.node(v)
        geometry = getattr(node, "geometry", None)
        diameter = getattr(node, "diameter", None)
        time = getattr(node, "time", None)
        time_hours = getattr(node, "time_hours", None)

        if geometry is None or len(geometry) < 2:
            continue  # Rien à densifier

        new_geometry = [geometry[0]]
        new_diameter = [diameter[0]] if diameter else None
        new_time = [time[0]] if time else None
        new_time_hours = [time_hours[0]] if time_hours else None

        for i in range(1, len(geometry)):
            p0 = np.array(geometry[i - 1])
            p1 = np.array(geometry[i])
            d = np.linalg.norm(p1 - p0)

            if d == 0:
                continue  # Points identiques

            n_steps = int(np.floor(d / step))
            if n_steps == 0:
                new_geometry.append(p1.tolist())
                # if diameter:
                #   new_diameter.append(diameter[i])
                continue

            direction = (p1 - p0) / d
            for s in range(1, n_steps + 1):
                new_point = p0 + direction * step * s
                new_geometry.append(new_point.tolist())
                # if diameter:
                # Interpolation linéaire du diamètre
                #   d0 = diameter[i - 1]
                #  d1 = diameter[i]
                # interp_d = d0 + (d1 - d0) * (s * step) / d
                # new_diameter.append(interp_d)
                if time:
                    # Interpolation linéaire du temps
                    t0 = time[i - 1]
                    t1 = time[i]
                    interp_t = t0 + (t1 - t0) * (s * step) / d
                    new_time.append(interp_t)
                if time_hours:
                    # Interpolation linéaire du temps en heures
                    th0 = time_hours[i - 1]
                    th1 = time_hours[i]
                    interp_th = th0 + (th1 - th0) * (s * step) / d
                    new_time_hours.append(interp_th)

            # Ajouter le point final si nécessaire
            if not np.allclose(new_geometry[-1], p1):
                new_geometry.append(p1.tolist())
                #                if diameter:
                #                   new_diameter.append(diameter[i])
                if time:
                    new_time.append(time[i])
                if time_hours:
                    new_time_hours.append(time_hours[i])

        # Mise à jour du nœud avec la nouvelle géométrie
        node.geometry = new_geometry
        # if diameter:
        #   node.diameter = new_diameter
        if time:
            node.time = new_time
        if time_hours:
            node.time_hours = new_time_hours
    return new_mtg


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


def intercept_curve(mtg: MTG, plant_id=1, time=None, nlengths=2500, step=1e-3):
    """
    Calcule la courbe intercepto pour une plante d'un mtg, éventuellement à un temps donné.
    """
    from hydroroot.analysis import intercept
    from hydroroot.hydro_io import import_rsml_to_discrete_mtg
    if time is not None:
        sub_mtg = mtg.sub_mtg(plant_id)
        mtg_at_t = mtg_at_time_t(sub_mtg, time)
        mtg_test = import_rsml_to_discrete_mtg(mtg_at_t)
    else:
        sub_mtg = mtg.sub_mtg(plant_id)
        mtg_test = import_rsml_to_discrete_mtg(sub_mtg)
    lengths = np.linspace(0, (nlengths - 1) * step, nlengths)
    intercepto = intercept(g=mtg_test, dists=lengths, dl=3e-3, max_order=None)
    return lengths, intercepto


def intercept_curve_at_all_time(mtg: MTG, plant_id=1, nlengths=2500, step=1e-3):
    """
    Calcule la courbe intercepto pour une plante d'un mtg, éventuellement à un temps donné.
    """
    from hydroroot.analysis import intercept
    from hydroroot.hydro_io import import_rsml_to_discrete_mtg
    times = mtg.properties()["time"]
    # get max time value from dict
    max_time = max(max(times.values()))
    times = [i for i in range(1, int(max_time) + 1)]
    lengths = np.linspace(0, (nlengths - 1) * step, nlengths)
    intercepto_all = []
    for time in times:
        sub_mtg = mtg.sub_mtg(plant_id)
        mtg_at_t = mtg_at_time_t(sub_mtg, time)
        mtg_test = import_rsml_to_discrete_mtg(mtg_at_t)
        intercepto = intercept(g=mtg_test, dists=lengths,
                               dl=3e-3, max_order=None)
        intercepto_all.append(intercepto)
    intercepto_all = np.array(intercepto_all)
    return lengths, intercepto_all


def get_cmap(n, name='tab20'):
    """
    Get a colormap with n unique colors.
    """
    cmap = plt.get_cmap(name)
    if n <= cmap.N:
        colors = [cmap(i) for i in range(n)]
    else:
        # Interpolate in the colormap if n > cmap.N
        colors = [cmap(i / n) for i in range(n)]
    return colors


def plot_interceptos(curves, labels=None, title="Intercepto vs Length"):
    """
    Affiche plusieurs courbes intercepto sur le même graphique, avec palette variée.
    """
    plt.figure(figsize=(10, 6))
    n = len(curves)
    # veridis
    for i, (lengths, intercepto) in enumerate(curves):
        label = labels[i] if labels else f"Curve {i + 1}"
        plt.plot(lengths, intercepto, label=label, lw=2, alpha=0.9,
                 color=plt.cm.viridis(i / n))
    plt.xlabel("Length")
    plt.ylabel("Intercepto")
    plt.title(title)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.grid(alpha=0.3)
    plt.show()


def plot_interceptos_3d(curves, labels=None, title="Intercepto vs Length (3D)"):
    """
    Affiche plusieurs courbes intercepto sur le même graphique en 3D, palette colorée.
    """
    fig = plt.figure(figsize=(11, 7))
    ax = fig.add_subplot(111, projection='3d')
    n = len(curves)
    for i, (lengths, intercepto) in enumerate(curves):
        label = labels[i] if labels else f"Curve {i + 1}"
        ax.plot(lengths, [i + 1] * len(lengths), intercepto, label=label, lw=2, alpha=0.85,
                color=plt.cm.viridis(i / n))
    ax.set_xlabel("Length")
    ax.set_ylabel("Time (index)")
    ax.set_zlabel("Intercepto")
    plt.title(title)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.show()


if __name__ == "__main__":
    rsml_1 = "/home/loai/Images/DataTest/UC1_data/230629PN033/61_graph.rsml"
    rsml_2 = None  # Facultatif
    time_1 = 15
    time_2 = None
    plant_id_1 = 1
    plant_id_2 = 1

    # Charger et calculer
    curves = []
    labels = []

    mtg1 = load_mtg(rsml_1)
    from rsml import plot2d

    plot2d(mtg1, title="MTG 1")
    plt.show()

    dens_mtg = densify_mtg(mtg1, step=1)
    plot2d(dens_mtg, title="MTG 1 densified")
    plt.show()

    print("plant_ids", mtg1.vertices(scale=1))
    l1, i1 = intercept_curve(mtg1, plant_id=plant_id_1, time=time_1)
    # curves.append((l1, i1))
    # labels.append(f"RSML 1{' (t='+str(time_1)+')' if time_1 else ''}")
    l1_time, i1_time = intercept_curve_at_all_time(mtg1, plant_id=plant_id_1)
    for i, intercepto in enumerate(i1_time):
        curves.append((l1_time, intercepto))
        labels.append(f"t={i + 1}")

    l1_dense_time, i1_dense_time = intercept_curve_at_all_time(
        dens_mtg, plant_id=plant_id_1)
    curves2 = []
    labels2 = []
    for i, intercepto in enumerate(i1_dense_time):
        curves2.append((l1_time, intercepto))
        labels2.append(f"t={i + 1}")

    plot_interceptos(curves, labels=labels,
                     title="Courbes intercepto (couleurs variées)")
    plot_interceptos_3d(curves, labels=labels,
                        title="Courbes intercepto 3D (couleurs variées)")
    plot_interceptos(curves2, labels=labels2,
                     title="Courbes intercepto densified (couleurs variées)")
    plot_interceptos_3d(curves2, labels=labels2,
                        title="Courbes intercepto 3D densified (couleurs variées)")

    if rsml_2:
        mtg2 = load_mtg(rsml_2)
        l2, i2 = intercept_curve(mtg2, plant_id=plant_id_2, time=time_2)
        curves.append((l2, i2))
        labels.append(f"RSML 2{' (t=' + str(time_2) + ')' if time_2 else ''}")

    plot_interceptos(curves, labels=labels,
                     title="Courbes intercepto (couleurs variées)")
    plot_interceptos_3d(curves, labels=labels,
                        title="Courbes intercepto 3D (couleurs variées)")
