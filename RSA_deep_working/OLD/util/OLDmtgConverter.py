# mtg_to_networkx.py
# mtg_to_networkx.py
import networkx as nx
from openalea.mtg import MTG  # seems limited to order 2
from openalea.mtg import MTG
from shapely.geometry import LineString, Point
from typing import Dict


# Assumptions:
## Only one primary root for each plant
def v0_mtg_to_networkx(
        mtg: MTG,
        copy_vertex_properties: bool = True
) -> Dict[int, nx.DiGraph]:
    print("keys of mtg properties", mtg.properties().keys())
    index_of_plants = mtg.vertices(scale=1)
    print("index_of_plants", index_of_plants)

    print(mtg.Ancestors(10))
    sub_mtgs = {}
    for vid in index_of_plants:
        # get the sub-mtg of the plant
        sub_mtg = mtg.sub_mtg(vid, copy=True)
        print("sub_mtg", sub_mtg)
        sub_mtgs[vid] = sub_mtg

    graph_dict = {}
    # add all mtg.vertices(scale=1) as keys to the graph_dict
    for vid0 in mtg.vertices(scale=1):
        G = nx.DiGraph()
        corresponding_sub_mtg = sub_mtgs[vid0]
        print("corresponding_sub_mtg", corresponding_sub_mtg)
        labels = corresponding_sub_mtg.properties().get("label", {})
        edge_type = corresponding_sub_mtg.properties().get("edge_type", {})
        geometry = corresponding_sub_mtg.properties().get("geometry", {})
        time = corresponding_sub_mtg.properties().get("time", {})
        time_hours = corresponding_sub_mtg.properties().get("time_hours", {})
        diameter = corresponding_sub_mtg.properties().get("diameter", {})
        for root_id, poly in geometry.items():
            prev_node = None
            for idx, (x, y) in enumerate(poly):
                node_id = (root_id, idx)  # identifiant unique
                attributes = {
                    "label": labels.get(root_id, ""),
                    "edge_type": edge_type.get(root_id, ""),
                    "time": time.get(root_id, "")[idx],
                    "time_hours": time_hours.get(root_id, "")[idx],
                    "diameter": diameter.get(root_id, "")[idx]
                }
                G.add_node(node_id, root=root_id, pos=(x, y), attr=attributes)
                if prev_node is not None:
                    # liaison séquentielle sur la même polyligne
                    G.add_edge(prev_node, node_id, type='segment')
                prev_node = node_id
        graph_dict[vid0] = G
    return graph_dict


def mtg_to_networkx(
        mtg: MTG
) -> Dict[int, nx.DiGraph]:
    """
    Pour chaque plant (scale=1) du MTG, construit un DiGraph où :
      - 'segment' relie les points successifs de chaque polyligne
      - 'hierarchy' relie le point projeté (sur la polyligne parent)
        au premier point de la polyligne enfant
    """
    graph_dict: Dict[int, nx.DiGraph] = {}

    # 1) extraire toutes les racines de plant (scale=1)
    plants = mtg.vertices(scale=1)
    sub_mtgs = {vid: mtg.sub_mtg(vid, copy=True) for vid in plants}

    for vid in plants:
        G = nx.DiGraph()
        sub = sub_mtgs[vid]
        geom = sub.properties().get("geometry", {})

        # 2.a) création des nœuds + arêtes 'segment'
        for root_id, poly in geom.items():
            prev = None
            for idx, (x, y) in enumerate(poly):
                nid = (root_id, idx)
                attr = {
                    "label": sub.properties().get("label", {}).get(root_id, ""),
                    "edge_type": sub.properties().get("edge_type", {}).get(root_id, ""),
                    "time": sub.properties().get("time", {}).get(root_id, [])[idx],
                    "time_hours": sub.properties().get("time_hours", {}).get(root_id, [])[idx],
                    "diameter": sub.properties().get("diameter", {}).get(root_id, [])[idx]
                }
                G.add_node(nid, root=root_id, pos=(x, y), attr=attr)
                if prev is not None:
                    G.add_edge(prev, nid, type="segment")
                prev = nid

        # 2.b) création des arêtes 'hierarchy' via sub.parent()
        for child_root, poly in geom.items():
            parent_root = sub.parent(child_root)
            # ignorer les racines sans parent utile
            if parent_root is None or parent_root not in geom:
                continue

            # 1) premier point de l'enfant
            child_nid = (child_root, 0)
            child_pt = Point(G.nodes[child_nid]['pos'])

            # 2) ligne de la racine parent
            parent_line = LineString(geom[parent_root])

            # 3) projection du point
            proj_d = parent_line.project(child_pt)
            proj_pt = parent_line.interpolate(proj_d)

            # 4) on calcule les distances cumulées pour trouver le segment
            cum = [0.0]
            for i in range(len(geom[parent_root]) - 1):
                p0 = Point(geom[parent_root][i])
                p1 = Point(geom[parent_root][i + 1])
                cum.append(cum[-1] + p0.distance(p1))

            seg_idx = next(
                i for i in range(len(cum) - 1)
                if cum[i] <= proj_d <= cum[i + 1]
            )

            # 5) insertion du nœud projeté sur la polyligne parent
            proj_nid = (parent_root, f"proj_{child_root}")
            attr = {
                "label": sub.properties().get("label", {}).get(parent_root, ""),
                "edge_type": sub.properties().get("edge_type", {}).get(parent_root, ""),
                "time": sub.properties().get("time", {}).get(parent_root, [])[seg_idx],
                "time_hours": sub.properties().get("time_hours", {}).get(parent_root, [])[seg_idx],
                "diameter": sub.properties().get("diameter", {}).get(parent_root, [])[seg_idx]
            }
            G.add_node(proj_nid,
                       root=parent_root,
                       pos=(proj_pt.x, proj_pt.y),
                       attr=attr,
                       projection_of=child_root)

            # 6) découpe du segment parent i→i+1
            u = (parent_root, seg_idx)
            v = (parent_root, seg_idx + 1)
            if G.has_edge(u, v):
                G.remove_edge(u, v)
            G.add_edge(u, proj_nid, type="segment")
            G.add_edge(proj_nid, v, type="segment")

            # 7) arête hiérarchique
            G.add_edge(proj_nid, child_nid, type="hierarchy")

        graph_dict[vid] = G

    return graph_dict


def mtg_to_networkx(
        mtg: MTG,
        at_time: int = -1
) -> Dict[int, nx.DiGraph]:
    """
    Pour chaque plant (scale=1) du MTG, construit un DiGraph où :
      - 'segment' relie les points successifs de chaque polyligne
      - 'hierarchy' relie le point projeté (sur la polyligne parent)
        au premier point de la polyligne enfant
    """
    graph_dict: Dict[int, nx.DiGraph] = {}

    # 1) extraire toutes les racines de plant (scale=1)
    plants = mtg.vertices(scale=1)
    sub_mtgs = {vid: mtg.sub_mtg(vid, copy=True) for vid in plants}

    for vid in plants:
        G = nx.DiGraph()
        sub = sub_mtgs[vid]
        geom = sub.properties().get("geometry", {})

        # 2.a) création des nœuds + arêtes 'segment'
        for root_id, poly in geom.items():
            prev = None
            for idx, (x, y) in enumerate(poly):
                if at_time != -1 and sub.properties().get("time", {}).get(root_id, [])[idx] > at_time:
                    continue
                nid = (root_id, idx)
                attr = {
                    "label": sub.properties().get("label", {}).get(root_id, ""),
                    "edge_type": sub.properties().get("edge_type", {}).get(root_id, ""),
                    "time": sub.properties().get("time", {}).get(root_id, [])[idx],
                    "time_hours": sub.properties().get("time_hours", {}).get(root_id, [])[idx],
                    "diameter": sub.properties().get("diameter", {}).get(root_id, [])[idx]
                }
                G.add_node(nid, root=root_id, pos=(x, y), attr=attr)
                if prev is not None:
                    G.add_edge(prev, nid, type="segment")
                prev = nid

        # 2.b) création des arêtes 'hierarchy' via sub.parent()
        for child_root, poly in geom.items():
            if at_time != -1 and sub.properties().get("time", {}).get(child_root, [])[0] > at_time:
                continue
            parent_root = sub.parent(child_root)
            # ignorer les racines sans parent utile
            if parent_root is None or parent_root not in geom:
                continue

            # 1) premier point de l'enfant
            child_nid = (child_root, 0)
            child_pt = Point(G.nodes[child_nid]['pos'])

            # 2) ligne de la racine parent
            parent_line = LineString(geom[parent_root])

            # 3) projection du point
            proj_d = parent_line.project(child_pt)
            proj_pt = parent_line.interpolate(proj_d)

            # 4) on calcule les distances cumulées pour trouver le segment
            cum = [0.0]
            for i in range(len(geom[parent_root]) - 1):
                p0 = Point(geom[parent_root][i])
                p1 = Point(geom[parent_root][i + 1])
                cum.append(cum[-1] + p0.distance(p1))

            seg_idx = next(
                i for i in range(len(cum) - 1)
                if cum[i] <= proj_d <= cum[i + 1]
            )

            # 5) insertion du nœud projeté sur la polyligne parent
            proj_nid = (parent_root, f"proj_{child_root}")
            attr = {
                "label": sub.properties().get("label", {}).get(parent_root, ""),
                "edge_type": sub.properties().get("edge_type", {}).get(parent_root, ""),
                "time": sub.properties().get("time", {}).get(parent_root, [])[seg_idx],
                "time_hours": sub.properties().get("time_hours", {}).get(parent_root, [])[seg_idx],
                "diameter": sub.properties().get("diameter", {}).get(parent_root, [])[seg_idx]
            }
            G.add_node(proj_nid,
                       root=parent_root,
                       pos=(proj_pt.x, proj_pt.y),
                       attr=attr,
                       projection_of=child_root)

            # 6) découpe du segment parent i→i+1
            u = (parent_root, seg_idx)
            v = (parent_root, seg_idx + 1)
            if G.has_edge(u, v):
                G.remove_edge(u, v)
            G.add_edge(u, proj_nid, type="segment")
            G.add_edge(proj_nid, v, type="segment")

            # 7) arête hiérarchique
            G.add_edge(proj_nid, child_nid, type="hierarchy")

        graph_dict[vid] = G

    return graph_dict


def plot_clean_graph(G, figsize=(8, 8)):
    import matplotlib.pyplot as plt
    plt.figure(figsize=figsize)
    # 1) tracer les segments (enchaînements intra-polyligne)
    for u, v, data in G.edges(data=True):
        x0, y0 = G.nodes[u]['pos']
        x1, y1 = G.nodes[v]['pos']
        if data.get('type') == 'segment':
            plt.plot([x0, x1], [y0, y1],
                     linewidth=1,  # épaisseur fine
                     color='saddlebrown',
                     alpha=0.8)
        elif data.get('type') == 'hierarchy':
            plt.plot([x0, x1], [y0, y1],
                     linewidth=1.5,
                     linestyle=':',
                     color='teal',
                     alpha=0.9)
        elif data.get('type') == 'parent':
            # si tu veux aussi montrer les liens parent-enfant
            plt.plot([x0, x1], [y0, y1],
                     linewidth=1,
                     linestyle='--',
                     color='firebrick',
                     alpha=0.6)

    # 2) (optionnel) dessiner les nœuds
    pos = nx.get_node_attributes(G, 'pos')
    xs, ys = zip(*pos.values())
    plt.scatter(xs, ys,
                s=5,  # taille très petite
                color='black',
                alpha=0.6)

    # 3) finition
    plt.gca().set_aspect('equal', 'box')
    plt.gca().invert_yaxis()  # inverser l'axe des ordonnées
    plt.axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    mtg_paths = "/home/loai/Images/DataTest/UC1_data/230629PN012/61_graph.rsml"
    from rsml import rsml2mtg

    mtg = rsml2mtg(mtg_paths)
    Graphs = mtg_to_networkx(mtg)
    print("G", Graphs)
    G = Graphs.get(1)
    print("G", G)
    # appel :
    plot_clean_graph(G, figsize=(10, 10))

    import matplotlib.pyplot as plt

    pos = nx.get_node_attributes(G, 'pos')
    labels = nx.get_node_attributes(G, 'label')
    edge_labels = nx.get_edge_attributes(G, 'type')
    nx.draw(G, pos, with_labels=True, node_size=50, node_color='blue', font_size=8)
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    from rsml import plot2d

    plot2d(mtg, show=True)
    plt.show()
