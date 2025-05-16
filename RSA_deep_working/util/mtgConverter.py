import logging
from typing import Dict, Optional
import networkx as nx
from shapely.geometry import LineString, Point
from openalea.mtg import MTG

logger = logging.getLogger(__name__)


def mtg_to_networkx(
    mtg: MTG,
    at_time: Optional[int] = None
) -> Dict[int, nx.DiGraph]:
    """
    Convert an MTG to a dict of NetworkX DiGraphs, one graph per plant (scale=1).

    Edges:
      - 'segment': successive points on the same polyline
      - 'hierarchy': projection of the first point of a child polyline onto its parent

    Parameters:
      mtg: MTG instance
      at_time: if provided, include only nodes whose time is <= at_time

    Returns:
      A mapping from plant IDs to their corresponding DiGraph.
    """
    graph_dict: Dict[int, nx.DiGraph] = {}
    plant_ids = mtg.vertices(scale=1)

    for vid in plant_ids:
        sub = mtg.sub_mtg(vid, copy=True)
        props = sub.properties()
        geom = props.get("geometry", {})
        G = nx.DiGraph()

        _add_segments(G, geom, props, at_time)
        _add_hierarchy(G, geom, props, sub, at_time)
        G.remove_edges_from(nx.selfloop_edges(G))
        graph_dict[vid] = G

    return graph_dict


def _add_segments(
    G: nx.DiGraph,
    geom: Dict[int, list],
    props: dict,
    at_time: Optional[int]
) -> None:
    """
    Add nodes and 'segment' edges for each polyline in geom.
    """
    for root_id, coords in geom.items():
        prev_nid = None
        times = props.get("time", {}).get(root_id, [])

        for idx, (x, y) in enumerate(coords):
            # filter by time
            if at_time is not None and idx < len(times) and times[idx] > at_time:
                break

            nid = (root_id, idx)
            attr = {
                "label": props.get("label", {}).get(root_id, ""),
                "edge_type": props.get("edge_type", {}).get(root_id, ""),
                "time": props.get("time", {}).get(root_id, [])[idx],
                "time_hours": props.get("time_hours", {}).get(root_id, [])[idx],
                "diameter": props.get("diameter", {}).get(root_id, [])[idx]
            }

            G.add_node(nid, root=root_id, pos=(x, y), attr=attr)

            if prev_nid is not None and prev_nid != nid:
                G.add_edge(prev_nid, nid, type="segment")

            prev_nid = nid


def _add_hierarchy(
    G: nx.DiGraph,
    geom: Dict[int, list],
    props: dict,
    sub: MTG,
    at_time: Optional[int]
) -> None:
    """
    Add 'hierarchy' edges by projecting each child root's first point onto its parent.
    """
    for child_root, _ in geom.items():
        time0 = props.get("time", {}).get(child_root, [float('inf')])[0]
        if at_time is not None and time0 > at_time:
            continue

        parent_root = sub.parent(child_root)
        if parent_root is None or parent_root not in geom:
            continue

        # project first child point
        child_nid = (child_root, 0) # first point of child polyline
        child_pt = Point(G.nodes[child_nid]["pos"])
        parent_coords = geom[parent_root] # list of coordinates
        parent_line = LineString(parent_coords) # line segment made of coordinates

        proj_d = parent_line.project(child_pt) # orthogonal projection of child_pt on parent_line
        proj_pt = parent_line.interpolate(proj_d) # projected point on parent polyline
        seg_idx = _find_segment_index(parent_coords, proj_d) # to find the right segment index

        proj_nid = (parent_root, f"proj_{child_root}") # new point (projection of child onto parent polyline)
        attr = {
            "label": props.get("label", {}).get(parent_root, ""),
            "edge_type": props.get("edge_type", {}).get(parent_root, ""),
            "time": props.get("time", {}).get(parent_root, [])[seg_idx], # TODO debatable - should be interpolated
            "time_hours": props.get("time_hours", {}).get(parent_root, [])[seg_idx], # debatable
            "diameter": props.get("diameter", {}).get(parent_root, [])[seg_idx] # debatable
        }

        G.add_node( # we add a new node for the projected point
            proj_nid,
            root=parent_root,
            pos=(proj_pt.x, proj_pt.y),
            projection_of=child_root,
            attr=attr
        )

        # replace segment with two subsegments
        u = (parent_root, seg_idx)
        v = (parent_root, seg_idx + 1)
        if G.has_edge(u, v):
            G.remove_edge(u, v) # remove the original edge
        if u != proj_nid and v != proj_nid:
            G.add_edge(u, proj_nid, type="segment")
            G.add_edge(proj_nid, v, type="segment")

        # hierarchy link
        if (proj_nid != child_nid):
            G.add_edge(proj_nid, child_nid, type="hierarchy")


def _find_segment_index(coords: list, distance: float) -> int:
    """
    Locate the segment index in a polyline given a projected distance.
    """
    cum = 0.0
    for i in range(len(coords) - 1):
        p0 = Point(coords[i])
        p1 = Point(coords[i + 1])
        seg_len = p0.distance(p1)
        if cum <= distance <= cum + seg_len:
            return i
        cum += seg_len
    return len(coords) - 2  # fallback

def verify_lengths(
    mtg: MTG,
    graphs: Dict[int, nx.DiGraph]
) -> None:
    """
    Compare original root polyline lengths in the MTG with segment lengths in the NetworkX graphs.
    Prints mismatches for each plant and root.
    """
    from shapely.geometry import Point
    for vid, G in graphs.items():
        sub = mtg.sub_mtg(vid, copy=True)
        geom = sub.properties().get("geometry", {})
        for root_id, coords in geom.items():
            # original length
            orig_len = sum(
                Point(coords[i]).distance(Point(coords[i+1]))
                for i in range(len(coords) - 1)
            )
            # graph length: from first point to last point
            nx_len = 0.0
            for u, v in G.edges():
                if G.nodes[u]['root'] == root_id and G.nodes[v]['root'] == root_id:
                    nx_len += Point(G.nodes[u]['pos']).distance(Point(G.nodes[v]['pos']))
            # compare lengths
            if abs(orig_len - nx_len) > 1e-6:
                logger.warning(
                    f"Length mismatch for plant {vid}, root {root_id}: "
                    f"MTG length = {orig_len}, NetworkX length = {nx_len}"
                    f"\nNumber of nodes in MTG = {len(coords)}, "
                    f"Number of nodes in NetworkX = {len(G.nodes())}, "
                )
            else:
                logger.warning(
                    f"Length match for plant {vid}, root {root_id}: "
                    f"MTG length = {orig_len}, NetworkX length = {nx_len}"
                    f"\nNumber of nodes in MTG = {len(coords)}, "
                    f"Number of nodes in NetworkX = {len(G.nodes())}, "
                )
            

def plot_clean_graph(
    G: nx.DiGraph,
    figsize: tuple = (8, 8)
) -> None:
    """
    Visualize the graph with 'segment' and 'hierarchy' edges.
    """
    import matplotlib.pyplot as plt
    plt.figure(figsize=figsize)

    for u, v, data in G.edges(data=True):
        x0, y0 = G.nodes[u]['pos']
        x1, y1 = G.nodes[v]['pos']
        linestyle = '-' if data['type'] == 'segment' else ':'
        plt.plot([x0, x1], [y0, y1], linewidth=1, linestyle=linestyle)

    xs, ys = zip(*nx.get_node_attributes(G, 'pos').values())
    plt.scatter(xs, ys, s=5)

    plt.gca().set_aspect('equal', 'box')
    plt.gca().invert_yaxis()
    plt.axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from rsml import rsml2mtg, plot2d

    path = "/home/loai/Images/DataTest/UC1_data/230629PN012/61_graph.rsml"
    mtg = rsml2mtg(path)
    graphs = mtg_to_networkx(mtg)

    plt.gca().invert_yaxis()
    
    for G in graphs.values():
        plot_clean_graph(G, figsize=(10, 10))
        pos = nx.get_node_attributes(G, 'pos')
        labels = nx.get_node_attributes(G, 'label')
        edge_labels = nx.get_edge_attributes(G, 'type')
        print(f"Graph {G}")
        # print g adjacency matrix
        print(nx.to_numpy_array(G))
        print("Nodes:", G.nodes())
        print("Edges:", G.edges())
        print("Labels:", labels)
        print("Edge labels:", edge_labels)
        nx.draw(G, pos, with_labels=True, node_size=50,
                node_color='blue', font_size=8)
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    plot2d(mtg, show=True)
    plt.show()
