import rsml
from rsml import data

from openalea.mtg import traversal, algo

import networkx as nx
import numpy as np


fn = '/home/loai/Documents/code/RSMLExtraction/RSA_deep_working/Data/Train/230629PN017/61_graph.rsml'

def distance_polyline(p1, p2):
    """
    Compute the distance between two polylines.
    """
    p1 = np.array(p1)
    p2 = np.array([p2[0]])

    point_p2 = p2[0]
    distances = np.linalg.norm(p1 - point_p2, axis=1)
    # Index of the closest point in p1
    closest_index = np.argmin(distances)
    closest_point = p1[closest_index]
    # Distance to the closest point
    distance = np.linalg.norm(closest_point - point_p2)
    if distance > 5:
        print("Distance to the closest point ", distance)

    return closest_index 


# Be carefull, the diameter is not of the
def convert_fine_mtg(fn):
    # the conversion of the MTG is at the axis level, not at the segment level.
    g= rsml.rsml2mtg(fn)

    plants = g.vertices(scale=1)

    geometry = g.property('geometry')
    time = g.property('time')
    time_hours = g.property('time_hours')
    diameters = g.property('diameter')

    indexes = {}

    g2 = g.copy()
    for pid in plants:
        aid =  next(g.component_roots_iter(pid))
        for axis_id in traversal.pre_order2(g, vtx_id=aid):
            poly = geometry[axis_id]
            time_v = time[axis_id]
            time_hours_v = time_hours[axis_id]
            diams = diameters[axis_id]

            if g.parent(axis_id) is None:
                # create the axis at segment level
                vid = g2.add_component(complex_id=axis_id, label='Segment', 
                                      x=poly[0][0], y=poly[0][1], 
                                      time=time_v[0], 
                                      time_hours=time_hours_v[0],
                                      diameter= diams[0])
                indexes[axis_id] = [vid]
            else:
                # find the closest point in the parent axis
                parent_axis = g.parent(axis_id)
                parent_poly = geometry[parent_axis]
                closest_index = distance_polyline(parent_poly, poly)

                pid = indexes[parent_axis][closest_index]
                vid, complex_ = g2.add_child_and_complex(
                            parent=pid,
                            child=None,
                            complex=axis_id, 
                            edge_type=g.edge_type(axis_id),
                            label='Segment', 
                            x=poly[0][0], y=poly[0][1], 
                            time=time_v[0], 
                            time_hours=time_hours_v[0],
                            diameter= diams[0],
                            )
                indexes[axis_id] = [vid]

            for i, (x,y) in enumerate(poly[1:]):
                vid = g2.add_child(
                            parent=vid,
                            child=None,
                            edge_type='<',
                            label='Segment', 
                            x=x, y=y, 
                            time=time_v[i+1], 
                            time_hours=time_hours_v[i+1],
                            diameter= diams[i+1],
                            )
                indexes[axis_id].append(vid)
    return g2

def split(g):
    return algo.split(g)

def convert_nx(g):

    orders = algo.orders(g)
    g.properties()['root_deg'] = orders
    max_scale = g.max_scale()
    root_id = next(g.component_roots_at_scale_iter(g.root, scale=max_scale))

    root_coord =[g.node(root_id).x, g.node(root_id).y]

    #edge_list = []
    nodes = list(traversal.pre_order2(g, vtx_id=root_id))
    
    g_nx = nx.DiGraph()
    for v in nodes:
        parent = g.parent(v)
        if parent is None:
            continue
        
        edge_type = g.edge_type(v)
        
        # Ajouter l'arête (parent, v) avec un attribut 'type'
        g_nx.add_edge(parent, v, type=edge_type)

        #edge_list.append((parent, v))

    #g_nx = nx.from_edgelist(edge_list, create_using=nx.DiGraph)

    props = ['x', 'y', 'time', 'time_hours', 'diameters', 'label', 'diameter', 'root_deg']
    for node in nodes:
        for prop in props:
            _prop = g.property(prop)
            if node in _prop:
                g_nx.nodes[node][prop] = _prop.get(node)

    # Adaptater to Ariadne 
    # Compute LR_index
    axis_root = g.complex(root_id)
    lr_index = dict((vid, i) for i, vid in enumerate(traversal.pre_order2(g, vtx_id=axis_root)))

    for node in nodes:
        g_nx.nodes[node]['LR_index'] = lr_index[g.complex(node)] if lr_index[g.complex(node)] else None

    for node in g_nx.nodes:
        x= g_nx.nodes[node]['x'] # -root_coord[0] OLD
        y= g_nx.nodes[node]['y'] # -root_coord[1] OLD
        g_nx.nodes[node]['pos'] = [x, y]
        
    tree_nx = nx.convert_node_labels_to_integers(g_nx)
        
    return tree_nx


def visualize_root_graphs(dgs, ax=None, show=True):
    """
    Visualise une liste de graphes racinaires (dgs) sur un seul axe Matplotlib.
    
    - Nœuds : Dégradé 'hot' (feu) basé sur l'attribut 'time'.
    - Arêtes '<' (Succession) : Ligne continue (solid).
    - Arêtes '+' (Branchement) : Ligne pointillée (dotted).
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.lines import Line2D 
    
    all_times = []
    for dg in dgs:
        times_dict = nx.get_node_attributes(dg, 'time')
        if times_dict:
            all_times.extend(times_dict.values())

    if not all_times:
        print("Avertissement : Aucun attribut 'time' trouvé.")
        min_time, max_time = 0, 1
    else:
        min_time = min(all_times)
        max_time = max(all_times)
        
    print(f"Échelle de temps (Time): Min={min_time}, Max={max_time}")


    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 10))
    
    cmap_nodes = plt.cm.hot 
        
    style_map = {
        '<': 'solid',
        '+': 'dotted', 
    }
    color_map = {
        '<': 'black',
        '+': 'red',
    }
    default_style = 'dashed'
    default_color = 'gray'

    for i, dg in enumerate(dgs):
        
        pos = nx.get_node_attributes(dg, 'pos') 
        if not pos:
            print(f"Graphe {i} sauté (pas de 'pos')")
            continue

        time_values = [dg.nodes[node].get('time', min_time) for node in dg.nodes()]
        
        nx.draw_networkx_nodes(
            dg, pos,
            ax=ax,
            node_color=time_values,
            cmap=cmap_nodes,
            node_size=10
        )
        
        edges_by_type = {etype: [] for etype in style_map}
        other_edges = []

        for u, v, data in dg.edges(data=True):
            edge_type = data.get('type')
            if edge_type in style_map:
                edges_by_type[edge_type].append((u, v))
            else:
                other_edges.append((u, v))

        # Dessiner chaque groupe d'arêtes avec son style
        for edge_type, edgelist in edges_by_type.items():
            if edgelist: # S'il y a des arêtes de ce type
                nx.draw_networkx_edges(
                    dg, pos,
                    ax=ax,
                    edgelist=edgelist,
                    style=style_map[edge_type],
                    edge_color=color_map[edge_type],
                    width=1.0
                )
        
        if other_edges:
            nx.draw_networkx_edges(
                dg, pos,
                ax=ax,
                edgelist=other_edges,
                style=default_style,
                edge_color=default_color,
                width=0.5
            )

    sm = plt.cm.ScalarMappable(cmap=cmap_nodes)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
    cbar.set_label('Time')

    legend_elements = [
        Line2D([0], [0], color=color_map['<'], lw=2, label="Succession ('<')", linestyle=style_map['<']),
        Line2D([0], [0], color=color_map['+'], lw=2, label="Branchement ('+')", linestyle=style_map['+'])
    ]
    ax.legend(handles=legend_elements, loc='best')

    ax.set_title(f"Visualisation de {len(dgs)} graphes")
    ax.axis('equal')
    ax.invert_yaxis() # Décommentez si nécessaire
    
    if show:
        plt.show()
        
    return ax

def test_all():
    g = convert_fine_mtg(fn)
    gs = split(g)
    dgs = [convert_nx(g) for g in gs] # Assurez-vous que convert_nx est la version corrigée !

    print(f"Conversion terminée. {len(dgs)} graphe(s) trouvé(s). Lancement de la visualisation...")
    visualize_root_graphs(dgs)

    return dgs


# Exécuter le test
if __name__ == "__main__":
    test_all()
                
