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

    edge_list = []
    nodes = list(traversal.pre_order2(g, vtx_id=root_id))
    for v in nodes:
        parent = g.parent(v)
        if parent is None:
            continue

        edge_list.append((parent, v))

    g_nx = nx.from_edgelist(edge_list, create_using=nx.DiGraph)

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

def test_all():
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    
    g = convert_fine_mtg(fn)

    gs = split(g)
    dgs = [convert_nx(g) for g in gs]

    print("Different edge types in the graphs:")
    edge_types = set()
    for dg in dgs:
        for u, v in dg.edges():
            edge_types.add(dg.edges[u, v].get('edge_type', 'unknown'))
    print(edge_types)

    # -------------------
    
    all_times = []
    for dg in dgs:
        times_dict = nx.get_node_attributes(dg, 'time')
        if times_dict:
            all_times.extend(times_dict.values())

    min_time = min(all_times)
    max_time = max(all_times)

    print(f"Min time: {min_time}, Max time: {max_time}")

    plt.figure(figsize=(12, 10))
    ax = plt.gca()
    
    cmap = plt.cm.hot 
    
    for i, dg in enumerate(dgs):
        
        pos = nx.get_node_attributes(dg, 'pos') 
        if not pos:
            print(f"Graphe {i} sauté (pas de 'pos')")
            continue

        time_values = [dg.nodes[node].get('time', min_time) for node in dg.nodes()]

        nx.draw(dg, pos,
                ax=ax,
                node_color=time_values, # La liste des valeurs de temps
                cmap=cmap,             # La colormap (le dégradé)
                with_labels=True,
                node_size=15,          # Taille de nœud légèrement augmentée
                width=0.5)             # Lignes un peu plus fines

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min_time, vmax=max_time))
    sm.set_array([]) 
    
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
    cbar.set_label('Time')

    plt.title(f"Visualisation de {len(dgs)} graphes (Couleur = Time)")
    plt.gca().invert_yaxis() # Décommentez si nécessaire
    plt.axis('equal')
    plt.show()

    return dgs


# Exécuter le test
if __name__ == "__main__":
    test_all()
                
