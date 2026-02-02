import openalea.rsml as rsml
from openalea.rsml import data

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

def _get_line_segment_intersection(p1, p2, p3, p4):
    """
    Trouve le point d'intersection de deux segments de ligne [p1, p2] et [p3, p4].
    Renvoie (x, y) s'ils s'intersectent, sinon None.
    Ignore les intersections aux points d'extrémité (t=0, t=1, u=0, u=1).
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denominator == 0:
        return None
    t_num = (x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)
    u_num = (x1 - x3) * (y1 - y2) - (y1 - y3) * (x1 - x2)
    t = t_num / denominator
    u = u_num / denominator
    if 0 < t < 1 and 0 < u < 1:
        ix = x1 + t * (x2 - x1)
        iy = y1 + t * (y2 - y1)
        return (ix, iy)
    return None

def find_all_edge_crossings(G, pos):
    """
    Identifie tous les croisements géométriques d'arêtes À L'INTÉRIEUR d'un seul graphe G.
    """
    crossings = []
    edges = list(G.edges())
    
    for i in range(len(edges)):
        for j in range(i + 1, len(edges)):
            e1 = edges[i]
            e2 = edges[j]
            u1, v1 = e1
            u2, v2 = e2
            if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
                continue
            p1, p2 = pos.get(u1), pos.get(v1)
            p3, p4 = pos.get(u2), pos.get(v2)
            if not all([p1, p2, p3, p4]): continue
            
            intersection_point = _get_line_segment_intersection(p1, p2, p3, p4)
            if intersection_point:
                # On retourne les graphes et arêtes pour une utilisation générique
                crossings.append((G, e1, G, e2, intersection_point))
                
    return crossings

def _resolve_crossing_pair(g1, e1, g2, e2):
    if g1.edges[e1].get('type') == '+' or g2.edges[e2].get('type') == '+':
        return False
    try:
        # Obtenir le temps des nœuds (on prend le nœud enfant 'v')
        u1, v1 = e1 # u1 est le parent, v1 est l'enfant
        u2, v2 = e2
        time_eu1 = g1.nodes[u1]['time']
        time_eu2 = g2.nodes[u2]['time']
        time_ev1 = g1.nodes[v1]['time']
        time_ev2 = g2.nodes[v2]['time']
        
        # Logique : l'arête ayant le temps le plus grand est cachée
        time_e1 = (time_eu1 + time_ev1) / 2.0
        time_e2 = (time_eu2 + time_ev2) / 2.0
        
        hidden_g, hidden_edge = None, None
        if time_e1 > time_e2:
            hidden_g, hidden_edge = g1, e1
        elif time_e2 > time_e1:
            hidden_g, hidden_edge = g2, e2
        else:
            print("Temps égaux, ne peut pas résoudre le croisement.")
            return False
        
        # Muter le graphe original
        if hidden_g:
            u, v = hidden_edge
            if hidden_g.edges[u, v].get('type') != 'hidden':
                hidden_g.edges[u, v]['type'] = 'hidden'
                return True 
                
    except (IndexError, KeyError) as e:
        print(f"Erreur lors de la résolution du croisement : {e}")
        pass
        
    return False

def find_inter_graph_crossings(g1, pos1, g2, pos2):
    """
    Trouve tous les croisements ENTRE g1 et g2.
    """
    crossings = []
    
    # Compare chaque arête de g1 avec chaque arête de g2
    for u1, v1 in g1.edges():
        for u2, v2 in g2.edges():
            
            p1, p2 = pos1.get(u1), pos1.get(v1)
            p3, p4 = pos2.get(u2), pos2.get(v2)
            
            if not all([p1, p2, p3, p4]): continue
            
            intersection_point = _get_line_segment_intersection(p1, p2, p3, p4)
            if intersection_point:
                crossings.append( (g1, (u1, v1), g2, (u2, v2), intersection_point) )
    return crossings

def resolve_all_crossings_in_list(dgs):
    hidden_edges_count = 0
    
    # Étape 1 : Résoudre les croisements INTRA-graphe (graphe vs lui-même)
    print("Étape 1/2 : Résolution des croisements intra-graphe...")
    for dg in dgs:
        pos = nx.get_node_attributes(dg, 'pos')
        # Trouve les croisements à l'intérieur de ce graphe
        intra_crossings = find_all_edge_crossings(dg, pos)
        
        for g1, e1, g2, e2, _ in intra_crossings:
            if _resolve_crossing_pair(g1, e1, g2, e2):
                hidden_edges_count += 1

    print(f"Total arêtes cachées (intra) : {hidden_edges_count}")
    
    if (len(dgs) > 5):
        print("Plus de 5 graphes, saut de la résolution inter-graphe pour des raisons de performance.")

    # Étape 2 : Résoudre les croisements INTER-graphe (graphe vs tous les autres)
    print("Étape 2/2 : Résolution des croisements inter-graphe...")
    pos_dict_list = [nx.get_node_attributes(dg, 'pos') for dg in dgs]

    for i in range(len(dgs)):
        for j in range(i + 1, len(dgs)):
            
            g1, pos1 = dgs[i], pos_dict_list[i]
            g2, pos2 = dgs[j], pos_dict_list[j]
            
            # Trouve les croisements entre g1 et g2
            inter_crossings = find_inter_graph_crossings(g1, pos1, g2, pos2)
            
            for g_a, e_a, g_b, e_b, _ in inter_crossings:
                if _resolve_crossing_pair(g_a, e_a, g_b, e_b):
                    hidden_edges_count += 1

    if (len(dgs) > 5):
        print("Plus de 5 graphes, saut de la résolution inter-graphe pour des raisons de performance.")  
    print(f"Total arêtes cachées (final) : {hidden_edges_count}")
    
    return dgs

def visualize_root_graphs(dgs, ax=None, show=True, background_image_path=None, show_label=False):
    """
    Visualise une liste de graphes racinaires (dgs) sur un seul axe Matplotlib.
    
    - Image de fond : si 'background_image_path' est fourni.
    - Nœuds : Dégradé 'hot' (feu) basé sur l'attribut 'time'.
    - Arêtes '<' (Succession) : Ligne continue (noir).
    - Arêtes '+' (Branchement) : Ligne pointillée (rouge).
    - Arêtes 'hidden' (Caché) : Ligne en tirets (gris).
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import tifffile as tf
    all_times = []
    for dg in dgs:
        times_dict = nx.get_node_attributes(dg, 'time')
        if times_dict:
            all_times.extend(times_dict.values())

    if not all_times:
        print("Avertissement : Aucun attribut 'time' trouvé.")
        min_time, _ = 0, 1
    else:
        min_time = min(all_times)
        _ = max(all_times)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 10))
    
    xlim, ylim = None, None
    if background_image_path:
        try:
            with tf.TiffFile(background_image_path) as tif:
                img = tif.pages[-1].asarray()
                ax.imshow(img, cmap='gray')
                xlim, ylim = ax.get_xlim(), ax.get_ylim()
        except Exception as e:
            print(f"Avertissement : Erreur lors du chargement de l'image : {e}")

    cmap_nodes = plt.cm.hot 
    style_map = {
        '<': 'solid',   # Succession
        '+': 'dotted',  # Branchement
        'hidden': 'dashed'  # Caché (en tirets)
    }
    color_map = {
        '<': 'black',
        '+': 'red',
        'hidden': 'gray'
    }
    default_style = 'dotted' 
    default_color = 'purple' 
    
    for i, dg in enumerate(dgs):
        
        pos = nx.get_node_attributes(dg, 'pos') 
        if not pos:
            print(f"Graphe {i} sauté (pas de 'pos')")
            continue

        time_values = [dg.nodes[node].get('time', min_time) for node in dg.nodes()]
        nx.draw_networkx_nodes(dg, pos, ax=ax, node_color=time_values,
                               cmap=cmap_nodes, node_size=10)
        
        if show_label:
            labels = {node: str(node) for node in dg.nodes()}
            nx.draw_networkx_labels(dg, pos, labels=labels, font_size=6, ax=ax)

        edges_by_type = {etype: [] for etype in style_map}
        other_edges = []

        for u, v, data in dg.edges(data=True):
            edge_type = data.get('type')
            if edge_type in style_map:
                edges_by_type[edge_type].append((u, v))
            else:
                other_edges.append((u, v))

        for edge_type, edgelist in edges_by_type.items():
            if edgelist:
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
        Line2D([0], [0], color=color_map['+'], lw=2, label="Branchement ('+')", linestyle=style_map['+']),
        Line2D([0], [0], color=color_map['hidden'], lw=2, label="Caché ('hidden')", linestyle=style_map['hidden'])
    ]
    ax.legend(handles=legend_elements, loc='best')
    ax.set_title(f"Visualisation de {len(dgs)} graphes")
    
    if xlim and ylim:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
    else:
        ax.axis('equal')
    
    if not background_image_path:
        ax.invert_yaxis()
    
    if show:
        plt.show()
    return ax


def test_all():
    g = convert_fine_mtg(fn)
    gs = split(g)
    dgs = [convert_nx(g) for g in gs] # Assurez-vous que convert_nx est la version corrigée !

    print(f"Conversion terminée. {len(dgs)} graphe(s) trouvé(s). Lancement de la visualisation...")
    #visualize_root_graphs(dgs, background_image_path="/home/loai/Documents/code/RSMLExtraction/RSA_deep_working/Data/Train/230629PN017/22_registered_stack.tif", show_label=True)

    dgs_resolved = resolve_all_crossings_in_list(dgs)
    
    visualize_root_graphs(dgs_resolved, background_image_path="/home/loai/Documents/code/RSMLExtraction/RSA_deep_working/Data/Train/230629PN017/22_registered_stack.tif", show_label=False)

    # save results in temp folder
    temp_folder = "/home/loai/Documents/code/RSMLExtraction/temp/mtg_to_nx"
    from pathlib import Path
    Path(temp_folder).mkdir(parents=True, exist_ok=True)
    import pickle
    for i, dg in enumerate(dgs_resolved):
        pickle.dump(dg, open(f"{temp_folder}/graph_{i}.pkl", "wb"))
    
    for i, dg in enumerate(dgs_resolved):
        other_dg = pickle.load(open(f"{temp_folder}/graph_{i}.pkl", "rb"))
        assert nx.is_isomorphic(dg, other_dg), f"Graphe {i} n'est pas isomorphe après sauvegarde et chargement."
        # assert all edges and their attributes are the same
        for u, v in dg.edges():
            assert dg.edges[u, v] == other_dg.edges[u, v], f"Attributs d'arête différents pour l'arête ({u}, {v}) dans le graphe {i}."
    return dgs


def construct_weird_dataset():
    
    root_folder = "/home/loai/Documents/code/RSMLExtraction/RSA_deep_working/Data/" # contains Train and Test folders that each contain box folders that contain rsml and tiff files
    output_folder = "/home/loai/Documents/code/RSMLExtraction/temp/mtg_to_nx/"
    
    from pathlib import Path
    Path(output_folder).mkdir(parents=True, exist_ok=True)

   # replicate the folder structure
    for sp in ['Train', 'Test', 'Val']:
        split_folder = Path(root_folder) / sp
        box_folders = [f for f in split_folder.iterdir() if f.is_dir()]
        for box in box_folders:
            box_name = box.name
            output_box_folder = Path(output_folder) / sp / box_name
            output_box_folder.mkdir(parents=True, exist_ok=True)
            
            rsml_files = list(box.glob("*.rsml"))
            
            for rsml_file in rsml_files:
                g = convert_fine_mtg(rsml_file)
                gs = split(g)
                dgs = [convert_nx(g) for g in gs] 
                
                dgs_resolved = resolve_all_crossings_in_list(dgs)
                
                import pickle
                for i, dg in enumerate(dgs_resolved):
                    pickle.dump(dg, open(f"{output_box_folder}/graph_from_rsml{i}.pkl", "wb"))
                    
          
def load_and_visualize_example():
    import pickle
    folder_path = "/home/loai/Documents/code/RSMLExtraction/temp/mtg_to_nx/Val/230629PN031/"
    dgs = []
    import os
    import glob
    for i in range(os.listdir(folder_path).__len__()):
        file_path = os.path.join(folder_path, f"graph_from_rsml{i}.pkl")
        if os.path.exists(file_path):
            dg = pickle.load(open(file_path, "rb"))
            dgs.append(dg)
    visualize_root_graphs(dgs, show_label=False, background_image_path="/home/loai/Documents/code/RSMLExtraction/RSA_deep_working/Data/Val/230629PN031/22_registered_stack.tif")


# Exécuter le test
if __name__ == "__main__":
    load_and_visualize_example()
                
