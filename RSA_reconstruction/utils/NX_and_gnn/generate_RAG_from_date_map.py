import os 
import tifffile
import numpy as np
import networkx as nx
from scipy.ndimage import label, distance_transform_edt, center_of_mass, maximum_position
import tqdm
from joblib import Parallel, delayed

def process_date(date, date_map_data):
    if date == 0:
        return []

    mask = date_map_data == date
    labeled_array, num_features = label(mask)
    
    if num_features == 0:
        return []

    distance_map = distance_transform_edt(mask)
    
    indices = range(1, num_features + 1)
    
    centroids_for_check = center_of_mass(mask, labeled_array, indices)
    
    edt_centers = maximum_position(distance_map, labeled_array, indices)

    border_mask = distance_map == 1
    border_coords_all = np.column_stack(np.where(border_mask))
    border_labels = labeled_array[border_mask]

    borders_by_label = {i: [] for i in indices}
    for (r, c), label_id in zip(border_coords_all, border_labels):
        if label_id > 0:
            borders_by_label[label_id].append((r, c))

    nodes_for_this_date = []
    for i in indices:

        centroid_check = centroids_for_check[i-1]
        
        if np.isnan(centroid_check[0]):
            continue
            
        center_point_rc = edt_centers[i-1] 
        
        border_list = borders_by_label[i]
        
        node_id = (date, i)
        data_dict = {
            'pos': center_point_rc,
            'date': date,
            'border_coords': np.array(border_list)
        }
        nodes_for_this_date.append((node_id, data_dict))
        
    return nodes_for_this_date

date_map_path = '/home/loai/Documents/code/RSMLExtraction/temp/input/Train/230629PN007/40_date_map.tif'

date_map = tifffile.imread(date_map_path)
unique_dates = np.unique(date_map)
dates_to_process = [d for d in unique_dates if d != 0]

G = nx.Graph()

print(f"Processing {len(dates_to_process)} dates in parallel...")

results = Parallel(n_jobs=-1)(
    delayed(process_date)(date, date_map) for date in tqdm.tqdm(dates_to_process)
)

print("Collecting results and building graph...")

all_nodes_to_add = [node for sublist in results for node in sublist]

G.add_nodes_from(all_nodes_to_add)

print(f"Number of nodes in the graph: {G.number_of_nodes()}")

import matplotlib.pyplot as plt
plt.imshow(date_map, cmap='hot')
pos = {node: G.nodes[node]['pos'][::-1] for node in G.nodes}
nx.draw(G, pos=pos, node_size=10, node_color='blue')
plt.title("Graph Nodes on Date Map")
plt.show()

print("Building border pixel map...")
border_pixel_to_node_map = {}
for node_id, data in G.nodes(data=True):
    for (r, c) in data['border_coords']:
        border_pixel_to_node_map[(r, c)] = node_id

print("Connecting touching nodes...")
processed_edges = set() 
for (r, c), node_id in tqdm.tqdm(border_pixel_to_node_map.items()):
    
    for (dr, dc) in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        neighbor_coord = (r + dr, c + dc)
        
        neighbor_node = border_pixel_to_node_map.get(neighbor_coord)
        
        if neighbor_node is not None and neighbor_node != node_id:
            
            edge = tuple(sorted((node_id, neighbor_node)))
            
            if edge not in processed_edges:
                G.add_edge(node_id, neighbor_node)
                processed_edges.add(edge)

print(f"Number of edges in the graph: {G.number_of_edges()}")

        
plt.imshow(date_map, cmap='hot')
pos = {node: G.nodes[node]['pos'][::-1] for node in G.nodes}
nx.draw(G, pos=pos, node_size=10, node_color='blue', edge_color='red')
plt.title("Graph Nodes on Date Map")
plt.show()