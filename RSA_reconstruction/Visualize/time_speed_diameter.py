from __future__ import annotations

from pathlib import Path
from typing import Union, List, Dict, Optional, Sequence

import numpy as np
import tifffile  # pip install tifffile
from matplotlib import pyplot as plt
from openalea.mtg import MTG
from rsml import rsml2mtg
from rsml.misc import root_vertices
from scipy.ndimage import label as labelization

date_map = tifffile.imread("/home/loai/Documents/code/RSMLExtraction/RSA_deep_working/Data/Val/230629PN031/40_date_map.tif")
binary_map = date_map > 0
binary_map_labeled, label_count = labelization(binary_map)
print(f"Found {label_count} connected components in date_map")
mtg = rsml2mtg("/home/loai/Documents/code/RSMLExtraction/RSA_deep_working/Data/Val/230629PN031/61_graph.rsml")

primary_roots = [v for v in root_vertices(mtg) if mtg.parent(v) is None]
lateral_vertices = [v for v in root_vertices(mtg) if mtg.parent(v) is not None]

# for each primary root first node, get its corresponding connected component in date_map
geometries = mtg.properties().get('geometry', {})
plt.figure()
for v_primary in primary_roots:
    labelplant_datemap = np.zeros_like(date_map)
    if v_primary not in geometries:
        continue
    geom = geometries[v_primary] # gives [[x0,y0], [x1,y1], ...]
    if geom is None or len(geom) == 0:
        continue
    list_label_info = []
    max_time = 0
    min_time = 1e9
    for x, y in geom:
        x, y = int(x), int(y)        
        label_plant = binary_map_labeled[y, x] # number of the plant
        if label_plant == 0:
            continue
        label_date_map = date_map[y, x] # appearance date of the "excroissance"
        
        plant_and_time_filter = (binary_map_labeled == label_plant) & (date_map == label_date_map) # time and plant filter
        plant_and_time_filter = labelization(plant_and_time_filter)[0] # labeling of the connected components (assuming they are not connected)
        component_label = plant_and_time_filter[y, x] # getting the label of the connected component
        region_of_cc = (plant_and_time_filter == component_label)
        #plt.imshow(region_of_cc)
        #plt.title(f"Connected component for primary root {v_primary} at date {label_date_map}")
        #plt.show()
        labelplant_datemap[region_of_cc] = label_date_map # filling the connected component in the output map (recreating the datemap but for a single root)
        max_time = max(max_time, label_date_map)
        min_time = min(min_time, label_date_map)
        
    # show labelplant_datemap
    #plt.imshow(labelplant_datemap)
    #plt.title(f"Connected component for primary root {v_primary}")
    #plt.show()
    
    time_prop = mtg.properties().get("time", {})
    time_prop_root = time_prop.get(v_primary, [])
    time_in_hours_prop = mtg.properties().get("time_hours", {})
    time_in_hours_prop_root = time_in_hours_prop.get(v_primary, [])

    def _cumulated_lengths(geom: Sequence[Sequence[float]], time: Optional[Sequence[float]] = None, time_hours: Optional[Sequence[float]] = None) -> np.ndarray:
        time_length = {}  # clé: t (float), valeur: longueur
        time_length_in_hours = {}  # clé: t_en_heures (float), valeur: longueur
        dist = 0.0
        for t in time:
            time_length[t] = dist
            time_length_in_hours[time_hours[time.index(t)]] = dist
            if time.index(t) + 1 < len(geom):
                pt0 = geom[time.index(t)]
                pt1 = geom[time.index(t) + 1]
            else:
                pt0 = geom[time.index(t)]
                pt1 = geom[time.index(t)]
            dist += ((pt0[0] - pt1[0]) ** 2 + (pt0[1] - pt1[1]) ** 2) ** 0.5
        return time_length, time_length_in_hours

    root_length_at_time, root_length_at_time_hours = _cumulated_lengths(geom, time_prop_root, time_in_hours_prop_root)

    # Diameter = (Aire) / (Longueur totale)
    diameter_at_time = {}
    diameter_at_time_hours = {}
    for t in range(int(min_time), int(max_time + 1)):
        area = np.sum(labelplant_datemap == t)
        length = root_length_at_time.get(t, 0.0)
        diameter_at_time[t] = area / length if length > 0 else 0.0

    for t in diameter_at_time.keys():
        diameter_at_time_hours[time_in_hours_prop_root[time_prop_root.index(t)]] = diameter_at_time[t]

    # pixel size is 760µm
    pixel_size = 1
    root_length_at_time_hours = {k: v * pixel_size for k, v in root_length_at_time_hours.items()}
    diameter_at_time_hours = {k: v * pixel_size for k, v in diameter_at_time_hours.items()}
    
    speed_at_time_hours = {}
    previous_time = None
    previous_length = None
    for t in sorted(root_length_at_time_hours.keys()):
        length = root_length_at_time_hours[t]
        if previous_time is not None and previous_length is not None and t != previous_time:
            speed = (length - previous_length) / (t - previous_time)  # µm/hour
            speed_at_time_hours[t] = speed
        previous_time = t
        previous_length = length

    # (optionnel) visualisation rapide des courbes longueur/diamètre
    #plt.plot(list(root_length_at_time_hours.keys()), list(root_length_at_time_hours.values()))
    #plt.title(f"Longueur vs temps (indices) - racine {v_primary}"); plt.xlabel("temps (index)"); plt.ylabel("longueur (m)"); # plt.show()
    #plt.figure(); 
    #plt.plot(list(diameter_at_time_hours.keys()), list(diameter_at_time_hours.values()))
    #plt.title(f"Diamètre vs temps (indices) - racine {v_primary}"); plt.xlabel("temps (index)"); plt.ylabel("diamètre (m)"); plt.show()
    #plt.figure(); 
    plt.plot(list(speed_at_time_hours.keys()), list(speed_at_time_hours.values()))
    #plt.title(f"Vitesse de croissance vs temps (indices) - racine {v_primary}"); plt.xlabel("temps (index)"); plt.ylabel("vitesse (m/h)") ; plt.show()
#plt.show()

for v_lateral in lateral_vertices:
    labelplant_datemap = np.zeros_like(date_map)
    if v_lateral not in geometries:
        continue
    geom = geometries[v_lateral] # gives [[x0,y0], [x1,y1], ...]
    if geom is None or len(geom) == 0:
        continue
    list_label_info = []
    max_time = 0
    min_time = 1e9
    for x, y in geom:
        x, y = int(x), int(y)        
        label_plant = binary_map_labeled[y, x] # number of the plant
        if label_plant == 0:
            continue
        label_date_map = date_map[y, x] # appearance date of the "excroissance"
        
        plant_and_time_filter = (binary_map_labeled == label_plant) & (date_map == label_date_map) # time and plant filter
        plant_and_time_filter = labelization(plant_and_time_filter)[0] # labeling of the connected components (assuming they are not connected)
        component_label = plant_and_time_filter[y, x] # getting the label of the connected component
        region_of_cc = (plant_and_time_filter == component_label)
        #plt.imshow(region_of_cc)
        #plt.title(f"Connected component for primary root {v_lateral} at date {label_date_map}")
        #plt.show()
        labelplant_datemap[region_of_cc] = label_date_map # filling the connected component in the output map (recreating the datemap but for a single root)
        max_time = max(max_time, label_date_map)
        min_time = min(min_time, label_date_map)
        
    # show labelplant_datemap
    #plt.imshow(labelplant_datemap)
    #plt.title(f"Connected component for primary root {v_lateral}")
    #plt.show()
    
    time_prop = mtg.properties().get("time", {})
    time_prop_root = time_prop.get(v_lateral, [])
    time_in_hours_prop = mtg.properties().get("time_hours", {})
    time_in_hours_prop_root = time_in_hours_prop.get(v_lateral, [])

    def _cumulated_lengths(geom: Sequence[Sequence[float]], time: Optional[Sequence[float]] = None, time_hours: Optional[Sequence[float]] = None) -> np.ndarray:
        time_length = {}  # clé: t (float), valeur: longueur
        time_length_in_hours = {}  # clé: t_en_heures (float), valeur: longueur
        dist = 0.0
        for t in time:
            time_length[t] = dist
            time_length_in_hours[time_hours[time.index(t)]] = dist
            if time.index(t) + 1 < len(geom):
                pt0 = geom[time.index(t)]
                pt1 = geom[time.index(t) + 1]
            else:
                pt0 = geom[time.index(t)]
                pt1 = geom[time.index(t)]
            dist += ((pt0[0] - pt1[0]) ** 2 + (pt0[1] - pt1[1]) ** 2) ** 0.5
        return time_length, time_length_in_hours

    root_length_at_time, root_length_at_time_hours = _cumulated_lengths(geom, time_prop_root, time_in_hours_prop_root)

    # Diameter = (Aire) / (Longueur totale)
    diameter_at_time = {}
    diameter_at_time_hours = {}
    for t in range(int(min_time), int(max_time + 1)):
        area = np.sum(labelplant_datemap == t)
        length = root_length_at_time.get(t, 0.0)
        diameter_at_time[t] = area / length if length > 0 else 0.0

    for t in diameter_at_time.keys():
        try:
            diameter_at_time_hours[time_in_hours_prop_root[time_prop_root.index(t)]] = diameter_at_time[t]
        except ValueError:
            continue
    # pixel size is 80µm
    pixel_size = 1
    root_length_at_time_hours = {k: v * pixel_size for k, v in root_length_at_time_hours.items()}
    diameter_at_time_hours = {k: v * pixel_size for k, v in diameter_at_time_hours.items()}
    
    speed_at_time_hours = {}
    previous_time = None
    previous_length = None
    stop = False
    decalage = 0
    for t in sorted(root_length_at_time_hours.keys()):
        length = root_length_at_time_hours[t]
        if previous_time is not None and previous_length is not None and t != previous_time:
            if decalage == 0:
                decalage = t
            speed = (length - previous_length) / (t - previous_time)  # µm/hour
            if (abs(speed) >= 10): # 0.0009): 
                stop = True
            speed_at_time_hours[t - decalage] = speed
        previous_time = t
        previous_length = length
    if (stop): 
        stop = False
        continue

    # (optionnel) visualisation rapide des courbes longueur/diamètre
    #plt.plot(list(root_length_at_time_hours.keys()), list(root_length_at_time_hours.values()))
    #plt.title(f"Longueur vs temps (indices) - racine {v_primary}"); plt.xlabel("temps (index)"); plt.ylabel("longueur (µm)"); #plt.show()
    #plt.figure(); 
    #plt.plot(list(diameter_at_time_hours.keys()), list(diameter_at_time_hours.values()))
    #plt.title(f"Diamètre vs temps (indices) - racine {v_primary}"); plt.xlabel("temps (index)"); plt.ylabel("diamètre (µm)"); plt.show()
    #plt.figure(); 
    plt.plot(list(speed_at_time_hours.keys()), list(speed_at_time_hours.values()))
    #plt.title(f"Vitesse de croissance vs temps (indices) - racine {v_primary}"); plt.xlabel("temps (index)"); plt.ylabel("vitesse (µm/h)") ; plt.show()
plt.show()