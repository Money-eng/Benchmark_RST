import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize
from Root_System_class import RootSystem

def compute_skeleton_and_diameter(date_map, threshold=0):
    """
    Calcule le masque binaire, la transformée de distance, le squelette et la carte de diamètre.
    
    Args:
        date_map (ndarray): Image (grayscale) du date_map.
        threshold (float): Seuil pour binariser l'image (par défaut 0).
        
    Returns:
        skeleton (ndarray): Squelette binaire extrait du masque.
        diameter_map (ndarray): Carte où pour chaque pixel du squelette, la valeur est 2*distance,
                                correspondant au diamètre estimé.
    """
    mask = date_map > threshold
    dt = distance_transform_edt(mask)
    skeleton = skeletonize(mask)
    # Opération vectorisée : affectation du diamètre pour tous les pixels du squelette
    diameter_map = np.zeros_like(skeleton, dtype=float)
    diameter_map[skeleton] = 2 * dt[skeleton]
    
    return skeleton, diameter_map

def project_root_system_on_diameter_map(root_system: RootSystem, threshold=0):
    """
    Pour chaque vertex du système racinaire, trouve le point le plus proche sur le squelette
    et récupère le diamètre estimé à cet endroit.
    
    Args:
        root_system (RootSystem): Instance du système racinaire contenant notamment le date_map.
        threshold (float): Seuil pour la binarisation (par défaut 0).
    
    Returns:
        diameter_4_root_system (dict): Dictionnaire associant à chaque vertex (clé) le diamètre (valeur)
                                       sous forme de liste.
    """
    skeleton, diameter_map = compute_skeleton_and_diameter(root_system.date_map, threshold)
    
    # Récupération des coordonnées (indices) des pixels du squelette sous forme (row, col)
    skel_coords = np.column_stack(np.nonzero(skeleton))
    if skel_coords.shape[0] == 0:
        raise ValueError("Aucun pixel dans le squelette n'a été trouvé.")
    
    tree = cKDTree(skel_coords)
    diameter_4_root_system = {}

    for vertex, polyline in root_system.geometry.items():
        polyline = np.array(polyline)
        if polyline.size == 0:
            best_diameter = 0
        else:
            # Conversion des coordonnées (x, y) en indices (row, col)
            rows = np.rint(polyline[:, 1]).astype(int)
            cols = np.rint(polyline[:, 0]).astype(int)
            # Filtrer les points hors bornes
            valid = (rows >= 0) & (rows < skeleton.shape[0]) & (cols >= 0) & (cols < skeleton.shape[1])
            if not np.any(valid):
                best_diameter = 0
            else:
                valid_points = np.column_stack((rows[valid], cols[valid]))
                distances, indices = tree.query(valid_points)
                # Sélection du point avec la distance minimale
                best_index = np.argmin(distances)
                best_coord = skel_coords[indices[best_index]]
                best_diameter = diameter_map[best_coord[0], best_coord[1]]
        
        # Limitation du diamètre entre 4 et 9
        best_diameter = max(min(best_diameter, 9), 4)
        # Constitution de la liste pour ce vertex :
        # On crée d'abord une partie avec des zéros jusqu'au temps d'apparition,
        # puis on complète avec le diamètre constant pour le reste des time_hours.
        debut = int(root_system.time[vertex][0])
        nb_time_points = len(root_system.time_hours)
        diameter_list = [0] * debut + [float(best_diameter)] * (nb_time_points - debut)
        diameter_4_root_system[vertex] = diameter_list

    return diameter_4_root_system



def rafine_diameter(root_system: RootSystem, diameter_4_root_system: dict):
    """ 
    Hypothesis: 
    - Diameter + position gives us an (almost) centered shape that overlaps with the root structures.
    - The diameter of the root system can always increase in time. 
    - The diameter of a root is almost constant in space. (at a given time, for each root, we look at the diameter of each node to look for anomalies (median value +- 10 percent))
    """
    # For each root, we see if it countains a lot of "clear" pixels
    # If it does, we reduce the diameter of the root until these anomalies disappear

    
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
    
#### Maybe another time : 
def compute_skeleton_and_diameter_more(date_map, threshold=0, threshold2=1):
    """
    Calcule le masque binaire, la transformée de distance, le squelette et la carte de diamètre
    pour les pixels dont l'intensité est dans [threshold, threshold2].
    
    Args:
        date_map (ndarray): Image (grayscale) du date_map.
        threshold (float): Seuil inférieur pour la binarisation (par défaut 0).
        threshold2 (float): Seuil supérieur pour la binarisation (par défaut 1).
        
    Returns:
        skeleton (ndarray): Squelette binaire extrait du masque.
        diameter_map (ndarray): Carte où pour chaque pixel du squelette, la valeur est 2*distance,
                                ce qui correspond au diamètre estimé.
    """
    # Création du masque pour les intensités dans [threshold, threshold2]
    mask = (date_map > threshold) & (date_map <= threshold2)
    # Calcul de la transformée de distance
    dt = distance_transform_edt(mask)
    # Extraction du squelette centré
    skeleton = skeletonize(mask)
    # Calcul du diamètre : pour chaque pixel du squelette, le diamètre = 2 * distance
    diameter_map = np.zeros_like(skeleton, dtype=float)
    rows, cols = skeleton.shape
    for i in range(rows):
        for j in range(cols):
            if skeleton[i, j]:
                diameter_map[i, j] = 2 * dt[i, j]
    return skeleton, diameter_map


def project_root_system_on_diameter_maps(root_system: RootSystem, begin_threshold=0):
    """
    Pour chaque vertex du système racinaire, et pour chaque intervalle de seuil 
    [begin_threshold + i, begin_threshold + i + 1) (i allant de 0 à int(max_time)-1), 
    trouve le point du squelette le plus proche et récupère le diamètre estimé à cet endroit.
    
    Si le vertex n'est pas encore apparu (c'est-à-dire que le temps courant est inférieur à son temps de naissance),
    la valeur du diamètre est fixée à 0.
    
    Retourne un dictionnaire de la forme :
        { vertex: { time_index: diameter, ... }, ... }
        
    où time_index correspond à l'intervalle [begin_threshold+time_index, begin_threshold+time_index+1).
    """
    # Détermination du nombre d'intervalles temporels à partir du temps maximum
    times = root_system.mtg.properties()['time']
    max_time = 0
    for value in times.values():
        for t in value:
            if t > max_time:
                max_time = t
    num_steps = int(max_time)

    # Calcul pour chaque intervalle temporel du squelette, de la carte de diamètre et construction d'un KDTree
    skeleton_maps = {}
    diameter_maps = {}
    kdtrees = {}
    for i in range(num_steps):
        thr_low = 0
        thr_high = begin_threshold + i + 1
        skeleton, diam_map = compute_skeleton_and_diameter_more(root_system.date_map, thr_low, thr_high)
        skeleton_maps[i] = skeleton
        diameter_maps[i] = diam_map
        skel_coords = np.array(np.nonzero(skeleton)).T
        kdtrees[i] = cKDTree(skel_coords) if skel_coords.shape[0] > 0 else None

    # Pour chaque vertex, pour chaque intervalle temporel, recherche du diamètre le plus proche
    # On utilise root_system.time_hours pour obtenir le temps de naissance du vertex.
    diameter_4_root_system = {}
    for vertex, polyline in root_system.geometry.items():
        vertex_diameters = {}
        birth_time = root_system.time[vertex][0]  # temps de première apparition
        for i in range(num_steps):
            current_time = begin_threshold + i
            # Si le vertex n'est pas encore apparu à ce moment, on affecte 0
            if current_time < birth_time:
                vertex_diameters[i] = 0.0
                continue

            kd = kdtrees[i]
            if kd is None:
                vertex_diameters[i] = 0.0
                continue

            best_diam = None
            best_dist = np.inf
            for pt in polyline:
                row = int(round(pt[1]))
                col = int(round(pt[0]))
                if row < 0 or row >= skeleton_maps[i].shape[0] or col < 0 or col >= skeleton_maps[i].shape[1]:
                    continue
                dist, idx = kd.query([row, col])
                if dist < best_dist:
                    best_dist = dist
                    skel_coords = np.array(np.nonzero(skeleton_maps[i])).T
                    best_coord = skel_coords[idx]
                    best_diam = diameter_maps[i][best_coord[0], best_coord[1]]
            if best_diam is None:
                best_diam = 0.0
            vertex_diameters[i] = float(best_diam)
        # vertex_diameters to list
        vertex_diameters = [d for d in vertex_diameters.values()]
        diameter_4_root_system[vertex] = vertex_diameters
    rafine_diameter(root_system, diameter_4_root_system)
    return diameter_4_root_system

def rafine_diameter(root_system: RootSystem, diameter_4_root_system: dict):
    """ 
    Hypothesis: 
    - The diameter of the root system is always increasing in time. (for each node, if the diamter is bigger at t-i than at t, we set the diameter to the value of t)
    - The diameter of a root is almost constant in space. (at a given time, for each root, we look at the diameter of each node to look for anomalies (median value +- 10 percent))
    """
    # time refinement