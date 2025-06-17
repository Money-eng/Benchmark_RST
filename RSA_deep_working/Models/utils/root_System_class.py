import numpy as np
import os
import shutil
import tifffile as tiff
import utils.custom_dumper as CD
from rsml import hirros, rsml2mtg


class RootSystem:
    """
    Classe pour charger un système racinaire (RSML → MTG) et calculer les propriétés.
    On peut fournir directement date_map pour éviter de recharger le fichier depuis le disque.
    """

    def __init__(self, folder_path: str, date_map: np.ndarray = None):
        self.folder_path = folder_path
        self.date_map = date_map
        self.mtg = None
        self.obs_times = None
        self.geometry = None
        self.time_hours = None
        self.time = None
        self._load_data()

    def _load_data(self):
        # Si date_map n'est pas fourni, on cherche le fichier "40_date_map.tif" dans folder_path
        if self.date_map is None:
            date_map_file = os.path.join(self.folder_path, "40_date_map.tif")
            if os.path.exists(date_map_file):
                self.date_map = tiff.imread(date_map_file)
            else:
                raise FileNotFoundError(f"Date map introuvable dans {self.folder_path}")

        # Chargement du RSML (expertisé ou non)
        expertized_rsml = os.path.join(self.folder_path, "61_graph_expertized.rsml")
        default_rsml = os.path.join(self.folder_path, "61_graph.rsml")
        if os.path.exists(expertized_rsml):
            self.mtg = rsml2mtg(expertized_rsml)
        elif os.path.exists(default_rsml):
            self.mtg = rsml2mtg(default_rsml)
        else:
            raise FileNotFoundError(f"Aucun fichier RSML (61_graph*.rsml) dans {self.folder_path}")

        # Extraction des propriétés MTG
        self.obs_times = hirros.times(self.mtg)
        self.geometry = self.mtg.property('geometry')
        self.time_hours = self.mtg.property('time_hours')
        self.time = self.mtg.property('time')

        # Mise à jour metadata image (optionnel)
        metadata = self.mtg.graph_properties().get('metadata', {})
        image_meta = metadata.get('image', {})
        if image_meta.get('name') is None:
            image_meta['name'] = os.path.basename("22_registered_stack.tif")
            metadata['image'] = image_meta

        # Section "functions" : on ajoute time, time_hours et diameter (si date_map fourni)
        if 'functions' not in metadata:
            metadata['functions'] = {}
        metadata['functions']['time'] = self.mtg.properties().get('time', {})
        metadata['functions']['time_hours'] = self.mtg.properties().get('time_hours', {})

        if 'diameter' not in metadata['functions']:
            if self.date_map is not None:
                # Calcul du diamètre à partir de date_map
                try:
                    
                    diameter = project_root_system_on_diameter_map(self)
                    metadata['functions']['diameter'] = diameter
                    self.mtg.add_property('diameter')
                    self.mtg.properties()['diameter'] = diameter
                except ImportError:
                    print("Module right_Diameter introuvable, diamètre non calculé.")
            else:
                print("Pas de date_map, diamètre non calculé.")
        self.mtg.graph_properties()['metadata'] = metadata

    def save2folder(self, destination_folder: str, save_date_map: bool = False):
        """
        Sauvegarde le fichier RSML et, optionnellement, la date_map dans destination_folder.
        """
        if self.mtg is None:
            raise ValueError("Aucun MTG chargé à sauvegarder.")

        os.makedirs(destination_folder, exist_ok=True)
        # Sauvegarde du RSML depuis le MTG
        rsml_path = os.path.join(destination_folder, "61_graph.rsml")
        mtg2rsml(self.mtg, rsml_path)
        print(f"RSML sauvegardé dans : {rsml_path}")

        if save_date_map and self.date_map is not None:
            date_map_path = os.path.join(destination_folder, "40_date_map.tif")
            tiff.imwrite(date_map_path, self.date_map.astype(np.float32))
            print(f"Date_map sauvegardée dans : {date_map_path}")

        # Copier fichier InfoSerieRootSystemTracker.csv si présent au même endroit que folder_path
        info_src = os.path.join(self.folder_path, "InfoSerieRootSystemTracker.csv")
        if os.path.exists(info_src):
            info_dst = os.path.join(destination_folder, "InfoSerieRootSystemTracker.csv")
            shutil.copy(info_src, info_dst)
            print(f"InfoSerieRootSystemTracker.csv copié dans : {info_dst}")
        else:
            print("InfoSerieRootSystemTracker.csv non trouvé, pas copié.")


# Re-write de mtg2rsml pour éviter l'erreur d'écriture
def mtg2rsml(g, rsml_file):
    """
    Write **continuous** mtg `g` in `rsml_file`
    :See also: `Dumper`, `rsml.continuous`
    """
    dump = CD.Dumper()
    s = dump.dump(g)
    if isinstance(rsml_file, str):
        with open(rsml_file, 'wb') as f:
            f.write(s)
    else:
        rsml_file.write(s)


######## DIAMETER CALCULATION ########

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize

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
        
        nb_time_points = len(root_system.time[vertex])
        diameter_list = [float(best_diameter)] * nb_time_points
        diameter_4_root_system[vertex] = diameter_list

    return diameter_4_root_system
   
   