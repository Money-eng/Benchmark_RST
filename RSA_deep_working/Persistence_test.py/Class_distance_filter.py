import sys
import os
import numpy as np
import importlib
from scipy.ndimage import distance_transform_edt

# Remontée de deux niveaux pour accéder à Data_loader
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.normpath(os.path.join(current_dir, "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

print(f"Chemin du projet : {project_root}")

# Import du module de chargement des données
module_name = "RSA_deep_working.Data_loader.class_data_loaders"
try:
    class_data_loaders = importlib.import_module(module_name)
    DirectoryRSAClass = class_data_loaders.DirectoryRSAClass
except ModuleNotFoundError as e:
    print(f"Erreur lors de l'importation du module {module_name} : {e}")
    sys.exit(1)

# Fonctions de calcul de distance map
def compute_distance_map_threshold(datemap, thresholdhigh=5, thresholdlow=0):
    """
    Calcule la distance minimale pour chaque pixel de l'objet (pixels > 0)
    par rapport aux pixels du fond (0) et applique deux seuils :
      - Les distances supérieures à thresholdhigh sont mises à 0
      - Les distances inférieures à thresholdlow sont également mises à 0.
    """
    mask_background = (datemap == 0)
    dist = distance_transform_edt(~mask_background)
    distance_map = np.where(dist > thresholdhigh, 0, dist)
    distance_map = np.where(dist < thresholdlow, 0, distance_map)
    return distance_map

# Classe de dataset pour la segmentation
class SegmentationDataset(DirectoryRSAClass):
    """
    Dataset pour entraîner un modèle de deep learning en segmentation.
    Pour chaque index, __getitem__ renvoie un tuple (image, date_map, mtg).
    Le chargement est lazy (les données sont chargées lors de l'appel à __getitem__).
    """
    def __getitem__(self, idx):
        # Appel au getitem de la classe parente qui gère (potentiellement) le lazy loading.
        data = super().__getitem__(idx)
        
        # On suppose que 'data' est un dictionnaire contenant les clés suivantes :
        # "image", "date_map" et "mtg"
        image = data.get("image")
        date_map = data.get("date_map")
        mtg = data.get("mtg")
        
        # Optionnel : calculer la distance map à partir du date_map
        # distance_map = compute_distance_map_threshold(date_map)
        # Vous pouvez ensuite ajouter la distance_map dans la structure de sortie si besoin.
        
        return image, date_map, mtg

def main():
    # Chemin vers le dossier contenant les datasets
    base_directory = "/home/loai/Images/DataTest/UC1_data"
    
    # Instanciation du dataset en mode lazy
    dataset = SegmentationDataset(base_directory, load_date_map=True, lazy=True)
    
    if len(dataset) == 0:
        print("Aucun dataset trouvé dans l'arborescence.")
        return

    # Exemple d'utilisation du __getitem__
    _, date_map, _ = dataset[0]
    
    filter_map = compute_distance_map_threshold(date_map)
    
    # Affichage du date_map
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(date_map, cmap='viridis')
    ax[0].set_title("Date map")
    ax[1].imshow(filter_map, cmap='viridis')
    ax[1].set_title("Distance map")
    plt.show()

if __name__ == "__main__":
    main()
