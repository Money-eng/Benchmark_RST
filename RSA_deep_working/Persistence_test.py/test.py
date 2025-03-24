import sys
import os
import numpy as np
import importlib
from scipy.ndimage import distance_transform_edt
from gudhi.representations import PersistenceImage


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
    
    import gudhi as gd
    import matplotlib.pyplot as plt
    
    # plot selon chaques step du filtrage
    fig, ax = plt.subplots(2, 3, figsize=(15, 10))
    for i in range(0, 6):
        plt.subplot(2, 3, i+1)
        plt.imshow(compute_distance_map_threshold(date_map, thresholdhigh=i, thresholdlow=i-1), cmap='viridis')
        plt.title(f"Distance map (thresholdhigh={i}, thresholdlow={i-1})")
    plt.show()
    

    # Supposons que filter_map est votre image filtrée obtenue avec compute_distance_map_threshold
    # Par exemple : filter_map = compute_distance_map_threshold(date_map, thresholdhigh=5, thresholdlow=0)

    # Créer le complexe cubique à partir de filter_map
    cubical_complex = gd.CubicalComplex(top_dimensional_cells=filter_map)
    cubical_complex.set_verbose(True)

    # Calculer la persistance
    cubical_complex.compute_persistence()
    
    # Récupérer et afficher le diagramme de persistance H0 
    gd.plot_persistence_diagram(cubical_complex.persistence(), legend=True)
    plt.title("Diagramme de persistance (CubicalComplex)")
    
    
    # Visualisation de chaque valeur de filtration avec la fonction plot de gudhi
    gd.plot_persistence_barcode(cubical_complex.persistence())
    plt.title("Barcode de persistance (CubicalComplex)")
    
    
    # peritence image
    gd.plot_persistence_density(cubical_complex.persistence())
    plt.title("Density of persistence (CubicalComplex)")
    plt.show()
    
    

if __name__ == "__main__":
    main()
