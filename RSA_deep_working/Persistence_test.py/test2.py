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
    
    import gudhi
    import numpy as np
    import matplotlib.pyplot as plt

    rows, cols = filter_map.shape

    # Fonction pour obtenir un index unique pour chaque pixel
    def index(i, j):
        return i * cols + j

    # Création du SimplexTree
    st = gudhi.SimplexTree()

    # Insertion des sommets avec leur valeur de filtration
    for i in range(rows):
        for j in range(cols):
            vertex = index(i, j)
            filtration_value = filter_map[i, j]
            st.insert([vertex], filtration=filtration_value)

    # Insertion des arêtes entre pixels adjacents (voisinage 4, par exemple)
    for i in range(rows):
        for j in range(cols):
            current = index(i, j)
            if i + 1 < rows:
                neighbor = index(i + 1, j)
                # Pour l'arête, on prend par exemple la valeur maximale des deux
                filtration_edge = max(filter_map[i, j], filter_map[i + 1, j])
                st.insert([current, neighbor], filtration=filtration_edge)
            if j + 1 < cols:
                neighbor = index(i, j + 1)
                filtration_edge = max(filter_map[i, j], filter_map[i, j + 1])
                st.insert([current, neighbor], filtration=filtration_edge)

    # Optionnel : ajuster la filtration pour respecter la propriété
    st.make_filtration_non_decreasing()

    # Calcul de la persistance
    st.compute_persistence()

    # Exemple : récupération et affichage du diagramme de persistance pour H0
    import numpy as np
    diag0 = np.array(st.persistence_intervals_in_dimension(0))

    plt.figure()
    plt.title("Diagramme de persistance - H0")
    plt.scatter(diag0[:, 0], diag0[:, 1], color='blue')
    plt.xlabel("Naissance")
    plt.ylabel("Mort")
    plt.show()


if __name__ == "__main__":
    main()
