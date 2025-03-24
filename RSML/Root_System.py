import os
import tifffile
import numpy as np
import networkx as nx
import unittest
from RSML_Parser import parse_rsml, rsml_to_nx

########################################
# Classe RootSystem
########################################

class RootSystem:
    def __init__(self, folder_path: str, load_date_map: bool = False):
        """
        Initialise le système racinaire en chargeant :
         - L'image stack (série temporelle d'images au format TIFF)
         - Le dateMap (optionnel)
         - Le fichier RSML, converti en graphe NetworkX
         
        :param folder_path: Chemin vers le dossier contenant les fichiers.
        :param load_date_map: Si True, charge également le dateMap (default: False).
        """
        self.folder_path = folder_path
        self.load_date_map = load_date_map

        # Attributs à charger
        self.image_stack = None
        self.date_map = None
        self.rsml_doc = None
        self.graph = None

        self._load_data()

    def _load_data(self):
        # Chargement de l'image stack
        image_stack_file = os.path.join(self.folder_path, "22_registered_stack.tif")
        if os.path.exists(image_stack_file):
            self.image_stack = tifffile.imread(image_stack_file)
        else:
            raise FileNotFoundError(f"Fichier image stack non trouvé: {image_stack_file}")

        # Chargement optionnel du dateMap
        if self.load_date_map:
            date_map_file = os.path.join(self.folder_path, "40_date_map.tif")
            if os.path.exists(date_map_file):
                self.date_map = tifffile.imread(date_map_file)
            else:
                print(f"Avertissement: dateMap non trouvé dans {self.folder_path}")

        # Recherche d'un fichier RSML
        expertized_rsml_file = os.path.join(self.folder_path, "61_graph_expertized.rsml")
        default_rsml_file = os.path.join(self.folder_path, "61_graph.rsml")
        rsml_file = None
        if os.path.exists(expertized_rsml_file):
            rsml_file = expertized_rsml_file
        elif os.path.exists(default_rsml_file):
            rsml_file = default_rsml_file
        else:
            raise FileNotFoundError(f"Aucun fichier RSML trouvé dans {self.folder_path}")

        # Parsing du fichier RSML
        self.rsml_doc = parse_rsml(rsml_file)
        # Conversion du RSML en graphe NetworkX
        self.graph = rsml_to_nx(self.rsml_doc)

    def get_graph(self) -> nx.DiGraph:
        """Retourne le graphe NetworkX généré à partir du RSML."""
        return self.graph

########################################
# Exemple de visualisation avec Napari (optionnel)
########################################

def visualize_with_napari(root_system: RootSystem):
    import napari

    # Cette fonction est un exemple de visualisation des données
    # en superposant l'image stack et le graphe (nœuds et arêtes).
    viewer = napari.Viewer()
    viewer.add_image(root_system.image_stack,
                     name="Image Stack",
                     colormap="gray",
                     contrast_limits=[root_system.image_stack.min(), root_system.image_stack.max()])

    # Préparation des données pour afficher les arêtes
    edges = []
    for u, v in root_system.graph.edges():
        # Récupération des coordonnées des nœuds
        node_u = root_system.graph.nodes[u]
        node_v = root_system.graph.nodes[v]
        if node_u.get("x") is not None and node_v.get("x") is not None:
            edges.append([[node_u["x"], node_u["y"]], [node_v["x"], node_v["y"]]])
    
    viewer.add_shapes(edges,
                      shape_type='line',
                      edge_color='red',
                      name='Graph Edges')
    viewer.add_points(
        np.array([[node["x"], node["y"]] for _, node in root_system.graph.nodes(data=True)
                  if node.get("x") is not None]),
        face_color='blue',
        size=5,
        name='Graph Nodes'
    )
    napari.run()

########################################
# Tests unitaires
########################################

class TestRootSystem(unittest.TestCase):
    def test_load_data_and_graph(self):
        # Pour ce test, on s'attend à ce qu'un dossier de test (test_data_folder)
        # existe et contienne les fichiers "22_registered_stack.tif" et un RSML (soit "61_graph_expertized.rsml" ou "61_graph.rsml").
        test_folder = "test_data_folder"  # À adapter selon votre environnement de test
        if not os.path.exists(test_folder):
            self.skipTest(f"Dossier de test '{test_folder}' inexistant.")
        rs = RootSystem(test_folder)
        self.assertIsNotNone(rs.image_stack, "L'image stack n'a pas été chargée.")
        self.assertIsNotNone(rs.graph, "Le graphe RSML n'a pas été généré.")
        self.assertGreater(rs.graph.number_of_nodes(), 0, "Le graphe ne contient aucun nœud.")

########################################
# Exécution en mode script
########################################

if __name__ == '__main__':
    # Exécution d'un test simple
    unittest.main(exit=False)

    # Exemple d'utilisation en direct (adapté à votre dossier de données)
    # Remplacez 'votre_dossier' par le chemin réel
    folder_path = "/home/loai/Images/DataTest/230629PN016/"
    if os.path.exists(folder_path):
        rs_system = RootSystem(folder_path)
        print("Graphe généré (nombre de nœuds):", rs_system.graph.number_of_nodes())
        # Vous pouvez lancer la visualisation avec Napari si souhaité :
        # visualize_with_napari(rs_system)
    else:
        print(f"Le dossier '{folder_path}' n'existe pas. Vérifiez votre chemin.")
