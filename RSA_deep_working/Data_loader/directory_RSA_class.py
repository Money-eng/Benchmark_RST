import os

import openalea.rsml as rsml
import tifffile


class LightRSAClass:
    def __init__(self, folder_path: str, load_date_map: bool = False, lazy: bool = True):
        """
        Initialise le DataLoader léger pour un dossier contenant les données.
        
        :param folder_path: Chemin vers le dossier contenant les fichiers.
        :param load_date_map: Si True, le date_map sera chargé (optionnel).
        :param lazy: Si True, le chargement se fera à la demande (lazy loading). 
                     Si False, toutes les données seront chargées immédiatement.
        """
        self.folder_path = folder_path
        self.load_date_map_flag = load_date_map
        self.lazy = lazy

        # Définition des chemins vers les fichiers
        self.image_stack_path = os.path.join(folder_path, "22_registered_stack.tif")
        self.date_map_path = os.path.join(folder_path, "40_date_map.tif")
        self.rsml_expert_file = os.path.join(folder_path, "61_graph_expertized.rsml")
        self.rsml_default_file = os.path.join(folder_path, "61_graph.rsml")

        # Initialisation des attributs (données)
        self._image_stack = None
        self._date_map = None
        self._mtg = None

        # Si lazy=False, charger immédiatement toutes les données
        if not self.lazy:
            self.load_all()

    @property
    def image_stack(self):
        if self._image_stack is None:
            if os.path.exists(self.image_stack_path):
                self._image_stack = tifffile.imread(self.image_stack_path)
            else:
                raise FileNotFoundError(f"Image stack non trouvé : {self.image_stack_path}")
        return self._image_stack

    @property
    def date_map(self):
        if not self.load_date_map_flag:
            return None
        if self._date_map is None:
            if os.path.exists(self.date_map_path):
                self._date_map = tifffile.imread(self.date_map_path)
            else:
                print(f"Avertissement : date_map non trouvé dans {self.date_map_path}")
        return self._date_map

    @property
    def mtg(self):
        if self._mtg is None:
            if os.path.exists(self.rsml_expert_file):
                self._mtg = rsml.rsml2mtg(self.rsml_expert_file)
            elif os.path.exists(self.rsml_default_file):
                self._mtg = rsml.rsml2mtg(self.rsml_default_file)
            else:
                raise FileNotFoundError("Aucun fichier RSML trouvé dans " + self.folder_path)
        return self._mtg

    def load_all(self):
        """
        Charge toutes les données (image stack, MTG et date_map si activé).
        """
        _ = self.image_stack  # Force le chargement de l'image stack
        _ = self.mtg  # Force le chargement du MTG
        if self.load_date_map_flag:
            _ = self.date_map

    def get_data(self):
        """
        Renvoie un dictionnaire contenant l'image stack, le MTG et, éventuellement, le date_map.
        Cette méthode est appelée dans __getitem__ pour retourner les données.
        
        :return: dict avec les clés 'image_stack', 'mtg' et 'date_map'
        """
        return {
            "image_stack": self.image_stack,
            "mtg": self.mtg,
            "date_map": self.date_map
        }


class DirectoryRSAClass:
    def __init__(self, base_dir: str, load_date_map: bool = False, lazy: bool = True):
        """
        Parcourt de manière récursive l'arborescence à partir de base_dir et
        crée une instance LightRSAClass pour chaque dossier contenant les fichiers requis.
        
        :param base_dir: Répertoire racine contenant les sous-dossiers.
        :param load_date_map: Si True, le date_map sera chargé pour chaque dataset.
        :param lazy: Si True, chaque LightRSAClass utilise le chargement paresseux.
        """
        self.base_dir = base_dir
        self.load_date_map_flag = load_date_map
        self.lazy = lazy
        self.loaders = []
        self._scan_directories()

    def _scan_directories(self):
        """
        Parcourt l'arborescence des fichiers et ajoute dans self.loaders tous les dossiers
        contenant le fichier '22_registered_stack.tif'.
        """
        for root, dirs, files in os.walk(self.base_dir):
            # On considère que le dossier est valide s'il contient l'image stack
            if "22_registered_stack.tif" in files:
                loader = LightRSAClass(root, load_date_map=self.load_date_map_flag, lazy=self.lazy)
                self.loaders.append(loader)

    def get_loaders(self):
        """Retourne la liste des LightRSAClass trouvés."""
        return self.loaders

    def __iter__(self):
        return iter(self.loaders)

    def __len__(self):
        return len(self.loaders)

    def __getitem__(self, index):
        """
        Renvoie le dataset correspondant à l'index donné sous forme d'un dictionnaire.
        Si le chargement est lazy, les données seront chargées lors de cet appel.
        """
        loader = self.loaders[index]
        return loader.get_data()


# Exemple d'utilisation pour un modèle de segmentation deep learning
if __name__ == "__main__":
    base_directory = "/home/loai/Images/DataTest/UC1_data"

    # Création d'un DirectoryRSAClass pour parcourir l'arborescence
    dataset = DirectoryRSAClass(base_directory, load_date_map=True, lazy=True)

    print(f"{len(dataset)} dataset(s) trouvé(s) dans l'arborescence.")

    # Exemple d'utilisation du getitem pour récupérer le dataset n°0
    try:
        data = dataset[0]  # Renvoie un dictionnaire contenant 'image_stack', 'mtg' et 'date_map'
        for i in range(len(dataset)):
            print("RSA n°", i, " : ")
            print("  - Image stack, forme :", dataset[i]["image_stack"].shape)
            print("  - MTG, nombre de sommets :", len(dataset[i]["mtg"].vertices()))
            if dataset[i]["date_map"] is not None:
                print("  - Date map, forme :", dataset[i]["date_map"].shape)
    except Exception as e:
        print("Erreur lors du chargement du dataset :", e)
