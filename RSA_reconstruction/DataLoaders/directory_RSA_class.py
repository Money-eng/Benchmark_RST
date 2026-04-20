import os

import openalea.rsml as rsml
import tifffile
from utils.misc import set_seed, SEED

set_seed(SEED)


class LightRSAClass:
    def __init__(self, folder_path: str, load_date_map: bool = False, lazy: bool = True):
        self.folder_path = folder_path
        self.load_date_map_flag = load_date_map
        self.lazy = lazy

        self.image_stack_path = os.path.join(
            folder_path, "22_registered_stack.tif")
        self.date_map_path = os.path.join(folder_path, "40_date_map.tif")
        self.rsml_expert_file = os.path.join(
            folder_path, "61_graph_expertized.rsml")
        self.rsml_default_file = os.path.join(folder_path, "61_graph.rsml")

        self._image_stack = None
        self._date_map = None
        self._mtg = None

        if not self.lazy:
            self.load_all()

    @property
    def image_stack(self):
        if self._image_stack is None:
            if os.path.exists(self.image_stack_path):
                self._image_stack = tifffile.imread(self.image_stack_path)
            else:
                raise FileNotFoundError(
                    f"Image stack not found for {self.folder_path}, expected at {self.image_stack_path}.")
        return self._image_stack

    @property
    def date_map(self):
        if not self.load_date_map_flag:
            return None
        if self._date_map is None:
            if os.path.exists(self.date_map_path):
                self._date_map = tifffile.imread(self.date_map_path)
            else:
                print(
                    f"Date map not found for {self.folder_path}, expected at {self.date_map_path}. Returning None.")
        return self._date_map

    @property
    def mtg(self):
        if self._mtg is None:
            if os.path.exists(self.rsml_expert_file):
                self._mtg = rsml.rsml2mtg(self.rsml_expert_file)
            elif os.path.exists(self.rsml_default_file):
                self._mtg = rsml.rsml2mtg(self.rsml_default_file)
            else:
                raise FileNotFoundError(
                    "No RSML file found in the folder: "
                    f"{self.rsml_expert_file} or {self.rsml_default_file}"
                )
        return self._mtg

    def load_all(self):
        _ = self.image_stack  
        _ = self.mtg 
        if self.load_date_map_flag:
            _ = self.date_map

    def get_data(self):
        return {
            "image_stack": self.image_stack,
            "mtg": self.mtg,
            "date_map": self.date_map
        }


class DirectoryRSAClass:
    def __init__(self, base_dir: str, load_date_map: bool = False, lazy: bool = True):
        self.base_dir = base_dir
        self.load_date_map_flag = load_date_map
        self.lazy = lazy
        self.loaders = []
        self._scan_directories()

    def _scan_directories(self):
        for root, dirs, files in os.walk(self.base_dir):
            # On considère que le dossier est valide s'il contient l'image stack
            if "22_registered_stack.tif" in files:
                loader = LightRSAClass(
                    root, load_date_map=self.load_date_map_flag, lazy=self.lazy)
                self.loaders.append(loader)

    def get_loaders(self):
        return self.loaders

    def __iter__(self):
        return iter(self.loaders)

    def __len__(self):
        return len(self.loaders)

    def __getitem__(self, index):
        loader = self.loaders[index]
        return loader.get_data()


if __name__ == "__main__":
    base_directory = "/home/loai/Images/DataTest/UC1_data"

    dataset = DirectoryRSAClass(base_directory, load_date_map=True, lazy=True)

    print(f"{len(dataset)} dataset(s) trouvé(s) dans l'arborescence.")

    # try:
    #     data = dataset[0]
    #     for i in range(len(dataset)):
    #         print("RSA n°", i, " : ")
    #         print("  - Image stack, forme :", dataset[i]["image_stack"].shape)
    #         print("  - MTG, nombre de sommets :",
    #               len(dataset[i]["mtg"].vertices()))
    #         if dataset[i]["date_map"] is not None:
    #             print("  - Date map, forme :", dataset[i]["date_map"].shape)
    # except Exception as e:
    #     print("Erreur lors du chargement du dataset :", e)
