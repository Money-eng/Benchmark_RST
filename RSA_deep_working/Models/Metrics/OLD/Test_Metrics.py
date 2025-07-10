import importlib
import os
import sys

import numpy as np

# Remontée de deux niveaux pour accéder à Data_loader
current_dir = os.getcwd()
project_root = os.path.normpath(os.path.join(current_dir))
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

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import rsml
import tifffile

from RSA_deep_working.Data_loader.class_data_loaders import DirectoryRSAClass

# importing pretrained segmentation model
import segmentation_models_pytorch as smp
from torch.utils.tensorboard import SummaryWriter

base_directory = "/home/loai/Images/DataTest/UC1_data"

from RSA_deep_working.Data_loader.class_data_loaders import DirectoryRSAClass
import os
import numpy as np
import tifffile
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

# Dimensions d'origine et calculs
H, W = (1166, 1348)

H_new, W_new = (1184, 1376)
padding_bottom, padding_right = H_new - H, W_new - W  # the opposite lol
print("pad :", padding_right, padding_bottom)

# Transformations pré-calculées
img_transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Resize((H_new, W_new), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.Pad(padding=(0, 0, padding_right, padding_bottom), fill=0),
    transforms.Normalize(mean=[0.5], std=[0.5])
])
mask_transform_series = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Resize((H_new, W_new), interpolation=transforms.InterpolationMode.NEAREST),
    transforms.Pad(padding=(0, 0, padding_right, padding_bottom), fill=0),

])
mask_transform_image = transforms.Compose([
    transforms.ToTensor(),

    # transforms.Resize((H_new, W_new), interpolation=transforms.InterpolationMode.NEAREST),
    transforms.Pad(padding=(0, 0, padding_right, padding_bottom), fill=0),
])


def mtg_transform(mtg):
    """
    Transforme le MTG en un tenseur PyTorch.
    """
    # Convertir le MTG en une représentation adaptée
    # Par exemple, convertir les coordonnées en tenseur
    # ou appliquer d'autres transformations spécifiques
    return mtg


class CachedTiffReader:
    def __init__(self):
        self.cache = {}

    def get_page(self, img_path, key):
        if img_path not in self.cache:
            # Chargement unique du fichier, stockage de toutes les pages
            print(f"Chargement de {img_path} dans le cache.")
            with tifffile.TiffFile(img_path) as tif:
                self.cache[img_path] = [page.asarray() for page in tif.pages]
        return self.cache[img_path][key]


tiff_reader = CachedTiffReader()


class RSASeg2DDataset(Dataset):
    def __init__(self, rsa_dir_loader, mode='series', img_transform=None,
                 mask_transform_series=None, mask_transform_image=None, image_with_mtg=False, as_RGB=False):
        """
        mode: 'series' pour charger l'ensemble de la série temporelle,
              'image' pour charger image par image.
        """
        self.mode = mode
        self.samples = []  # contiendra les tuples en fonction du mode
        self.img_transform = img_transform
        self.mask_transform_series = mask_transform_series
        self.mask_transform_image = mask_transform_image
        self.image_with_mtg = image_with_mtg
        self.as_RGB = as_RGB

        for loader in rsa_dir_loader.loaders:
            img_path = loader.image_stack_path
            mask_path = loader.date_map_path
            mtg_path = loader.rsml_default_file if os.path.exists(loader.rsml_default_file) else loader.rsml_expert_file

            # Lecture du nombre de slices dans la série
            with tifffile.TiffFile(img_path) as tif:
                num_slices = len(tif.pages)

            if mode == 'series':
                # Une entrée par série temporelle complète, on stocke num_slices
                self.samples.append((img_path, mask_path, num_slices, mtg_path))
            elif mode == 'image':
                # Une entrée par image (slice)
                for z in range(num_slices):
                    self.samples.append((img_path, mask_path, z, mtg_path))
            else:
                raise ValueError("Mode non reconnu, choisissez 'series' ou 'image'")

    def __len__(self):
        return len(self.samples)

    def num_times(self, idx):
        """
        Retourne le nombre de slices (temps) que comporte la série.
        """
        img_path, _, _, _ = self.samples[idx]
        with tifffile.TiffFile(img_path) as tif:
            num_slices = len(tif.pages)
        return num_slices

    def __getitem__(self, idx):
        # Extraction des informations de l'échantillon
        img_path, mask_path, slice_info, mtg_path = self.samples[idx]

        if self.mode == 'series':
            # Mode série : chargement complet de la série d'images
            img = tifffile.imread(img_path)
            mask_raw = tifffile.imread(mask_path)
            extra_info = slice_info  # ici, slice_info correspond à num_slices

            if self.img_transform:
                img = self.img_transform(img)
            # Appliquer la transformation du masque ou utiliser le masque brut
            mask = self.mask_transform_series(mask_raw) if self.mask_transform_series else mask_raw

        else:  # mode 'image'
            # Mode image : slice_info correspond à l'index de la slice (z)
            z = slice_info
            img = tiff_reader.get_page(img_path, z)
            mask_raw = tifffile.imread(mask_path)
            extra_info = z

            # Calcul vectorisé du masque : les pixels non nuls et <= z valent 1, sinon 0
            print(z + 1)
            mask = np.where((mask_raw != 0) & (mask_raw <= z + 1), 1, 0)

            if self.img_transform:
                img = self.img_transform(img)
            mask = self.mask_transform_image(mask) if self.mask_transform_image else mask

        if (self.as_RGB):
            img = img.repeat(3, 1, 1)
        # Gestion du retour : si image_with_mtg est activé, on retourne mtg_path, sinon un tensor nul
        additional = mtg_path if self.image_with_mtg else torch.tensor(0)
        # mask to float64
        mask = mask.float()
        return img, mask, extra_info, additional


dir_loader = DirectoryRSAClass(base_directory, load_date_map=True, lazy=True)

# Pour entraîner image par image :
rsa_dataset_image = RSASeg2DDataset(
    dir_loader,
    mode='image',
    img_transform=img_transform,
    mask_transform_image=mask_transform_image,  # pipeline dédié
    image_with_mtg=True
)

print("Nombre d'échantillons :", len(rsa_dataset_image), "images\n")

# for reproducibility
torch.manual_seed(42)
np.random.seed(42)

generator = torch.Generator().manual_seed(42)

train_set, val_set, test_set = torch.utils.data.random_split(
    rsa_dataset_image,
    [int(len(rsa_dataset_image) * 0.7), int(len(rsa_dataset_image) * 0.2),
     int(len(rsa_dataset_image) * 0.1) + 2],
    generator=generator
)

# Affichage des tailles des ensembles
print("Ensemble d'entraînement (image) :", len(train_set))
print("Ensemble de validation (image) :", len(val_set))
print("Ensemble de test (image) :", len(test_set))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Utilisation du device : {device}")

BATCH_SIZE = 2
# data loader optimization
train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=8,
                                           pin_memory=True,
                                           worker_init_fn=lambda worker_id: np.random.seed(42 + worker_id))
val_loader = torch.utils.data.DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True,
                                         worker_init_fn=lambda worker_id: np.random.seed(42 + worker_id))
test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=8,
                                          pin_memory=True,
                                          worker_init_fn=lambda worker_id: np.random.seed(42 + worker_id))

model_BCE = smp.Unet(
    encoder_name="resnet34",
    encoder_depth=5,
    encoder_weights=None,  # "imagenet",
    in_channels=1,
    classes=1
)

model_Dice = smp.Unet(
    encoder_name="resnet34",
    encoder_depth=5,
    encoder_weights=None,  # "imagenet",
    in_channels=1,
    classes=1
)

model_clDice = smp.Unet(
    encoder_name="resnet34",
    encoder_depth=5,
    encoder_weights=None,  # "imagenet",
    in_channels=1,
    classes=1
)

model_Dice_clDice = smp.Unet(
    encoder_name="resnet34",
    encoder_depth=5,
    encoder_weights=None,  # "imagenet",
    in_channels=1,
    classes=1
)

model_skRecall = smp.Unet(
    encoder_name="resnet34",
    encoder_depth=5,
    encoder_weights=None,  # "imagenet",
    in_channels=1,
    classes=1
)

model_superVoxel = smp.Unet(
    encoder_name="resnet34",
    encoder_depth=5,
    encoder_weights=None,  # "imagenet",
    in_channels=1,
    classes=1
)

model_skseg = smp.Unet(
    encoder_name="resnet34",
    encoder_depth=5,
    encoder_weights=None,  # "imagenet",
    in_channels=1,
    classes=1
)

# Charger les poids pré-entraînés si disponibles
model_BCE.to(device)
model_Dice.to(device)
model_clDice.to(device)
model_skRecall.to(device)
model_superVoxel.to(device)

List_models = [model_BCE, model_Dice, model_clDice, model_Dice_clDice, model_skRecall, model_superVoxel, model_skseg]

from PIL import Image


def tensor_to_heatmap_image(tensor, cmap='hot'):
    """
    Convertit un tenseur PyTorch en une heatmap image qui sera sauvagardée dans tensorboard.
    """
    # Normaliser le tenseur entre 0 et 1
    tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())

    cpu_tensor = tensor.cpu()

    # Convertir le tenseur en image PIL
    image = Image.fromarray((cpu_tensor.numpy() * 255).astype(np.uint8), mode='L')

    # Convertir l'image PIL en tableau numpy
    image_np = np.array(image)

    # Créer la heatmap
    heatmap = plt.get_cmap(cmap)(image_np)[:, :, :3]  # Ignorer l'alpha channel
    heatmap = (heatmap * 255).astype(np.uint8)  # Convertir en uint8

    return heatmap


import torch
import numpy as np
from tqdm import tqdm


def evaluate_segmentation_on_loader(loader, metrics: list, device='cpu'):
    """
    Évalue le modèle sur l'ensemble d'un DataLoader et accumule les scores pour chaque métrique.
    Enregistre également quelques exemples d'images, masques, prédictions et heatmaps dans TensorBoard.

    Args:
        model: Le modèle de segmentation.
        loader: DataLoader contenant les batches (images, masques, temps, mtgs).
        metrics (list): Liste de fonctions de métriques à évaluer.
        threshold (float): Seuil pour la binarisation.
        writer (SummaryWriter, optionnel): Writer TensorBoard pour loguer les métriques.
        global_step (int, optionnel): Pas global pour TensorBoard.
        device (str): Périphérique (cpu, cuda, etc.).

    Returns:
        dict: Dictionnaire contenant les scores finaux, ainsi que la concaténation des prédictions et masques.
    """

    metric_scores = {metric.__name__: [] for metric in metrics}

    with torch.no_grad():
        for batch in tqdm(
                loader,
                desc="Evaluation iteration",
                bar_format="{l_bar}{bar}{r_bar}",
                unit="batch",
                total=len(loader),
                position=0,
                leave=True,
                dynamic_ncols=True
        ):
            images, masks, time, mtgs = batch
            images, masks = images.to(device), masks.to(device)

            # Calcul et accumulation des métriques
            for metric in metrics:
                predictions = masks.clone()
                score = metric(predictions, masks, time, mtg=mtgs[0])
                metric_scores[metric.__name__].append(score)


import RSA_deep_working.Metrics.simple_metrics as sm
import RSA_deep_working.Metrics.topo_explicit_metrics as tm

metrics = sm.all_metrics()
tubular_metrics = tm.all_metrics()
connectivity_metric = tm.Connectivity_Preserving_Instance_Segmentation
all_metrics = []
for metric in metrics:
    all_metrics.append(metric)
for metric in tubular_metrics:
    all_metrics.append(metric)

if __name__ == "__main__":
    # get number of times of first image of the val_loader
    num_times = rsa_dataset_image.num_times(0)
    print("Nombre de temps :", num_times)
    dic_image_mask = {}
    for i in range(num_times):
        dic_image_mask[rsa_dataset_image[i][0][0]] = rsa_dataset_image[i][1][0]

    # for each metric in all_metrics, compute the metric for each mask and mask clone, print the name and the score 
    for metric in all_metrics:
        print(f"Metric: {metric.__name__}")
        for i in range(num_times):
            mask = rsa_dataset_image[i][1].cuda()
            mask_clone = rsa_dataset_image[i][1].clone().cuda()
            score = connectivity_metric(mask_clone, mask)
            print(f"Image {i}: {score}")
            print(f"Metric: {metric.__name__} - Score: {score}")
        print("\n")

    # superpose image (in green) and mask (in red)
    # for i in range(num_times+5):
    #   image = rsa_dataset_image[i][0][0].cpu().numpy()
    #  mask = rsa_dataset_image[i][1][0].cpu().numpy()
    # plt.imshow(image, cmap='gray')
    # plt.imshow(mask, cmap='jet', alpha=0.5)
    #  plt.axis('off')
    # plt.show()
