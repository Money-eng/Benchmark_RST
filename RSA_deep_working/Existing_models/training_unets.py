import sys
import os
import numpy as np
import importlib

# Remontée de deux niveaux pour accéder à Data_loader
current_dir = os.getcwd()
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

# %%
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
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


# %% [markdown]
# ## Dataset and data loaders

# %%
base_directory = "/home/loai/Test/data/UC1_data"

# %%
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
padding_bottom, padding_right = H_new - H, W_new - W # the opposite lol
print("pad :", padding_right, padding_bottom)

# Transformations pré-calculées
img_transform = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Resize((H_new, W_new), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.Pad(padding=(0, 0, padding_right, padding_bottom), fill=0),
    transforms.Normalize(mean=[0.5], std=[0.5])
])
mask_transform_series = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Resize((H_new, W_new), interpolation=transforms.InterpolationMode.NEAREST),
    transforms.Pad(padding=(0, 0, padding_right, padding_bottom), fill=0),
    
])
mask_transform_image = transforms.Compose([
    transforms.ToTensor(),
    
    #transforms.Resize((H_new, W_new), interpolation=transforms.InterpolationMode.NEAREST),
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

# %%
# Optimisation de la lecture des TIFF : mise en cache par fichier
class CachedTiffReader:
    def __init__(self):
        self.cache = {}

    def get_page(self, img_path, key):
        if img_path not in self.cache:
            # Chargement unique du fichier, stockage de toutes les pages
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
            mask = np.where((mask_raw != 0) & (mask_raw <= z + 1), 1, 0)

            if self.img_transform:
                img = self.img_transform(img)
            mask = self.mask_transform_image(mask) if self.mask_transform_image else mask

        if (self.as_RGB):
            img = img.repeat(3, 1, 1)
        # Gestion du retour : si image_with_mtg est activé, on retourne mtg_path, sinon un tensor nul
        additional = mtg_path if self.image_with_mtg else torch.tensor(0)
        #mask to float64 
        mask = mask.float()
        return img, mask, extra_info, additional


# %%

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
    int(len(rsa_dataset_image) * 0.1)+2],
    generator=generator
)

# Affichage des tailles des ensembles
print("Ensemble d'entraînement (image) :", len(train_set))
print("Ensemble de validation (image) :", len(val_set))
print("Ensemble de test (image) :", len(test_set))

# %% [markdown]
# ### 2D Image loaders

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Utilisation du device : {device}")

BATCH_SIZE = 4
# data loader optimization
train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True, worker_init_fn=lambda worker_id: np.random.seed(42 + worker_id))
val_loader = torch.utils.data.DataLoader(val_set, batch_size= 3* BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True, worker_init_fn=lambda worker_id: np.random.seed(42 + worker_id))
test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True, worker_init_fn=lambda worker_id: np.random.seed(42 + worker_id))

# %% [markdown]
# ## Model

# %%
model_BCE = smp.Unet(
    encoder_name="resnet34",
    encoder_depth=5,
    encoder_weights=None, # "imagenet",    
    in_channels=1,                 
    classes=1                      
)

model_Dice = smp.Unet(
    encoder_name="resnet34",       
    encoder_depth=5,
    encoder_weights=None, # "imagenet",    
    in_channels=1,                 
    classes=1                      
)

model_clDice = smp.Unet(
    encoder_name="resnet34",       
    encoder_depth=5,
    encoder_weights=None, # "imagenet",    
    in_channels=1,                 
    classes=1                      
)

model_Dice_clDice = smp.Unet(
    encoder_name="resnet34",       
    encoder_depth=5,
    encoder_weights=None, # "imagenet",    
    in_channels=1,                 
    classes=1                      
)

model_skRecall = smp.Unet(
    encoder_name="resnet34",       
    encoder_depth=5,
    encoder_weights=None, # "imagenet",    
    in_channels=1,                 
    classes=1                      
)

model_superVoxel = smp.Unet(
    encoder_name="resnet34",       
    encoder_depth=5,
    encoder_weights=None, # "imagenet",    
    in_channels=1,                 
    classes=1                      
)

# Charger les poids pré-entraînés si disponibles
model_clDice.to(device)
model_Dice_clDice.to(device)
model_skRecall.to(device)
model_superVoxel.to(device)

List_models = [model_clDice, model_Dice_clDice, model_skRecall, model_superVoxel] # model_BCE, model_Dice,

# %% [markdown]
# ## Evaluation

# %%
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

# %%
import torch
import numpy as np
from tqdm import tqdm
import torchvision.transforms.functional as TF


def evaluate_segmentation(model, image, mask, mtg, metrics: list, prediction=None, threshold=0.5, writer=None, global_step=None, device='cpu'):
    """
    Évalue le modèle sur une image (ou un batch d'images) et calcule les métriques fournies.

    Args:
        model: Le modèle de segmentation.
        image: Image ou batch d'images.
        mask: Masque(s) correspondant(s).
        mtg: Informations supplémentaires (ex. métadonnées).
        metrics (list): Liste de fonctions de métriques à évaluer.
        prediction (tensor, optionnel): Prédiction pré-calculée.
        threshold (float): Seuil pour la binarisation.
        writer (SummaryWriter, optionnel): Writer TensorBoard pour loguer les métriques.
        global_step (int, optionnel): Pas global pour TensorBoard.
        device (str): Périphérique (cpu, cuda, etc.).

    Returns:
        dict: Dictionnaire contenant la prédiction et les scores calculés.
    """
    model.eval()
    image, mask = image.to(device), mask.to(device)

    # Calcul de la prédiction si non fournie
    if prediction is None:
        with torch.no_grad():
            output = model(image)
            prediction = (torch.sigmoid(output) > threshold).float()

    # Calcul des métriques via compréhension de dictionnaire
    scores = {metric.__name__: metric(
        prediction, mask, mtg=mtg) for metric in metrics}

    # Log des métriques dans TensorBoard si nécessaire
    if writer is not None and global_step is not None:
        for name, score in scores.items():
            writer.add_scalar(f"Eval/{name}", score, global_step)

    return {'prediction': prediction, 'scores': scores}


def evaluate_segmentation_on_loader(model, loader, metrics: list, threshold=0.5, writer=None, global_step=None, device='cpu'):
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
    model.eval()
    metric_scores = {metric.__name__: [] for metric in metrics}
    sample_batch = None  # Pour conserver un batch d'exemple pour l'affichage

    with torch.no_grad():
        for batch in tqdm(
            loader,
            desc="Evaluation",
            bar_format="{l_bar}{bar}{r_bar}",
            unit="batch",
            position=0,
            leave=True,
            dynamic_ncols=True
        ):
            images, masks, time, mtgs = batch
            images, masks = images.to(device), masks.to(device)
            output = model(images)
            prediction = (torch.sigmoid(output) > threshold).float()

            # Calcul et accumulation des métriques
            for metric in metrics:
                score = metric(prediction, masks, time, mtg=mtgs[0])
                metric_scores[metric.__name__].append(score)

            # Enregistre le premier batch pour affichage dans TensorBoard
            if sample_batch is None and writer is not None and global_step is not None:
                sample_batch = (images.cpu(), masks.cpu(), output.cpu(), prediction.cpu())
            

    # Log des images si writer et global_step sont fournis et un batch exemple est disponible
    if writer is not None and global_step is not None:
        # for each metric, on calcule la moyenne
        for metric_name, scores in metric_scores.items():
            mean_score = np.mean(scores)
            writer.add_scalar(f"Eval_Mean/{metric_name}_mean", mean_score, global_step)
        
        # for each metric, on calcule la variance
        for metric_name, scores in metric_scores.items():
            var_score = np.var(scores)
            writer.add_scalar(f"Eval_Var/{metric_name}_var", var_score, global_step)
    
    
        if sample_batch is not None:
            sample_images, sample_masks, sample_outputs, sample_preds = sample_batch
            n_samples = min(4, sample_images.shape[0])

            images = sample_images[:n_samples]
            masks = sample_masks[:n_samples]
            predictions = sample_preds[:n_samples]
            outputs = sample_outputs[:n_samples]

            sigmoid_heatmaps = []
            outputs_heatmaps = []
            for i in range(n_samples):
                # Supposons que l'output et la sigmoid sont des cartes 2D (dimension [1, H, W])
                out = outputs[i].squeeze()
                sig_out = torch.sigmoid(outputs[i]).squeeze()
                
                # Création de l'image heatmap
                out_img = tensor_to_heatmap_image(out, cmap='hot')
                sig_img = tensor_to_heatmap_image(sig_out, cmap='hot')
                
                # Conversion en tensor
                out_tensor = TF.to_tensor(out_img)
                sig_tensor = TF.to_tensor(sig_img)
                outputs_heatmaps.append(out_tensor)
                sigmoid_heatmaps.append(sig_tensor)
            
            outputs_heatmaps_tensor = torch.stack(outputs_heatmaps)
            sigmoid_heatmaps_tensor = torch.stack(sigmoid_heatmaps)
            
            # concaténation des images du sample
            images_concat = torch.cat([images[i] for i in range(n_samples)], dim=2)
            masks_concat = torch.cat([masks[i] for i in range(n_samples)], dim=2)
            predictions_concat = torch.cat([predictions[i] for i in range(n_samples)], dim=2)
            outputs_heatmaps_concat = torch.cat([outputs_heatmaps_tensor[i] for i in range(n_samples)], dim=2)
            sigmoid_heatmaps_concat = torch.cat([sigmoid_heatmaps_tensor[i] for i in range(n_samples)], dim=2)
            # Enregistrement des images concaténées dans TensorBoard
            writer.add_image("Sample/Images", images_concat, global_step)
            writer.add_image("Sample/Masks", masks_concat, global_step)  
            writer.add_image("Sample/Predictions", predictions_concat, global_step)
            writer.add_image("Sample/Heatmaps", outputs_heatmaps_concat, global_step)
            writer.add_image("Sample/Sigmoid_Heatmaps", sigmoid_heatmaps_concat, global_step)
    return metric_scores

# %%
import RSA_deep_working.Metrics.simple_metrics as sm
import RSA_deep_working.Metrics.topo_explicit_metrics as tm

metrics = sm.all_metrics() 
tubular_metrics = tm.all_metrics()    
all_metrics = []
for metric in metrics:
    all_metrics.append(metric)
for metric in tubular_metrics:
    all_metrics.append(metric)

# %% [markdown]
# ## Training

# %%
# Train the model
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device='cpu', writer=None, writer_name=None, scheduler=None, lr_change_threshold=1e-5):
    torch.cuda.empty_cache()
    model.train()
    old_metric_scores = None
    # mkdir writer if not exist in checkpoints
    writer_folder = f"checkpoints/{writer_name}_Model"
    if not os.path.exists(writer_folder):
        os.makedirs(writer_folder)
    
    list_train_loss = []
    lr_deacrease = 0
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch in tqdm(
                train_loader, 
                desc=f"Loss: {running_loss:.1e} / {len(train_loader)}, LR: {optimizer.param_groups[0]['lr']:.1e}", 
                unit="batch", 
                postfix=f"Epoch: {epoch+1}/{num_epochs}",
                leave=True, 
                dynamic_ncols=True
            ):
            
            images, masks, _, _ = batch
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            # 1 - that loss is the cldice loss
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            

        epoch_loss = running_loss / len(train_loader)
        
        list_train_loss.append(epoch_loss)
        torch.cuda.empty_cache()
        
        # if the loss didn't change more than 5 times in a row, we decrease the learning rate with a factor 0.8
        if len(list_train_loss) > 5 and all(abs(list_train_loss[-i] - list_train_loss[-i-1]) < lr_change_threshold for i in range(1, 5)):
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.8
                print(f"Learning rate decreased to {param_group['lr']:.1e}")
                lr_deacrease += 1
                
        # if we decreased the learning rate more than 5 times and if the loss still didn't change, we stop the training
        if lr_deacrease > 3 and len(list_train_loss) > 5 and all(abs(list_train_loss[-i] - list_train_loss[-i-1]) < lr_change_threshold for i in range(1, 5)):
            print("Learning rate didn't change for 5 epochs and 5 different lr, stopping training.")
            break

        # Log the loss to TensorBoard
        if writer is not None:
            writer.add_scalar("Loss/train", epoch_loss, epoch)
            writer.add_scalar("Learning Rate", optimizer.param_groups[0]['lr'], epoch)

        # Evaluation on validation set
        if val_loader is not None:
            metric_scores = evaluate_segmentation_on_loader(
                model,
                val_loader,
                metrics=all_metrics,
                threshold=0.5,
                writer=writer,
                global_step=epoch,
                device=device
            )
            mean_metric_scores = {metric_name: np.mean(scores) for metric_name, scores in metric_scores.items()}
            
        # Mise à jour du scheduler et vérification de l'évolution de la loss
        if scheduler is not None:
            scheduler.step(epoch_loss)
        torch.cuda.empty_cache()
        
        # save model in checkpoints
        torch.save(model.state_dict(), os.path.join(writer_folder, f"model_last.pth"))
        
        # for each metric in metrics, if there is an improvement, save the model as best_model_lossname_metric_name.pth
        if old_metric_scores is not None:
            for metric_name, score in mean_metric_scores.items():
                if old_metric_scores[metric_name] < score:
                    # Save the model
                    torch.save(model.state_dict(), os.path.join(writer_folder, f"best_model_{metric_name}.pth"))
        else:
            old_metric_scores = mean_metric_scores

    return model

# %%
# cuda - clear cache
torch.cuda.empty_cache()

# %%
# adam optimizer with weight decay for each model
LR = 5e-5
def get_optimizer(model, lr=5e-5):
    return optim.Adam(model.parameters(), lr=lr)

optimizers = []
for model in List_models:
    optimizers.append(get_optimizer(model, lr=LR))

# %%
def cldice(prediction, mask, time=0, mtg=None):
    """
    Custom clDice loss function.
    """
    from RSA_deep_working.Metrics.Losses.clDice.cldice_loss.pytorch.cldice import soft_cldice
    soft_cldice_instance = soft_cldice()
    return soft_cldice_instance(prediction, mask)

def bce(prediction, mask, time=0, mtg=None):
    """
    Binary Cross Entropy loss function.
    """
    return F.binary_cross_entropy_with_logits(prediction, mask)

def dice(prediction, mask, time=0, mtg=None):
    """
    Dice loss function.
    """
    return smp.losses.DiceLoss(mode='binary')(prediction, mask)

def dice_cldice(prediction, mask, time=0, mtg=None):
    """
    Combined Dice and clDice loss function.
    """
    return 0.5 * smp.losses.DiceLoss(mode='binary')(prediction, mask) + 0.5 * cldice(prediction, mask, time, mtg)

def skeleton_recall(prediction, mask, time=0, mtg=None):
    """
    Skeleton Recall loss function.
    """
    from RSA_deep_working.Metrics.Losses.Skeleton_Recall.nnunetv2.training.loss.dice import SoftSkeletonRecallLoss

    soft_skeleton_recall = SoftSkeletonRecallLoss(do_bg=False)
    prediction_2c = torch.cat([1 - prediction, prediction], dim=1)
    mask_2c = torch.cat([1 - mask, mask], dim=1)
    return soft_skeleton_recall(prediction_2c, mask_2c)

# list of loss functions corresponding to the models
loss_functions = [
    #bce,
    #dice,
    cldice,
    dice_cldice,
    skeleton_recall,
    tm.Connectivity_Preserving_Instance_Segmentation,
]

# %%
# one writer for each model
writers = []
writer_names = [
    #"BCE",
    #"Dice",
    "clDice",
    "Dice_clDice",
    "skRecall",
    "superVoxel"
]

for i in range(len(loss_functions)):
    writer = SummaryWriter(f"runs/uc1_segmentation_{writer_names[i]}")
    writers.append(writer)
global_steps = [0] * len(loss_functions)

# %%
def train_and_evaluate(model, loss_function, optimizer, writer, writer_name, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader, metrics=all_metrics, num_epochs=50, scheduler=None, lr_change_threshold=1e-8, device='cpu'):
    # TensorBoard writer
    writer = SummaryWriter(log_dir=f"runs/uc1_segmentation_{writer_name}")
    
    # Training
    model = train_model(
        model,
        train_loader,
        val_loader,
        loss_function,
        optimizer,
        num_epochs=num_epochs,
        device=device,
        writer=writer,
        writer_name=writer_name,
    )

    # Evaluation on validation set
    evaluate_segmentation_on_loader(
        model,
        val_loader,
        metrics=metrics,
        threshold=0.5,
        writer=writer,
        global_step=num_epochs,
        device=device
    )

    # Close the TensorBoard writer
    writer.close()
    # save the model
    torch.save(model.state_dict(), f"Unet_{writer_name}.pth")
    # return the model
    return model

# %%
# for each model, train and evaluate
i = 0
for model, loss_function in zip(List_models, loss_functions):
    # Création du scheduler pour un optimizer avec une patience de 5 époques et une réduction par un facteur 0.5
    
    trained_model = train_and_evaluate(
        model,
        loss_function,
        optimizers[i],
        writers[i],
        writer_name=writer_names[i],
        num_epochs=150,
        scheduler=None,
        lr_change_threshold=1e-8,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader
    )
    # Clear the cache
    torch.cuda.empty_cache()
    # empty memory 
    import gc
    gc.collect()
    i += 1
