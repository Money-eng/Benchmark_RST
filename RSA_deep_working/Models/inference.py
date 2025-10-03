import torch
from pathlib import Path
import tifffile
from Data_loader.tiff_reader import CachedTiffReader
from Models.Model import get_model
import torchvision.transforms as T

# --- CONFIG ---
MODEL_CONFIG = {
    "name": "unet",  # or "segformer"
    "params": {
        "in_channels": 1,
        "out_channels": 1,
        "encoder_name": "efficientnet-b0",
        "encoder_weights": None,
        "return_logits": True,
    },
}
MODEL_WEIGHTS_PATH = "path/to/model_weights.pt"  # <-- update this path
IMAGE_PATH = "path/to/image.tif"  # <-- update this path
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- LOAD MODEL ---
def load_model():
    model = get_model(MODEL_CONFIG)
    model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=DEVICE))
    model.eval()
    model.to(DEVICE)
    return model

# --- LOAD IMAGE ---
def load_tif_image(image_path):
    reader = CachedTiffReader()
    # Load all pages (time series)
    with tifffile.TiffFile(image_path) as tif:
        images = [reader.get_page(image_path, i) for i in range(len(tif.pages))]
    return images

# --- PREPROCESS ---
def preprocess(img):
    # Example: convert to tensor, normalize, add batch/channel dims
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize([0.5], [0.5]),  # adjust if needed
    ])
    img = transform(img)
    img = img.unsqueeze(0)  # batch size 1
    return img

# --- INFERENCE ---
def run_inference():
    model = load_model()
    images = load_tif_image(IMAGE_PATH)
    for i, img in enumerate(images):
        input_tensor = preprocess(img).to(DEVICE)
        with torch.no_grad():
            output = model(input_tensor)
        # Post-process output if needed
        print(f"Frame {i}: output shape {output.shape}")
        # Save or visualize output as needed

if __name__ == "__main__":
    run_inference()
