import math

import torch
from openalea.rsml import rsml2mtg

from .mtg_operations import extract_mtg_at_time_t


def _segment_mask(
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        H: int,
        W: int,
        radius: float,
        device: torch.device,
) -> torch.Tensor:
    """
    Retourne un masque [1, H, W] où les pixels à distance <= radius du segment
    (x1,y1)-(x2,y2) sont à 1. Limité à la bbox du segment élargie pour efficacité.
    """
    # Définir bbox élargie avec math.floor/ceil sur scalaires
    min_x = max(int(math.floor(min(x1, x2) - radius)), 0)
    max_x = min(int(math.ceil(max(x1, x2) + radius)), W)
    min_y = max(int(math.floor(min(y1, y2) - radius)), 0)
    max_y = min(int(math.ceil(max(y1, y2) + radius)), H)

    if min_x >= max_x or min_y >= max_y:
        return torch.zeros((1, H, W), device=device)

    # Grille locale (float tensors)
    ys = torch.arange(min_y, max_y, device=device, dtype=torch.float32)
    xs = torch.arange(min_x, max_x, device=device, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")  # shape (h, w)

    # Vecteurs du segment
    px = grid_x - x1
    py = grid_y - y1
    vx = x2 - x1
    vy = y2 - y1
    seg_len2 = vx * vx + vy * vy  # scalaire

    if seg_len2 == 0.0:
        # segment dégénéré : balle autour de (x1,y1)
        dist2 = (grid_x - x1) ** 2 + (grid_y - y1) ** 2
    else:
        t = (px * vx + py * vy) / seg_len2
        t_clamped = torch.clamp(t, 0.0, 1.0)
        proj_x = x1 + t_clamped * vx
        proj_y = y1 + t_clamped * vy
        dist2 = (grid_x - proj_x) ** 2 + (grid_y - proj_y) ** 2

    mask_local = (dist2 <= (radius ** 2)).float()  # (h, w)

    # Insérer dans masque global
    mask = torch.zeros((1, H, W), device=device)
    mask[:, min_y:max_y, min_x:max_x] = mask_local.unsqueeze(0)
    return mask


def roi_fnc(imgs: torch.Tensor, time: list, mtgs: list, diameter: float = 10.0) -> torch.Tensor:
    device = imgs.device
    B, _, H, W = imgs.shape
    roi_masks = torch.zeros((B, 1, H, W), device=device)

    radius = diameter / 2.0  # rayon en pixels

    for i in range(B):
        mtg = rsml2mtg(mtgs[i])
        t = time[i]
        mtg_at_time = extract_mtg_at_time_t(mtg, t)
        geometry = mtg_at_time.property("geometry")  # {root_id: [[x1,y1], [x2,y2], ...]}

        accum_mask = torch.zeros((1, H, W), device=device)
        for _, coords in geometry.items():
            if len(coords) < 1:
                continue
            # Noeuds : segment dégénéré autour de chaque coordonnée
            for j in range(len(coords)):
                x, y = coords[j]
                node_mask = _segment_mask(x, y, x, y, H, W, radius, device)
                accum_mask = torch.maximum(accum_mask, node_mask)

            # Tubes entre points consécutifs
            for j in range(len(coords) - 1):
                x1, y1 = coords[j]
                x2, y2 = coords[j + 1]
                seg_mask = _segment_mask(x1, y1, x2, y2, H, W, radius, device)
                accum_mask = torch.maximum(accum_mask, seg_mask)

        roi_masks[i] = (accum_mask > 0).float()

    # import matplotlib.pyplot as plt
    # Visualisation des masques ROI
    # num_imgs = roi_masks.shape[0]
    # fig, axs = plt.subplots(1, num_imgs, figsize=(15, 5))
    # for i in range(num_imgs):
    #   axs[i].imshow(roi_masks[i, 0].cpu().numpy(), cmap='gray')
    #  axs[i].set_title(f'ROI Mask {i+1}')
    # axs[i].axis('off')
    # plt.tight_layout()
    # plt.show()

    return roi_masks


if __name__ == "__main__":

    # Example usage
    imgs = torch.randn(6, 3, 1166, 1366)  # Example image tensor
    time = [5, 10, 15, 20, 25, 29]  # Example time list
    from openalea.rsml import rsml2mtg

    mtg1 = "/home/loai/Documents/code/RSMLExtraction/Results/Reconstruction_0.55/Segformer_bce/Test/230629PN024/61_prediction_before_expertized_graph.rsml"
    mtgs = [mtg1, mtg1, mtg1, mtg1, mtg1, mtg1]  # Example list of MTG objects

    roi_masks = roi_fnc(imgs, time, mtgs)
    print(roi_masks.shape)  # Should print (6, 1, 1166, 1366)

    # plot all roi_masks in subplots
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    axs = axs.flatten()
    for i in range(6):
        axs[i].imshow(roi_masks[i, 0].cpu().numpy(), cmap='gray')
        axs[i].set_title(f'ROI Mask {i + 1}')
        axs[i].axis('off')
    plt.tight_layout()
    plt.show()
