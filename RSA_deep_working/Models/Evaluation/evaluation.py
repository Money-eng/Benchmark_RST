import os
import numpy as np
import torch
from tqdm import tqdm
import torchvision.transforms.functional as TF
from PIL import Image
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed

def tensor_to_heatmap_image(tensor, cmap='hot'):
    """
    Convert a PyTorch tensor to a heatmap image that will be saved in TensorBoard.
    Args:
        tensor (torch.Tensor): The input tensor to convert.
        cmap (str): The colormap to use for the heatmap. Default is 'hot'.
    Returns:
        np.ndarray: The heatmap image as a numpy array.
    """
    tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
    cpu_tensor = tensor.cpu()
    image = Image.fromarray(
        (cpu_tensor.numpy() * 255).astype(np.uint8), mode='L')
    image_np = np.array(image)
    heatmap = plt.get_cmap(cmap)(image_np)[:, :, :3]
    heatmap = (heatmap * 255).astype(np.uint8)
    return heatmap

def evaluate_segmentation_on_loader(model: torch.nn.Module,
                                    loader: torch.utils.data.DataLoader,
                                    metrics: dict,
                                    threshold: float = 0.5,
                                    writer: torch.utils.tensorboard.SummaryWriter = None,
                                    global_step: int = None,
                                    device: torch.device = torch.device(
                                        "cuda" if torch.cuda.is_available() else "cpu")
                                    ):
    model.eval()
    #  dict : {cpu: {metric1: score1, metric2: score2}, gpu: {metric3: score3, metric4: score4}}
    metric_scores = {'cpu': {}, 'gpu': {}}
    metric_scores_list = {'cpu': {}, 'gpu': {}}
    for metric in metrics['cpu']:
        metric_scores_list['cpu'][metric] = []
    sample_batch = None

    max_workers = os.cpu_count() or 4
    cpu_futures = []  # [(future, metric), ...]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        with torch.no_grad():
            for batch in tqdm(
                loader,
                desc="Evaluation iteration",
                postfix=f"Epoch: {global_step}" if global_step is not None else "",
                bar_format="{l_bar}{bar}{r_bar}",
                unit="batch",
                total=len(loader),
                position=0,
                leave=True,
                dynamic_ncols=True
            ):
                images, masks, time, mtgs = batch
                images, masks = images.to(device), masks.to(device)
                output = model(images)
                prediction = (torch.sigmoid(output)).float() # Threshold ? 
                preds_cpu = prediction.cpu().numpy()
                masks_cpu = masks.cpu().numpy()

                # for each metric, compute the score (cpu parrallel then gpu parrallel)
                cpu_metrics = metrics['cpu']  # list of metrics
                gpu_metrics = metrics['gpu']
                
                for i in range(images.shape[0]):
                    args = (preds_cpu[i], masks_cpu[i], time[i], mtgs[i])
                    for metric in cpu_metrics:
                        # on soumet l'appel à executor
                        fut = executor.submit(metric, *args)
                        cpu_futures.append((fut, metric))

                    # GPU metrics
                    for metric in gpu_metrics:
                        if metric not in metric_scores_list['gpu']:
                            metric_scores_list['gpu'][metric] = []
                        score = metric(
                            prediction[i], masks[i][i], time[i], mtgs[i])
                        metric_scores_list['gpu'][metric].append(score)

                if sample_batch is None and writer is not None and global_step is not None:
                    sample_batch = (images.cpu(), masks.cpu(),
                                    output.cpu(), prediction.cpu())
                    
        # une fois toutes les images traitées, on attend les résultats CPU
        for fut, metric in cpu_futures:
            score = fut.result()
            metric_scores_list['cpu'][metric].append(score)

    # metrics scores will be the mean of the scores for each metric
    #  CPU metrics
    for metric_name, scores in metric_scores_list['cpu'].items():
        mean_score = np.mean(scores)
        var_score = np.var(scores)
        metric_scores['cpu'][metric_name] = mean_score
        if writer is not None and global_step is not None:
            writer.add_scalar(
                f"Eval/Mean/{metric_name}_mean", mean_score, global_step)
            writer.add_scalar(
                f"Eval/Var/{metric_name}_var", var_score, global_step)
    # GPU metrics
    for metric_name, scores in metric_scores_list['gpu'].items():
        mean_score = np.mean(scores)
        var_score = np.var(scores)
        metric_scores['gpu'][metric_name] = mean_score
        if writer is not None and global_step is not None:
            writer.add_scalar(
                f"Eval/Mean/{metric_name}_mean", mean_score, global_step)
            writer.add_scalar(
                f"Eval/Var/{metric_name}_var", var_score, global_step)

    #  save sample images
    if writer is not None and global_step is not None:
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
                out = outputs[i].squeeze()
                sig_out = torch.sigmoid(outputs[i]).squeeze()

                out_img = tensor_to_heatmap_image(out, cmap='hot')
                sig_img = tensor_to_heatmap_image(sig_out, cmap='hot')

                out_tensor = TF.to_tensor(out_img)
                sig_tensor = TF.to_tensor(sig_img)
                outputs_heatmaps.append(out_tensor)
                sigmoid_heatmaps.append(sig_tensor)

            outputs_heatmaps_tensor = torch.stack(outputs_heatmaps)
            sigmoid_heatmaps_tensor = torch.stack(sigmoid_heatmaps)

            images_concat = torch.cat([images[i]
                                      for i in range(n_samples)], dim=2)
            masks_concat = torch.cat([masks[i]
                                     for i in range(n_samples)], dim=2)
            predictions_concat = torch.cat(
                [predictions[i] for i in range(n_samples)], dim=2)
            outputs_heatmaps_concat = torch.cat(
                [outputs_heatmaps_tensor[i] for i in range(n_samples)], dim=2)
            sigmoid_heatmaps_concat = torch.cat(
                [sigmoid_heatmaps_tensor[i] for i in range(n_samples)], dim=2)

            writer.add_image("Sample/Images", images_concat, global_step)
            writer.add_image("Sample/Masks", masks_concat, global_step)
            writer.add_image("Sample/Predictions",
                             predictions_concat, global_step)
            writer.add_image("Sample/Heatmaps",
                             outputs_heatmaps_concat, global_step)
            writer.add_image("Sample/Sigmoid_Heatmaps",
                             sigmoid_heatmaps_concat, global_step)

    return metric_scores
