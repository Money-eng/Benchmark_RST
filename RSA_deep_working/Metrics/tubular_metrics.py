# ----------------------------
# Métriques additionnelles pour la segmentation tubulaire
# ----------------------------

# Pour les distances, nous utilisons scipy et skimage
def all_metrics():
    return [cldice, skeleton_recall]


def cldice(prediction, mask, mtg=None):
    from RSA_deep_working.Metrics.Losses.clDice.cldice_loss.pytorch.cldice import soft_cldice
    prediction = prediction.unsqueeze(0)
    mask = mask.unsqueeze(0)
    soft_cldice = soft_cldice()
    return 1 - soft_cldice(prediction, mask).item() # '1 -' to convert from loss to a measure metric 

def skeleton_recall(prediction, mask, mtg=None):
    from RSA_deep_working.Metrics.Losses.SkeletonRecall.nnunetv2.training.loss.dice import SoftSkeletonRecallLoss
    prediction = prediction.unsqueeze(0)
    mask = mask.unsqueeze(0)
    soft_skeleton_recall = SoftSkeletonRecallLoss()
    return soft_skeleton_recall(prediction, mask).item() # '1 -' to convert from loss to a measure metric
