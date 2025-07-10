import logging
from torch.utils.tensorboard import SummaryWriter


def get_logger(log_file=None) -> logging.Logger:
    """
    Returns a logger instance.
    If log_file is specified, logs are also written to this file.
    """
    logger = logging.getLogger("train_logger")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger


class TensorboardLogger:
    def __init__(self, log_dir: str, flush_secs: int = 30):
        self.writer = SummaryWriter(log_dir=log_dir, flush_secs=flush_secs)

    def log_scalar(self, tag: str, value: float, step: int):
        self.writer.add_scalar(tag, value, step)
        self.writer.flush()

    def log_image(self, tag, img_tensor, global_step=None, walltime=None, dataformats="NCHW"):
        # img_tensor : Tensor CxHxW ou NxCxHxW
        self.writer.add_images(tag, img_tensor, global_step=global_step, walltime=walltime, dataformats=dataformats)
        self.writer.flush()

    def close(self):
        self.writer.flush()
        self.writer.close()


def log_dataset_stats(
        n_series, n_images, n_train_series, n_val_series, n_test_series,
        n_train_images, n_val_images, n_test_images
):
    """ 
    Logs dataset statistics including the number of series and images in training, validation, and testing sets.
    """
    print(f"Total series: {n_series}")
    print(f"Total images: {n_images}")
    print(f"Number of Training series : {n_train_series}")
    print(f"Number of Validation series : {n_val_series}")
    print(f"Number of Testing series : {n_test_series}")
    print(f"Number of Training images : {n_train_images}")
    print(f"Number of Validation images : {n_val_images}")
    print(f"Number of Testing images : {n_test_images}\n")
