from tifffile import imread

from utils.misc import set_seed, SEED

set_seed(SEED)


class TiffReader:

    def __init__(self):
        """
        A simple class to read TIFF files using tifffile.
        """
        pass

    def get_page(self, img_path, key):
        return imread(img_path, key=key)

    def get_series(self, img_path):
        return imread(img_path)
