import tifffile


class TiffReader:

    def __init__(self):
        """
        A simple class to read TIFF files using tifffile.
        """
        pass
        

    def get_page(self, img_path, key):
        with tifffile.TiffFile(img_path) as tif:
            return tif.pages[key].asarray()
    
    def get_series(self, img_path):
        with tifffile.TiffFile(img_path) as tif:
            return tif.asarray()
