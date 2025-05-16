import tifffile

class CachedTiffReader:
    """
    A class to read and cache pages from TIFF files.

    This class provides a mechanism to load and cache pages from TIFF files
    to avoid redundant file I/O operations. It uses an internal dictionary
    to store the pages of each TIFF file, indexed by the file path.

    Attributes:
        cache (dict): A dictionary to store cached pages of TIFF files. 
                      The keys are file paths, and the values are lists of 
                      numpy arrays representing the pages of the TIFF file.

    Methods:
        get_page(img_path, key):
            Retrieves a specific page from a TIFF file. If the file is not
            already cached, it reads and caches all pages of the file.
    """
    def __init__(self):
        """
        Initializes the CachedTiffReader with an empty cache.
        """
        self.cache = {}

    def get_page(self, img_path, key):
        """
        Retrieves a specific page from a TIFF file.

        If the TIFF file is not already cached, it reads all pages of the file
        and stores them in the cache. Then, it returns the requested page.

        Args:
            img_path (str): The file path to the TIFF file.
            key (int): The index of the page to retrieve.

        Returns:
            numpy.ndarray: The requested page as a numpy array.

        Raises:
            IndexError: If the specified page index is out of range.
        """
        if img_path not in self.cache:
            with tifffile.TiffFile(img_path) as tif:
                self.cache[img_path] = [page.asarray() for page in tif.pages]
        return self.cache[img_path][key]