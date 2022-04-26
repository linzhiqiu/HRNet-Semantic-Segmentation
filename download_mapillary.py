from tqdm import tqdm
import zipfile
import gdown
import os

# helper function to download and extract zip file from Google drive
def _extract_zip(from_path, to_path, compression):
    with zipfile.ZipFile(
        from_path, "r", compression=zipfile.ZIP_STORED
    ) as zip:
        zip.extractall(to_path)

def extract_archive(from_path, to_path=None, remove_finished=False):
    """Extract an archive.

    The archive type and a possible compression is automatically detected from the file name. If the file is compressed
    but not an archive the call is dispatched to :func:`decompress`.

    Args:
        from_path (str): Path to the file to be extracted.
        to_path (str): Path to the directory the file will be extracted to. If omitted, the directory of the file is
            used.
        remove_finished (bool): If ``True``, remove the file after the extraction.

    Returns:
        (str): Path to the directory the file was extracted to.
    """
    if to_path is None:
        to_path = os.path.dirname(from_path)

    _extract_zip(from_path, to_path, None)
    if remove_finished:
        os.remove(from_path)

    return to_path
