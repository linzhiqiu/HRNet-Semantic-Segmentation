from tqdm import tqdm
import zipfile
import gdown
import os
import argparse

MAPILLARY_GDRIVE_ID = "1MR7M-as-stOIog2EyhNXwnKKC9-aXXiM"
MAPILLARY_MD5 = "a821a7620adfa4a73ca2ec00c16fd054"

def parse_args():
    parser = argparse.ArgumentParser(description='Download mapillary vistas v1.2 and v2.0')
    
    parser.add_argument('--save_dir',
                        help='save to this directory',
                        required=True,
                        type=str)
    parser.add_argument('--gdrive_id', type=str, default=MAPILLARY_GDRIVE_ID)
    parser.add_argument("--md5", type=str, default=MAPILLARY_MD5) 
    parser.add_argument("--remove_finished", type=bool, default=True)       
    args = parser.parse_args()
    return args

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


def download_from_google_drive(save_dir,
                               name="mapillary_vistas",
                               gdrive_id=MAPILLARY_GDRIVE_ID,
                               md5=MAPILLARY_MD5,
                               remove_finished=True):
    dataset_dir = os.path.join(save_dir, name)
    if not os.path.exists(dataset_dir):
        zip_path = os.path.join(save_dir, f'{name}.zip')
        # gdrive_url = f"https://drive.google.com/u/0/uc?id={gdrive_id}"
        gdrive_url = f"https://drive.google.com/u/0/uc?export=download&confirm=pbef&id={gdrive_id}"
        gdown.download(gdrive_url, zip_path, quiet=False)
        gdown.cached_download(gdrive_url, zip_path,
                              md5=md5)
        extract_archive(
            zip_path, to_path=save_dir, remove_finished=remove_finished
        )
    else:
        print(f"{dataset_dir} already exists.")
        
    
if __name__ == "__main__":
    args = parse_args()
    download_from_google_drive(
        args.save_dir,
        gdrive_id=args.gdrive_id,
        md5=args.md5,
        remove_finished=args.remove_finished
    )
    