import re
import os
import urllib
import unicodedata
import glob


def encode_path_safe(filename: str, allow_unicode=False):
    """
    Makes a string path safe by removing / replacing not by the os allowed patterns.
    This converts:
    spaces 2 dashes, repeated dashes 2 single dashes, remove non alphanumerics, underscores, or hyphen, string 2 lowercase
    strip
    """
    filename = str(filename)
    if allow_unicode:
        filename = unicodedata.normalize('NFKC', filename)
    else:
        filename = (
            unicodedata.normalize('NFKD', filename)
            .encode('ascii', 'ignore')
            .decode('ascii')
        )
    filename = re.sub(r'[^\w\s-]', '', filename.lower())
    return re.sub(r'[-\s]+', '-', filename).strip('-_')


def get_files_in_dir(path: str, extensions: list | str = None) -> list:
    """returns all files in a directory filtered by extension list"""
    if not os.path.isdir(path):
        print(f"{path} is not a directory. Returning empty list")
        return []

    files = []
    if extensions is None:
        # return all files
        files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    else:
        # return files filtered by extensions
        if extensions is str:
            extensions = [extensions]
        for ext in extensions:
            files.extend(glob.glob(os.path.join(path, "*" + ext)))

    return files


def download_file(download_url: str, save_path: str):
    model_dir = os.path.dirname(save_path)
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir, exist_ok=True)

    if not os.path.isfile(save_path):
        print(f'Downloading {download_url}')
        urllib.request.urlretrieve(download_url, save_path)
        print(f'Downloaded {download_url}')
    return save_path
