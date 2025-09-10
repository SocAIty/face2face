import re
import os
import zipfile
from typing import Union
import numpy as np
import cv2

import unicodedata
import glob
from face2face.model_definitions import SWAPPER_MODELS, FACE_ENHANCER_MODELS, INSIGHT_FACE_MODELS
from media_toolkit import ImageFile
from media_toolkit.utils.download_helper import download_file


def load_image(img: Union[str, np.array, ImageFile]):
    try:
        image = ImageFile().from_any(img)
        image = image.to_np_array()
        # convert to cv2 BGR image
        # case 4 channels
        if len(image.shape) == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        # case 1 channel
        elif len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    except Exception as e:
        raise ValueError(f"Could not load image {img}. Error: {e}")

    return np.array(image)


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


def extract_zip(zip_path: str, extract_path: str):
    # only extract non existing files
    with zipfile.ZipFile(zip_path, 'r') as zf:
        # Iterate over the files in the zip archive
        for file_name in zf.namelist():
            # Define the full path for the file in the model folder
            extracted_file_path = os.path.join(extract_path, file_name)
            # Only extract if the file does not exist already
            if not os.path.exists(extracted_file_path):
                zf.extract(file_name, extract_path)
    return extract_path


def download_model(model_name: str) -> str:
    """
    Download the models specified in the download urls
    :param model_name: name of the model to download. Look into model_definitions.py for available models
    :return: path to the downloaded model
    """
    # get model config
    model_config = SWAPPER_MODELS.get(model_name, None)
    model_config = model_config or FACE_ENHANCER_MODELS.get(model_name, None)
    model_config = model_config or INSIGHT_FACE_MODELS.get(model_name, None)

    if model_config is None:
        raise ValueError(f"Model {model_name} not found")

    # download model
    download_url = model_config.get('url', None)
    save_path = model_config.get('path', None)

    model_dir = os.path.dirname(save_path)
    os.makedirs(model_dir, exist_ok=True)

    if not os.path.isfile(save_path):
        save_path, _ = download_file(download_url=download_url, save_path=save_path, silent=False)

    if not download_url or not download_url.endswith(".zip"):
        return save_path

    # was provided as .zip file -> extract
    model_folder = save_path.replace(".zip", "")
    os.makedirs(model_folder, exist_ok=True)

    return extract_zip(save_path, model_folder)


