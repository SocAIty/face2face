import os
import urllib

from .settings import MODELS_DIR


def download_file(download_url: str, save_path: str):
    model_dir = os.path.dirname(save_path)
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir, exist_ok=True)

    if not os.path.isfile(save_path):
        print(f'Downloading {download_url}')
        urllib.request.urlretrieve(download_url, save_path)
        print(f'Downloaded {download_url}')
    return save_path


def download_face_swap_model():
    url = "https://github.com/facefusion/facefusion-assets/releases/download/models/inswapper_128.onnx"
    download_file(url, os.path.join(MODELS_DIR, "insightface", "inswapper_128.onnx"))


if __name__ == "__main__":
    download_face_swap_model()
