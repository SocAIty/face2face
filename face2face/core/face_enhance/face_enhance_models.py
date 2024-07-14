from typing import Literal
import numpy as np
from face2face.settings import MODELS_DIR

# supported face enhancer models
FACE_ENHANCER_MODELS = {
    'gfpgan_1.4': {
            'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models/gfpgan_1.4.onnx',
            'path': f'{MODELS_DIR}/gfpgan_1.4.onnx',
            'template': 'ffhq_512',
            'size': (512, 512)
    },
    'gpen_bfr_256':{
            'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models/gpen_bfr_256.onnx',
            'path': f'{MODELS_DIR}/gpen_bfr_256.onnx',
            'template': 'arcface_128_v2',
            'size': (256, 256)
    },
    'gpen_bfr_512': {
            'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models/gpen_bfr_512.onnx',
            'path': f'{MODELS_DIR}/gpen_bfr_512.onnx',
            'template': 'ffhq_512',
            'size': (512, 512)
    },
    'gpen_bfr_1024': {
            'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models/gpen_bfr_1024.onnx',
            'path': f'{MODELS_DIR}/gpen_bfr_1024.onnx',
            'template': 'ffhq_512',
            'size': (1024, 1024)
    },
    'gpen_bfr_2048': {
            'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models/gpen_bfr_2048.onnx',
            'path': f'{MODELS_DIR}/gpen_bfr_2048.onnx',
            'template': 'ffhq_512',
            'size': (2048, 2048)
    }
}

# supported warp templates
WarpTemplate = Literal['arcface_112_v1', 'arcface_112_v2', 'arcface_128_v2', 'ffhq_512']
WARP_TEMPLATES = {
    'arcface_112_v1': np.array(
        [
            [0.35473214, 0.45658929],
            [0.64526786, 0.45658929],
            [0.50000000, 0.61154464],
            [0.37913393, 0.77687500],
            [0.62086607, 0.77687500]
        ]),
    'arcface_112_v2': np.array(
        [
            [0.34191607, 0.46157411],
            [0.65653393, 0.45983393],
            [0.50022500, 0.64050536],
            [0.37097589, 0.82469196],
            [0.63151696, 0.82325089]
        ]),
    'arcface_128_v2': np.array(
        [
            [0.36167656, 0.40387734],
            [0.63696719, 0.40235469],
            [0.50019687, 0.56044219],
            [0.38710391, 0.72160547],
            [0.61507734, 0.72034453]
        ]),
    'ffhq_512': np.array(
        [
            [0.37691676, 0.46864664],
            [0.62285697, 0.46912813],
            [0.50123859, 0.61331904],
            [0.39308822, 0.72541100],
            [0.61150205, 0.72490465]
        ])
}


def get_model_config(key: str):
    key = key.lower()
    model_config = FACE_ENHANCER_MODELS.get(key, None)
    if model_config is None:
        raise ValueError(f"Model {key} not found")
    return model_config


OCCLUSION_MODELS = {
    'face_occluder': {
        'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models/face_occluder.onnx',
        'path': f'{MODELS_DIR}/face_occluder.onnx'
    },
    'face_parser': {
        'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models/face_parser.onnx',
        'path': f'{MODELS_DIR}/face_parser.onnx'
    }
}

FACE_MASK_REGIONS = {
    'skin': 1,
    'left-eyebrow': 2,
    'right-eyebrow': 3,
    'left-eye': 4,
    'right-eye': 5,
    'glasses': 6,
    'nose': 10,
    'mouth': 11,
    'upper-lip': 12,
    'lower-lip': 13
}
