from face2face.settings import MODELS_DIR


SWAPPER_MODELS = {
    "inswapper_128": {
        'url': "https://github.com/facefusion/facefusion-assets/releases/download/models/inswapper_128.onnx",
        'path': f'{MODELS_DIR}/insightface/inswapper_128.onnx',
    }
}

FACE_ENHANCER_MODELS = {
    'gfpgan_1.4': {
            'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models/gfpgan_1.4.onnx',
            'path': f'{MODELS_DIR}/face_enhancer/gfpgan_1.4.onnx',
            'template': 'ffhq_512',
            'size': (512, 512)
    },
    'gpen_bfr_256':{
            'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models/gpen_bfr_256.onnx',
            'path': f'{MODELS_DIR}/face_enhancer/gpen_bfr_256.onnx',
            'template': 'arcface_128_v2',
            'size': (256, 256)
    },
    'gpen_bfr_512': {
            'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models/gpen_bfr_512.onnx',
            'path': f'{MODELS_DIR}/face_enhancer/gpen_bfr_512.onnx',
            'template': 'ffhq_512',
            'size': (512, 512)
    },
    'gpen_bfr_1024': {
            'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models/gpen_bfr_1024.onnx',
            'path': f'{MODELS_DIR}/face_enhancer/gpen_bfr_1024.onnx',
            'template': 'ffhq_512',
            'size': (1024, 1024)
    },
    'gpen_bfr_2048': {
            'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models/gpen_bfr_2048.onnx',
            'path': f'{MODELS_DIR}/face_enhancer/gpen_bfr_2048.onnx',
            'template': 'ffhq_512',
            'size': (2048, 2048)
    }
}

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
