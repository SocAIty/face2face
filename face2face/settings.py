import os

ROOT_DIR = os.getenv("ROOT_DIR", os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT_DIR, "models")
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")
REF_FACES_DIR = os.getenv("REF_FACES_DIR", os.path.join(ROOT_DIR, "face_embeddings"))

PORT = 8020
PROVIDER = os.getenv("PROVIDER", "fastapi")

# ONNX Runtime Settings.
# Important in case of openvino or when using multiple gpus
DEVICE_ID = os.getenv("DEVICE_ID", 0)
EXECUTION_PROVIDER = os.getenv("EXECUTION_PROVIDERS", "CUDAExecutionProvider")


def set_device(device_id: int):
    # setting will apply it to the face enhancers
    global DEVICE_ID
    DEVICE_ID = device_id
    # for inswapper is needed to do it like this, because uses model zoo, and there's provider option not set
    #os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
