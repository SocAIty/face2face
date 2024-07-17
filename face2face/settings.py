import os

ROOT_DIR = os.getenv("ROOT_DIR", os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT_DIR, "models")
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")
REF_FACES_DIR = os.getenv("REF_FACES_DIR", os.path.join(ROOT_DIR, "face_embeddings"))

PORT = 8020
PROVIDER = os.getenv("PROVIDER", "fastapi")

# ONNX Runtime Settings. Note that this settings is only recognized on startup. Change the variable of f2f instance.
# Important in case of openvino or when using multiple gpus
DEVICE_ID = os.getenv("DEVICE_ID", 0)
EXECUTION_PROVIDER = os.getenv("EXECUTION_PROVIDERS", "CUDAExecutionProvider")

