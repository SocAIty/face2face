import os

ROOT_DIR = os.getenv("ROOT_DIR", os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT_DIR, "models")
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")
REF_FACES_DIR = os.getenv("REF_FACES_DIR", os.path.join(ROOT_DIR, "face_embeddings"))

PORT = 8020
