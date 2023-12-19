import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(os.path.dirname(ROOT_DIR), "models")
DEFAULT_OUTPUT_DIR = os.path.join(ROOT_DIR, "output")
DEFAULT_REF_FACES_DIR = os.path.join(MODELS_DIR, "reference_faces")

DEFAULT_PORT = 8020