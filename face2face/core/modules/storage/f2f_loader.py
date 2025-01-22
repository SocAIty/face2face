import os
import numpy as np
# from insightface.app import FaceAnalysis

from face2face.core.compatibility.Face import Face
from face2face.core.modules.storage.file_writable_face import FileWriteableFace
from face2face.core.modules.utils import get_files_in_dir


def load_reference_face_from_file(face_embedding_file_path: str) -> Face:
    """
    Load a reference face from a file.
    :param face_embedding_file_path: the file path of the reference face
    :return: the reference face
    """
    if not os.path.exists(face_embedding_file_path):
        raise ValueError(f"Reference face {face_embedding_file_path} not found.")

    try:
        # note: potential security issue, if the file was not created with face2face
        embedding = np.load(face_embedding_file_path, allow_pickle=True)
        return FileWriteableFace.to_face(embedding)
    except Exception as e:
        raise f"Error loading reference face {face_embedding_file_path}: {e}"


def load_reference_faces_from_folder(folder_path: str) -> dict:
    """
    Load reference faces from a folder. The folder must contain .npy files with the reference faces.
    The file name will be used as the reference face name.
    :param folder_path: the folder path
    :return:
    """
    files = get_files_in_dir(folder_path, [".npy"])
    reference_faces = {}
    for file in files:
        face_name = os.path.basename(file)[:-4]
        embedding = load_reference_face_from_file(file)
        if embedding is None:
            continue
        reference_faces[face_name] = embedding

    return reference_faces