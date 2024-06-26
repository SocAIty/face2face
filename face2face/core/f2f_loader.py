from typing import Union, List
import os
import numpy as np
import insightface
from insightface.app.common import Face

from face2face.core.file_writable_face import FileWriteableFace
from face2face.utils import get_files_in_dir


def get_face_analyser(model_path: str, providers, det_size=(320, 320)):
    """Get the face analyser model. The face analyser detects faces and extracts facial features."""
    # load default face analyser if model_path is None
    face_analyser = insightface.app.FaceAnalysis(
        name="buffalo_l", root=f"{model_path}./checkpoints", providers=providers
    )
    face_analyser.prepare(ctx_id=0, det_size=det_size)
    # TODO: load face analyser from file if model_path is not None

    return face_analyser


def load_reference_face_from_file(face_embedding_file_path: str) -> Union[List[Face], None]:
    """
    Load a reference face from a file.
    :param face_embedding_file_path: the file path of the reference face
    :return: the reference face
    """
    if not os.path.exists(face_embedding_file_path):
        print(f"Reference face {face_embedding_file_path} not found.")
        return None

    try:
        # note: potential security issue, if the file was not created with face2face
        embedding = np.load(face_embedding_file_path, allow_pickle=True)
        if len(embedding) > 0:
            embedding = [FileWriteableFace.to_face(face) for face in embedding]

        return embedding
    except Exception as e:
        print(f"Error loading reference face {face_embedding_file_path}: {e}")


def load_reference_faces_from_folder(folder_path: str) -> dict:
    """
    Load reference faces from a folder. The folder must contain .npz files with the reference faces.
    The file name will be used as the reference face name.
    :param folder_path: the folder path
    :return:
    """
    files = get_files_in_dir(folder_path, [".npz"])
    reference_faces = {}
    for file in files:
        face_name = os.path.basename(file)[:-4]
        embedding = load_reference_face_from_file(file)
        if embedding is None:
            continue
        reference_faces[face_name] = embedding

    return reference_faces