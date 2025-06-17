import os
import numpy as np
from typing import Union, Dict
from io import BytesIO

from face2face.core.compatibility.Face import Face
from face2face.core.modules.storage.file_writable_face import FileWriteableFace
from face2face.core.modules.utils import get_files_in_dir


def load_reference_face_from_file(face_embedding_file_path: Union[str, BytesIO]) -> Face:
    """
    Load a reference face from a file or BytesIO buffer.
    
    Args:
        face_embedding_file_path: Source of the face embedding. Can be:
            - str: Path to a .npy file containing the face embedding
            - BytesIO: Buffer containing the face embedding data
    
    Returns:
        Face: The loaded face object with its embedding and attributes
    
    Raises:
        ValueError: If the file is not found (for string paths) or cannot be loaded
        Exception: If the face data is invalid or cannot be converted to a Face object
    """
    if isinstance(face_embedding_file_path, str):
        if not os.path.exists(face_embedding_file_path):
            raise ValueError(f"Reference face {face_embedding_file_path} not found.")

    try:
        # note: potential security issue, if the file was not created with face2face
        embedding = np.load(face_embedding_file_path, allow_pickle=True)
        return FileWriteableFace.to_face(embedding)
    except Exception as e:
        raise f"Error loading reference face {face_embedding_file_path}: {e}"


def load_reference_faces_from_folder(folder_path: str) -> Dict[str, Face]:
    """
    Load all reference faces from a folder containing .npy files.
    
    Args:
        folder_path: Path to the folder containing face embedding files (.npy)
    
    Returns:
        Dict[str, Face]: Dictionary mapping face names (derived from filenames) to their Face objects
    
    Note:
        Only loads .npy files from the specified folder. Skips any files that cannot be loaded.
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

