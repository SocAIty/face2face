import numpy as np

import os
from io import BytesIO

from face2face.core.compatibility.Face import Face


class FileWriteableFace(dict):
    """
    Enables the insightface Face class to be saved.
    To recast the object simply use new_face = Face(pickleSafeFace)
    The problem was that Face overwrites __getattr__ which all the pickle methods use.
    In addition, some np types are not pickleable.
    """

    def __init__(self, face: Face):
        super().__init__()
        # copy all attributes
        for attr in dir(face):
            if not attr.startswith("__"):
                try:
                    self[attr] = face[attr]
                except Exception:
                    try:
                        if type(face.attr) in [
                            np.float32,
                            np.float64,
                            np.int32,
                            np.int64,
                        ]:
                            self.attr = (float)(face[attr])
                        else:
                            self.attr = face.attr
                    except Exception:
                        pass

    @staticmethod
    def to_face(pickle_safe_face):
        if type(pickle_safe_face) in [np.array, np.ndarray, list]:
            if len(pickle_safe_face) > 1:
                print("Warning: to_face only works with one face. Returning first face.")
            elif len(pickle_safe_face) == 0:
                raise ValueError("Warning: to_face only works with one face. Provided empty list.")
            pickle_safe_face = pickle_safe_face[0]

        if isinstance(pickle_safe_face, Face):
            return pickle_safe_face
        f = Face(pickle_safe_face)
        return f

    def to_file(self, file_path: str):
        """
        Save the face to a file.
        """
        # if file path is folder: add a default name
        if os.path.isdir(file_path):
            file_path = os.path.join(file_path, "face.npy")

        if not file_path.endswith(".npy"):
            file_path += ".npy"

        # if file path doesn't exist, create it
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # save to file
        np.save(file_path, arr=[self], allow_pickle=True)

        return file_path

    def to_bytes_io(self):
        """
        Save the face to a BytesIO object.
        """
        bytes_io = BytesIO()
        np.save(bytes_io, arr=[self], allow_pickle=True)
        bytes_io.seek(0)
        return bytes_io
