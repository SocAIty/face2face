# avoid circular dependency but provide type hints
from __future__ import annotations
from typing import TYPE_CHECKING, Union, List, Tuple

from face2face.core.compatibility.Face import Face
from media_toolkit import ImageFile

if TYPE_CHECKING:
    from face2face.core.face2face import Face2Face

# regular imports
import glob
import os

import numpy as np


from face2face.core.modules.storage.f2f_loader import load_reference_face_from_file
from face2face.core.modules.storage.file_writable_face import FileWriteableFace
from face2face.settings import EMBEDDINGS_DIR
from face2face.core.modules.utils.utils import encode_path_safe
from face2face.core.modules.utils.utils import load_image


class _FaceEmbedding:
    def load_face(self: Face2Face, face_name: str) -> Face:
        """
        Load a reference face embedding from a file.
        :param face_name: the name of the reference face embedding
        :return: the embedding of the reference face(s)
        """
        # check if is already in ram. If yes return that one
        embedding = self._face_embeddings.get(face_name, None)
        if embedding is not None:
            return embedding

        # load from file
        file = os.path.join(self._face_embedding_folder, f"{face_name}.npy")
        face = load_reference_face_from_file(file)

        # add to memory dict
        self._face_embeddings[face_name] = face
        return face

    def load_faces(self, face_names: Union[str, List[str], List[Face], None] = None) -> dict:
        """
        :param face_names: the faces to load from the _face_embeddings folder.
            If None all stored face_embeddings are loaded and returned.
            If list of strings, the faces with the names in the list are loaded.
            If list of Face objects, the faces are returned as { index: face }.
        :return: the loaded faces as dict {face_name: face_embedding}.
        """
        if face_names is None:
            return self.load_all_faces()
        elif isinstance(face_names, str):
            return {face_names: self.load_face(face_names)}

        # convert whatever list to dict
        ret = {}
        for i, face in enumerate(face_names):
            if isinstance(face, Face):
                ret[i] = face
            elif isinstance(face, str):
                ret[face] = self.load_face(face)

        return ret

    def load_all_faces(self: Face2Face):
        """
        Load all face embeddings from the _face_embeddings folder.
        """
        for face_name in glob.glob(self._face_embedding_folder + "/*.npy"):
            self.load_face(face_name)
        return self._face_embeddings

    def add_face(
        self: Face2Face,
        face_name: Union[str, List[str]],
        image: Union[np.array, str, ImageFile],
        save: bool = False
    ) -> Union[Tuple[str, FileWriteableFace], dict]:
        """
        Add one or multiple reference face(s) to the face swapper. This face(s) can be used for swapping in other images.

        :param face_name: The name for the reference face.
            In case you provide a list of face names, an embedding is stored for each face from left to right in the provided image.

        :param image: The image from which to extract the face(s) (can be a numpy array, file path, or ImageFile).
            If there are multiple faces in the image, an embedding will be created for each name from left to right.
            If you only provide one name, only the first face will be stored.
        :param save:
            If True, the reference face will be saved to the _face_embeddings folder for future use.
            If False, the reference face will only be stored in memory.
        :return: A tuple containing the safely encoded face name and the reference face.
            In case of multiple faces returns as dict {face_name: face_embedding}
        """


        if isinstance(face_name, list) and len(face_name) == 0 or face_name is None:
            raise ValueError("Please provide at least one face name.")

        try:
            image = load_image(image)
            detected_faces = self.detect_faces(image)
        except Exception as e:
            raise ValueError(f"Could not load image or face detection failed. Error: {e}")

        # Deal with errors of too much or too few faces in the reference image
        if not detected_faces:
            raise ValueError(f"No faces detected in the provided image for {face_name}.")

        if not isinstance(face_name, list):
            face_name = [face_name]

        if len(face_name) > len(detected_faces):
            print(f"Not enough faces in the image for all provided face names. "
                  f"Only {len(detected_faces)} faces found. Skipping the rest")
            face_name = face_name[:len(detected_faces)]

        face = None
        for i, name in enumerate(face_name):
            name = encode_path_safe(name)
            face = detected_faces[i]
            # Store the detected faces in memory
            self._face_embeddings[name] = face

            # store the detected faces on disc
            face = FileWriteableFace(face)
            # Save face to virtual file
            if save:
                os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
                filename = os.path.join(EMBEDDINGS_DIR, f"{name}.npy")
                if os.path.isfile(filename):
                    print(f"Reference face {name} already exists. Overwriting.")
                face.to_file(filename)
        if len(face_name) > 1:
            # Return a dict with the face names and embeddings
            return {
                name: self._face_embeddings[name]
                for name in face_name
            }
        else:
            return face_name[0], face
