# avoid circular dependency but provide type hints
from __future__ import annotations
from typing import TYPE_CHECKING, Union, List, Tuple

from media_toolkit import ImageFile

if TYPE_CHECKING:
    from face2face.core.face2face import Face2Face

# regular imports
import glob
import os
from io import BytesIO

import numpy as np
from insightface.app.common import Face

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
        face_name: str,
        image: Union[np.array, str, ImageFile],
        save: bool = False
    ) -> Tuple[str, FileWriteableFace]:
        """
        Add a reference face to the face swapper. This face will be used for swapping in other images.

        :param face_name: The name for the reference face
        :param image: The image from which to extract face (can be a numpy array, file path, or ImageFile).
        :param save:
            If True, the reference face will be saved to the _face_embeddings folder for future use.
            If False, the reference face will only be stored in memory.
        :return: A tuple containing the safely encoded face name and the reference face.
        :raises ValueError: If | detected faces | != 1
        """
        try:
            image = load_image(image)

            detected_faces = self.detect_faces(image)

            # Deal with errors of too much or too few faces in the reference image
            if not detected_faces:
                raise ValueError(f"No faces detected in the provided image for {face_name}.")

            if len(detected_faces) > 1:
                raise ValueError(f"Multiple faces detected in the provided image for {face_name}.")

            face_name = encode_path_safe(face_name)
            face = detected_faces[0]
            # Store the detected faces in memory
            self._face_embeddings[face_name] = face

            # store the detected faces on disc
            face = FileWriteableFace(face)

            # Save face to virtual file
            if save:
                os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
                filename = os.path.join(EMBEDDINGS_DIR, f"{face_name}.npy")
                if os.path.isfile(filename):
                    print(f"Reference face {face_name} already exists. Overwriting.")

                face.to_file(filename)

            return face_name, face
        except Exception as e:
            print(f"Error while adding face: {e}")
            raise
