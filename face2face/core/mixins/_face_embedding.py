# avoid circular dependency but provide type hints
from __future__ import annotations
from typing import TYPE_CHECKING, Union, List, Tuple, Dict

from face2face.core.compatibility.Face import Face
from media_toolkit import ImageFile, MediaFile, MediaList

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

    def load_faces(
        self, 
        face_names: Union[str, List[str], List[Face], MediaFile, MediaList, None] = None
    ) -> Dict[str, Face]:
        """
        Load face embeddings from various sources.
        
        Args:
            face_names: The source of face embeddings to load. Can be:
                - None: Loads all stored face embeddings from the _face_embeddings folder
                - str: Loads a single face embedding by name
                - List[str]: Loads multiple face embeddings by their names
                - List[Face]: Returns the faces as {index: face}
                - MediaFile: Loads a face from the file content (e.g., from API response)
                - MediaList: Loads faces from each file in the list, supporting both MediaFile and string paths
        
        Returns:
            Dict[str, Face]: Dictionary mapping face names to their embeddings.
                For MediaFile inputs, uses the file name as the key.
                For MediaList inputs, uses file names or string paths as keys.
                For List[Face] inputs, uses numeric indices as keys.
        
        Raises:
            ValueError: If a face file is not found or cannot be loaded
        """
        if face_names is None:
            return self.load_all_faces()
        elif isinstance(face_names, str):
            return {face_names: self.load_face(face_names)}
        elif isinstance(face_names, MediaFile):
            # Load face from MediaFile content
            face = load_reference_face_from_file(face_names.to_bytes_io())
            return {face_names.file_name: face}
        elif isinstance(face_names, MediaList):
            # Load faces from each file in MediaList
            ret = {}
            for i, file in enumerate(face_names):
                if isinstance(file, MediaFile):
                    face = load_reference_face_from_file(file.to_bytes_io())
                    ret[file.file_name] = face
                elif isinstance(file, str):
                    ret[file] = self.load_face(file)
            return ret

        # convert whatever list to dict
        ret = {}
        for i, face in enumerate(face_names):
            if isinstance(face, Face):
                ret[i] = face
            elif isinstance(face, str):
                ret[face] = self.load_face(face)

        return ret

    def load_all_faces(self: Face2Face) -> Dict[str, Face]:
        """
        Load all face embeddings from the _face_embeddings folder.
        
        Returns:
            Dict[str, Face]: Dictionary mapping face names to their embeddings
        """
        for face_name in glob.glob(self._face_embedding_folder + "/*.npy"):
            self.load_face(face_name)
        return self._face_embeddings

    def add_face(
        self: Face2Face,
        face_name: Union[str, List[str]],
        image: Union[np.array, str, ImageFile],
        save: bool = False
    ) -> Union[Tuple[str, FileWriteableFace], Dict[str, Face]]:
        """
        Add one or multiple reference face(s) to the face swapper. This face(s) can be used for swapping in other images.

        Args:
            face_name: The name(s) for the reference face(s).
                - If a single string, creates one face embedding
                - If a list of strings, creates embeddings for each face from left to right in the image
            image: The image from which to extract the face(s). Can be:
                - numpy array: Direct image data
                - str: Path to image file
                - ImageFile: MediaFile containing the image
            save: If True, saves the face embeddings to disk in the _face_embeddings folder.
                If False, only stores them in memory.

        Returns:
            Union[Tuple[str, FileWriteableFace], Dict[str, Face]]:
                - For single face: Tuple of (face_name, FileWriteableFace)
                - For multiple faces: Dict mapping face names to their embeddings

        Raises:
            ValueError: If no face name is provided or no faces are detected in the image
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
