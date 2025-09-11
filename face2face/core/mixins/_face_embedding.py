# avoid circular dependency but provide type hints
from __future__ import annotations
import uuid
from typing import TYPE_CHECKING, Union, List, Dict, Tuple
from io import BytesIO

from face2face.core.compatibility.Face import Face
from media_toolkit import ImageFile, MediaFile, MediaList, media_from_any
from media_toolkit.utils.data_type_utils import is_valid_file_path

if TYPE_CHECKING:
    from face2face.core.face2face import Face2Face

# regular imports
import glob
import os
import numpy as np


from face2face.core.modules.file_writable_face import FileWriteableFace
from face2face.settings import EMBEDDINGS_DIR
from face2face.core.modules.utils import encode_path_safe


class _FaceEmbedding:
    def convert_to_face(self, media: Union[ImageFile, MediaFile, str, Face]) -> List[Face]:
        """
        Convert various media inputs to face embeddings.

        Args:
            media: Input media to convert to faces. Can be:
                - Face: Returns the face as-is
                - ImageFile: Detects faces and returns them
                - MediaFile: Loads face embedding from file
                - str: path to image/embedding file

        Returns:
            List[Face]: List of face objects
        """
        if isinstance(media, Face):
            return [media]
        
        # Detect faces in ImageFile or image path
        if isinstance(media, ImageFile):
            detected_faces = self.detect_faces(media)
            if not detected_faces:
                raise ValueError(f"No faces detected in {media}")
            return detected_faces

        # Face embedding stored in MediaFile
        if isinstance(media, MediaFile) and not isinstance(media, ImageFile):
            return [self._load_reference_face_from_file(media.to_bytes_io())]
            
        # Face_name or path
        if isinstance(media, str) and is_valid_file_path(media):
            mf = media_from_any(media)
            return self.convert_to_face(mf)

        raise ValueError("Could not convert media to face")
        
    @staticmethod
    def _load_reference_face_from_file(face_embedding_file_path: Union[str, BytesIO]) -> Union[Face, None]:
        """
        Load a reference face from a file or BytesIO buffer.
        
        Args:
            face_embedding_file_path: Source of the face embedding. Can be:
                - str: Path to a .npy file containing the face embedding
                - BytesIO: Buffer containing the face embedding data

        Returns:
            Face: The loaded face object with its embedding and attributes
            None: If the file is not found or cannot be loaded
        """
        if not isinstance(face_embedding_file_path, (str, BytesIO)):
            return None
        
        if not is_valid_file_path(face_embedding_file_path):
            return None

        try:
            # note: potential security issue, if the file was not created with face2face
            embedding = np.load(face_embedding_file_path, allow_pickle=True)
            return FileWriteableFace.to_face(embedding)
        except Exception as e:
            print(f"Error loading reference face {face_embedding_file_path}: {e}")
            return None

    def _load_from_cache(self, face_name: str) -> Union[Face, None]:
        """
        Load a reference face from the _face_embeddings memory cache.
        If the face is not in the cache, returns None.
        """
        if not isinstance(face_name, str):
            return None
        
        normalized_face_name = os.path.basename(encode_path_safe(face_name))
        return self._face_embeddings.get(normalized_face_name)

    def _load_from_embeddings_dir(self, face_name: str) -> Union[Face, None]:
        """
        Load a reference face from the _face_embeddings folder.
        """
        embedding_name = os.path.join(EMBEDDINGS_DIR, os.path.basename(encode_path_safe(face_name)))
        extensions = [".npy", ".npz"]
        for ext in extensions:
            if is_valid_file_path(embedding_name + ext):
                return self._load_reference_face_from_file(embedding_name + ext)
        return None

    def load_all_face_embeddings(self: Face2Face) -> Dict[str, Face]:
        """
        Load all face embeddings from the _face_embeddings folder.
        
        Returns:
            Dict[str, Face]: Dictionary mapping face names to their embeddings
        """
        for face_name in glob.glob(self._face_embedding_folder + "/*.npy"):
            self._load_reference_face_from_file(face_name)
        return self._face_embeddings

    def _determine_face_name(self, source: Union[str, MediaFile]) -> str:
        """
        Determine a unique name for a face.
        """
        if isinstance(source, str):
            return source
        elif isinstance(source, MediaFile):
            return source.file_name
    
        return "face_" + str(uuid.uuid4())
        
    def get_faces(
        self, 
        faces: Union[str, Face, ImageFile, MediaFile, list, MediaList]
    ) -> Dict[str, Face]:
        """
        Load face embeddings from various sources.
        
        Args:
            faces: The source of face embeddings to load. Can be:
                - str: 
                    * registered face name → loads the face from the _face_embeddings folder
                    * Path/URL to an image → faces are detected from the image and used as swap targets.
                    * Path/URL to an embedding file (.npy, etc.).
                - MediaFile: Loads a face from the file content.
                - ImageFile: detects the face in the image and adds it temporarily to the face embeddings
                - Face: returns the face as is
                - List/MediaList: Loads multiple face embeddings how to proceed is defined for each item in the list.
        Returns:
            Dict[str, Face]: Dictionary mapping face names to their embeddings.
                For MediaFile/ImageFile inputs, uses the file name as the key.
                For MediaList inputs, uses file names or string paths as keys.
                For List[Face] inputs, uses numeric indices as keys.
                If a file has more than one face we enumerate the faces left to right per file.
        Raises:
            ValueError: If no face could be loaded
        """
        if faces is None:
            raise ValueError("Please provide at least one face name or input")
        
        # Convert to MediaList for unified handling
        if not isinstance(faces, (list, MediaList)):
            faces = [faces]

        strategies = [
            self._load_from_cache,
            self._load_from_embeddings_dir,
            self.convert_to_face,
        ]

        converted_faces = {}
        # Process each item in the list
        for i, media in enumerate(faces):
            for strategy in strategies:
                new_faces = strategy(media)
                if new_faces is None:
                    continue

                if not isinstance(new_faces, list):
                    new_faces = [new_faces]
                 
                for j, face in enumerate(new_faces):
                    face_name = self._determine_face_name(media)
                    if len(new_faces) > 1:
                        face_name = f"{face_name}_{j}"
                    converted_faces[face_name] = face
                break
  
        if len(converted_faces) == 0:
            raise ValueError("No faces could be loaded")
        
        return converted_faces

    def add_face(
        self: Face2Face,
        face_name: Union[str, List[str]],
        media: Union[np.array, str, ImageFile, MediaFile],
        save: bool = False
    ) -> Union[Tuple[str, Face], Dict[str, Face]]:
        """
        Add one or multiple reference face(s) to the face swapper. This face(s) can be used for swapping in other images.

        Args:
            face_name: The name(s) for the reference face(s). In an image with multiple faces:
                - If a single string, creates one face embedding for the first face in the image
                - If a list of strings, creates embeddings for each face from left to right in the image
            media: The media from which to extract the face(s). Can be:
                - numpy array: Direct image data
                - str: Path/URL to image file
                - ImageFile: MediaFile containing the image
                - MediaFile: A media file containing a face embedding
            save: If True, saves the face embeddings to disk in the _face_embeddings folder.
                If False, only stores them in memory.

        Returns:
            Single face:: Tuple with the face name and its embedding
            Multiple faces: Dictionary with the face names and their embeddings {face_name: Face}

        Raises:
            ValueError: If no face name is provided or no faces are detected in the image
        """
        # Validate face_name parameter
        if isinstance(face_name, list) and len(face_name) == 0 or face_name is None:
            raise ValueError("Please provide at least one face name.")

        # Convert media to faces using the unified convert_to_face method
        try:
            faces_list = self.convert_to_face(media)
        except Exception as e:
            raise ValueError(f"Could not convert media to faces: {e}")

        # Handle face naming
        if not isinstance(face_name, list):
            face_name = [face_name]

        # Handle case where we have more names than faces
        if len(face_name) > len(faces_list):
            print(f"Not enough faces in the media for all provided face names. "
                  f"Only {len(faces_list)} faces found. Using first {len(faces_list)} names.")
            face_name = face_name[:len(faces_list)]

        # Handle case where we have fewer names than faces
        elif len(face_name) < len(faces_list):
            print(f"More faces detected ({len(faces_list)}) than names provided ({len(face_name)}). "
                  f"Using provided names and generating additional names.")
            # Keep provided names and generate additional ones
            for i in range(len(face_name), len(faces_list)):
                face_name.append(f"{face_name[0]}_{i}" if face_name else f"face_{i}")

        result_faces = {}

        # Store faces with proper names
        for i, name in enumerate(face_name):
            if i >= len(faces_list):
                break

            encoded_name = encode_path_safe(name)
            face = faces_list[i]

            # Store in memory
            self._face_embeddings[encoded_name] = face

            # Optionally save to disk
            if save:
                os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
                filename = os.path.join(EMBEDDINGS_DIR, f"{encoded_name}.npy")
                if os.path.exists(filename):
                    print(f"Reference face {encoded_name} already exists. Overwriting.")
                FileWriteableFace(face).to_file(filename)

            result_faces[encoded_name] = face

        if len(result_faces) == 1:
            return result_faces.popitem()

        return result_faces
