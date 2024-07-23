# avoid circular dependency but provide type hints
import glob
import os
from io import BytesIO
from typing import TYPE_CHECKING, Union, List, Tuple

import numpy as np
from insightface.app.common import Face

from face2face.modules.storage.f2f_loader import load_reference_face_from_file
from face2face.modules.storage.file_writable_face import FileWriteableFace
from face2face.settings import REF_FACES_DIR
from face2face.modules.utils import encode_path_safe
from face2face.modules.utils.utils import load_image

if TYPE_CHECKING:
    from face2face.core.face2face import Face2Face


class _FaceEmbedding:
    def load_face(self: Face2Face, face_name: str) -> Union[List[Face], None]:
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
        file = os.path.join(self._face_embedding_folder, f"{face_name}.npz")
        embedding = load_reference_face_from_file(file)

        if embedding is None:
            raise ValueError(f"Reference face {face_name} not found. "
                             f"Please add the reference face first with add_reference_face")

        # convert embedding to face
        embedding = [Face(face) for face in embedding]

        # add to memory dict
        self._face_embeddings[face_name] = embedding
        return embedding

    def load_faces(self, face_names: Union[str, List[str], None] = None) -> dict:
        """
        :param face_names: the faces to load from the _face_embeddings folder.
            If None all stored face_embeddings are loaded and returned.
        :return: the loaded faces as dict {face_name: face_embedding}
        """
        if face_names is None:
            return self.load_all_faces()
        elif isinstance(face_names, str):
            return {face_names: self.load_face(face_names)}
        else:
            return {face_name: self.load_face(face_name) for face_name in face_names}

    def load_all_faces(self: Face2Face):
        """
        Load all face embeddings from the _face_embeddings folder.
        """
        for face_name in glob.glob(self._face_embedding_folder + "/*.npz"):
            self.load_face(face_name)
        return self._face_embeddings

    def add_face(self: Face2Face, face_name: str, ref_image: Union[np.array, str], save=False) -> Tuple[str, np.array]:
        """
        Add a reference face to the face swapper. The face swapper will use this face to swap it to other images.
        Use the method swap_from_reference_face to swap the reference face to other images.
        :param face_name: how the reference face is called
        :param ref_image: the image to get the faces from
        :param save: if True, the reference face will be saved to the _face_embeddings folder and available next startup
        :return: the savely encoded face name and the reference face
        """
        face_name = encode_path_safe(face_name)
        ref_image = load_image(ref_image)

        self._face_embeddings[face_name] = self.detect_faces(ref_image)

        # make faces pickle able by converting them to FileWriteableFace
        save_able_ref_faces = [FileWriteableFace(face) for face in self._face_embeddings[face_name]]

        # save face to virtual file
        virtual_file = BytesIO()
        np.save(virtual_file, arr=save_able_ref_faces, allow_pickle=True)
        virtual_file.seek(0)
        if save:
            if not os.path.isdir(REF_FACES_DIR):
                os.makedirs(REF_FACES_DIR)

            filename = os.path.join(REF_FACES_DIR, f"{face_name}.npz")
            if os.path.isfile(filename):
                print(f"Reference face {face_name} already exists. Overwriting.")

            with open(filename, "wb") as f:
                f.write(virtual_file.getbuffer())

        virtual_file.seek(0)
        return face_name, virtual_file
