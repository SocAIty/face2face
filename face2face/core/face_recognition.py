from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING, Union, List, Dict

from insightface.app.common import Face

from face2face.utils.utils import load_image

# avoid circular dependency but provide type hints
if TYPE_CHECKING:
    from face2face.core.face2face import Face2Face

import glob
import numpy as np


class _FaceRecognition:
    """
    Face recognition - Mixin. Provides functions for identification, and swapping of known reference faces.
    """

    def load_all_reference_faces(self: Face2Face):
        """
        Load all reference faces.
        """
        for face_name in glob.glob(self._reference_faces_folder + "/*.npz"):
            self.load_reference_embedding(face_name)

    def calculate_face_distances(
            self: Face2Face,
            face_list_1: List[Face],
            face_list_2: Union[List[Face], Dict[str, Face]]
    ) -> list:
        """
        Calculate the face distances between the detected faces and the reference faces.
        :param face_list_1: the detected faces in the image
        :param face_list_2:
            the reference faces to calculate the distance to. If list, the faces are enumerated to dict with index as key.
            If in a loaded face embedding there are multiple faces; only the first face is used.
        :return: the face distances as a list of ordered and distance sorted dictionaries in form of
            [{face_1_0_to_face_2_0: distance, face_1_0_to_face_2_1: distance}, {face_1_1_to_face_2_0: distance, ...}  ...]
        """
        if len(face_list_1) == 0 or len(face_list_2) == 0:
            return {}

        if isinstance(face_list_2, list):
            face_list_2 = dict(enumerate(face_list_2))

        # flatten embeddings with multiple faces
        face_list_2_flat = {}
        for face_name, face in face_list_2.items():
            if isinstance(face, list):
                face_list_2_flat[face_name] = face[0]
            else:
                face_list_2_flat[face_name] = face

        # Calculate distances
        face_distances = []  # [{face1_0: distance, face2_0: distance ..}, ...]
        for i, face in enumerate(face_list_1):
            face_dists = {}
            for reference_face_name, reference_face in face_list_2_flat.items():
                dist = self.calc_face_distance(face, reference_face)
                face_dists[reference_face_name] = dist
            face_dists = OrderedDict(sorted(face_dists.items(), key=lambda x: x[1], reverse=False))
            face_distances.append(face_dists)
        return face_distances

    # def swap_known_face_to_target(self: Face2Face, face_list_1: list, source_face: str, target_face: str):
    #    """
    #    1. Identify all persons in an image.
    #    2. Swap target_reference face to source_reference face.
    #    :param face: the face
    #    :param target_face: the target face
    #    :return: the swapped face
    #    """
    #    face_distances = self.calculate_ref_face_distances(face_list_1)
    #
    #
    #    target_faces = []
    #    for face in face_list_1:
    #        if face_distances[face][source_face] < 0.5:
    #            target_faces.append(face)
    #        else:
    #            # This makes sure, that the order of the faces is preserved
    #            # In addition, these faces are not swapped
    #            target_faces.append(None)

    def swap_known_faces_to_target_faces(
            self: Face2Face,
            image: Union[np.array, str],
            swap_pairs: dict,
            enhance_face_model: str = 'gpen_bfr_512',
            threshold: float = 0.5
    ):
        """
        Based on the swap_pairs, swap the source faces to the target faces if they are recognized.
        1. Identify all persons in an image, by calculating the cosine distance between the embeddings >= threshold.
        2. Call the swap function with the target faces.

        """
        image = load_image(image)

        # Load reference faces
        for f1, f2 in swap_pairs.items():
            self.load_reference_embedding(f1)
            self.load_reference_embedding(f2)
        ref_faces = {f: self.reference_faces[f] for f in swap_pairs.keys()}

        # if there's more than one reference face in an embedding remove it
        ref_faces = {
            k: v[0] if isinstance(v, list) else v
            for k, v in ref_faces.items()
        }

        # Detect faces and calculate face distances to source reference faces
        detected_faces = self.get_many_faces(image)
        face_distances = self.calculate_face_distances(detected_faces, ref_faces)

        # prepare target_face vector based on the most similar reference face for each detected face
        # The source faces vector will have same length as the detected faces
        _source_faces = []
        for i, face in enumerate(detected_faces):
            # check if the face is recognized
            closest_face, dist = next(iter(face_distances[i].items()), (None, 1))  # is a sorted dict
            if closest_face is not None and dist < threshold:
                # get swap partner
                swap_partner = swap_pairs[closest_face]
                swap_partner_face = self.reference_faces[swap_partner]
                if isinstance(swap_partner_face, list):
                    swap_partner_face = swap_partner_face[0]

                _source_faces.append(swap_partner_face)
            else:
                _source_faces.append(None)

        # swap the faces to the target faces
        return self._swap_detected_faces(
            source_faces=_source_faces,
            target_faces=detected_faces,
            target_image=image,
            enhance_face_model=enhance_face_model
        )

    # @staticmethod
    # def cosine_distance(embedding1, embedding2):
    #    """
    #    Calculate the cosine distance between two embeddings.
    #    :param embedding1: the first embedding
    #    :param embedding2: the second embedding
    #    :return: the cosine distance
    #    """
    #    return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

    @staticmethod
    def calc_face_distance(face: Face, reference_face: Face) -> float:
        if hasattr(face, 'normed_embedding') and hasattr(reference_face, 'normed_embedding'):
            return 1 - np.dot(face.normed_embedding, reference_face.normed_embedding)
        return 1
