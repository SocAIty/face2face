# avoid circular dependency but provide type hints
from __future__ import annotations
from typing import TYPE_CHECKING, Union, List, Dict
if TYPE_CHECKING:
    from face2face.core.face2face import Face2Face

# normal imports
from collections import OrderedDict
from insightface.app.common import Face
from face2face.core.modules.utils.utils import load_image
import numpy as np


class _FaceRecognition:
    """
    Face recognition - Mixin. Provides functions for identification, and swapping of known reference faces.
    """
    def face_recognition(
            self: Face2Face,
            image: np.array,
            face_names: Union[str, List[str], None] = None,
            threshold: float = 0.5
    ) -> list:
        """
        Given an image the method recognizes the faces in the image.
        :param image: the image to recognize the faces in
        :param face_names: the reference faces to compare the detected faces to.
            If None all stored face_embeddings are used.
        :param threshold: the threshold distance to recognize a face
        :return: A list with tuples in form [(recognized_face_name, dist, detected_face_in_image), ... ]
        """
        image = load_image(image)
        detected_faces = self.detect_faces(image)

        # Load reference faces
        ref_faces = self.load_faces(face_names)
        ref_faces = self._to_single_face_embeddings(ref_faces)

        # Detect faces and calculate face distances to source reference faces
        face_distances = self.calculate_face_distances(detected_faces, ref_faces)

        # prepare target_face vector based on the most similar reference face for each detected face
        # The source faces vector will have same length as the detected faces
        recognized_faces = []
        for i, face in enumerate(detected_faces):
            # check if the face is recognized
            closest_face, dist = next(iter(face_distances[i].items()), (None, 1))
            if closest_face is not None and dist < threshold:
                recognized_faces.append((closest_face, dist, face))
        return recognized_faces

    def swap_pairs(
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

        # recognize the faces of first swap partners in the image
        recognized_partner_faces = self.face_recognition(
            image,
            face_names=list(swap_pairs.keys()),
            threshold=threshold
        )

        # load the faces of the second swap partners
        swap_partner_embeddings = self.load_faces(face_names=list(swap_pairs.values()))
        swap_partner_embeddings = self._to_single_face_embeddings(swap_partner_embeddings)

        # swap faces of recognized partners to the second swap partners
        swap_embeddings = []
        for rec in recognized_partner_faces:
            partner = swap_pairs[rec[0]]
            swap_embeddings.append(swap_partner_embeddings[partner])

        return self._swap_faces(
            source_faces=swap_embeddings,
            target_faces=[rec[2] for rec in recognized_partner_faces],
            target_image=image,
            enhance_face_model=enhance_face_model
        )

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
        face_list_2_flat = self._to_single_face_embeddings(face_list_2)

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

    @staticmethod
    def _to_single_face_embeddings(embeddings: dict) -> dict:
        """
        An embedding can store multiple faces. This function loads only the first face.
        :param embeddings: the embeddings
        :return: the { face_name: first_face_embedding }
        """
        return {
            face_name: face[0] if isinstance(face, list) else face
            for face_name, face in embeddings.items()
        }

    @staticmethod
    def calc_face_distance(face: Face, reference_face: Face) -> float:
        if hasattr(face, 'normed_embedding') and hasattr(reference_face, 'normed_embedding'):
            return 1 - np.dot(face.normed_embedding, reference_face.normed_embedding)
        return 1


    # @staticmethod
    # def cosine_distance(embedding1, embedding2):
    #    """
    #    Calculate the cosine distance between two embeddings.
    #    :param embedding1: the first embedding
    #    :param embedding2: the second embedding
    #    :return: the cosine distance
    #    """
    #    return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
