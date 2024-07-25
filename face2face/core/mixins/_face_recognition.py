# avoid circular dependency but provide type hints
from __future__ import annotations
from typing import TYPE_CHECKING, Union, List, Dict

from media_toolkit import ImageFile

if TYPE_CHECKING:
    from face2face.core.face2face import Face2Face

# normal imports
from collections import OrderedDict
from insightface.app.common import Face
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
        image = ImageFile().from_any(image)
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
        image = ImageFile().from_any(image)

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
            image=image,
            enhance_face_model=enhance_face_model
        )

    def swap_pairs_generator(
            self: Face2Face,
            swap_pairs: dict,
            image_generator,
            enhance_face_model: Union[str, None] = 'gpen_bfr_2048',
            recognition_threshold: float = 0.5
    ):
        """
        Swaps the reference faces in the target image.
        :param swap_pairs: a dict with the structure {source_face_name: target_face_name}
        :param image_generator: a generator that yields images in BGR format (read with cv2).
            Or a video_stream that yields (image, audio) like in media_toolkit.
        :return: a generator that yields the swapped images or tuples (image, audio)
        """
        if not isinstance(swap_pairs, dict):
            raise ValueError("Please provide a dict with the structure {source_face_name: target_face_name}")

        for i, target_image in enumerate(image_generator):
            # check if generator yields tuples (video, audio) or only images
            audio = None
            if isinstance(target_image, tuple) and len(target_image) == 2:
                target_image, audio = target_image

            try:
                swapped = self.swap_pairs(
                    swap_pairs=swap_pairs,
                    image=target_image,
                    enhance_face_model=enhance_face_model,
                    threshold=recognition_threshold
                )

                if audio is not None:
                    yield swapped, audio
                    continue

                yield swapped
                continue
            except Exception as e:
                print(f"Error in swapping frame {i}: {e}. Skipping image")
                if audio is not None:
                    yield target_image, audio
                    continue

                yield np.array(target_image)

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
        :return: the { faces: first_face_embedding }
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
