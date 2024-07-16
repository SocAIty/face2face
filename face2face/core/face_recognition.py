from __future__ import annotations  # avoid circular import with import py_audio2face
from face2face import Face2Face

import glob
import numpy as np

class __FaceRecognition:
    """
    Face recognition - Mixin. Provides functions for identification, and swapping of faces.
    """
    def load_all_reference_faces(self: Face2Face):
        """
        Load all reference faces.
        """
        for face_name in glob.glob(self._reference_faces_folder + "/*.npz"):
            self.load_reference_embedding(face_name)

    def calculate_ref_face_distances(self: Face2Face, detected_faces: list):
        """
        Identify the faces given the reference faces.
        :param img: the image
        :param face_analyser: the face analyser
        :param face_size: the face size
        :return: the faces
        """
        self.load_all_reference_faces()

        face_distances = {}
        for face in detected_faces:
            for reference_face_name, reference_face in self._reference_faces.items():
                dist = self.cosine_distance(face.embedding, reference_face.embedding)
                face_distances[face][reference_face_name] = dist
        return face_distances

    def swap_known_face_to_target(self: Face2Face, detected_faces: list, source_face: str, target_face: str):
        """
        1. Identify all persons in an image.
        2. Swap target_reference face to source_reference face.
        :param face: the face
        :param target_face: the target face
        :return: the swapped face
        """
        face_distances = self.calculate_ref_face_distances(detected_faces)
        _swap_detected_faces
        for face in detected_faces:
            if face_distances[face][source_face] < 0.5:
                self.swap_one_image(face, self._reference_faces[target_face])


    @staticmethod
    def cosine_distance(embedding1, embedding2):
        """
        Calculate the cosine distance between two embeddings.
        :param embedding1: the first embedding
        :param embedding2: the second embedding
        :return: the cosine distance
        """
        return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))