# ordinary imports
from io import BytesIO
from typing import List, Union, Tuple
import copy
import os

# image processing imports
import numpy as np
# model imports
import insightface
import onnxruntime

from face2face.core.file_writable_face import FileWriteableFace
from face2face.utils.utils import encode_path_safe, get_files_in_dir, download_file
from face2face.settings import MODELS_DIR, REF_FACES_DIR, MODEL_DOWNLOAD_URL


class Face2Face:
    def __init__(self,
                 model_path: str = None,
                 inswapper_model_name: str = "inswapper_128.onnx",
                 reference_faces_folder: str = None,
                 ):
        """
        :param model_path: the folder where inswapper model and the buffalo_l model is stored.
        :param inswapper_model_name:
        """

        # prepare models
        if model_path is None:
            model_path = os.path.join(MODELS_DIR, 'insightface')

        # download models if not existing
        model_file_path = os.path.join(model_path, inswapper_model_name)
        if not os.path.isfile(model_file_path):
            download_file(MODEL_DOWNLOAD_URL, model_file_path)

        if reference_faces_folder is None:
            reference_faces_folder = REF_FACES_DIR

        self.providers = onnxruntime.get_available_providers()
        self._face_analyser = self._get_face_analyser(model_path, self.providers)
        self._face_swapper = insightface.model_zoo.get_model(model_file_path)

        # face swapper has the option to swap from image to image or
        # to have a reference images with reference faces and apply them to an image
        # they dict has structure {face_name: detected faces }
        self._reference_faces_folder = reference_faces_folder
        self.reference_faces = {}

    def _get_face_analyser(self, model_path: str, providers, det_size=(320, 320)):
        """Get the face analyser model. The face analyser detects faces and extracts facial features."""
        # load default face analyser if model_path is None
        face_analyser = insightface.app.FaceAnalysis(
            name="buffalo_l", root=f"{model_path}./checkpoints", providers=providers
        )
        face_analyser.prepare(ctx_id=0, det_size=det_size)
        # TODO: load face analyser from file if model_path is not None

        return face_analyser

    def get_many_faces(self, frame: np.ndarray) -> Union[List | None]:
        """
        get faces from left to right by order
        """
        try:
            face = self._face_analyser.get(frame)
            return sorted(face, key=lambda x: x.bbox[0])
        except IndexError:
            return None


    def _swap_detected_faces(self, source_faces, target_faces, target_image: np.array) -> np.array:
        """
        Changes the face(s) of the target image to the face of the source image.
        source_faces: the source faces
        target_image: the target image in BGR format (read with cv2)
        """

        if source_faces is None:
            raise Exception("No source faces found!")

        if target_faces is None:
            print(f"No face found in {target_image}. Return image as is")
            return target_image

        result = copy.deepcopy(target_image)

        for target_index in range(len(target_faces)):
            result = self._face_swapper.get(
                result,  # in place operation
                target_faces[target_index],  # There is only one source face.
                source_faces[0],
                paste_back=True,
            )

        return result

    def swap_one_image(
        self, source_image: np.array, target_image: np.array
    ) -> np.array:
        """
        Changes the face(s) of the target image to the face of the source image.
        source_image: the source image in BGR format (read with cv2)
        target_image: the target image in BGR format (read with cv2)
        """

        # get the bounding box of the faces
        source_faces = self.get_many_faces(source_image)

        if source_faces is None:
            raise Exception("No source faces found!")

        target_faces = self.get_many_faces(target_image)
        return self._swap_detected_faces(source_faces, target_faces, target_image)

    def load_reference_face(self, face_name: str):
        """
        Load a reference face from a file.
        :param face_name: the name of the reference face
        :return: the reference face
        """
        file = os.path.join(self._reference_faces_folder, f"{face_name}.npz")
        embedding = Face2Face.__load_reference_face_from_file(file)
        self.reference_faces[face_name] = embedding
        return embedding

    @staticmethod
    def __load_reference_face_from_file(face_embedding_file_path: str) -> Union[FileWriteableFace, None]:
        """
        Load a reference face from a file.
        :param face_embedding_file_path: the file path of the reference face
        :return: the reference face
        """
        if not os.path.exists(face_embedding_file_path):
            print(f"Reference face {face_embedding_file_path} not found.")
            return None

        try:
            read_face = np.load(face_embedding_file_path, allow_pickle=True)
            return FileWriteableFace.to_face(read_face)
        except Exception as e:
            print(f"Error loading reference face {face_embedding_file_path}: {e}")

    @staticmethod
    def __load_reference_faces_from_folder(folder_path: str) -> dict:
        """
        Load reference faces from a folder. The folder must contain .npz files with the reference faces.
        The file name will be used as the reference face name.
        :param folder_path: the folder path
        :return:
        """
        files = get_files_in_dir(folder_path, [".npz"])
        reference_faces = {}
        for file in files:
            face_name = os.path.basename(file)[:-4]
            embedding = Face2Face.__load_reference_face_from_file(file)
            if embedding is None:
                continue
            reference_faces[face_name] = embedding

        return reference_faces

    def add_reference_face(self, face_name: str, ref_image: np.array, save=False) -> Tuple[str, np.array]:
        """
        Add a reference face to the face swapper. The face swapper will use this face to swap it to other images.
        Use the method swap_from_reference_face to swap the reference face to other images.
        :param face_name: how the reference face is called
        :param ref_image: the image to get the faces from
        :param save: if True, the reference face will be saved to the reference_faces folder and available next startup
        :return: the savely encoded face name and the reference face
        """
        face_name = encode_path_safe(face_name)

        self.reference_faces[face_name] = self.get_many_faces(ref_image)

        # make classes pickle safe
        for i, face in enumerate(self.reference_faces[face_name]):
            self.reference_faces[face_name][i] = FileWriteableFace(face)

        # save face to virtual file
        virtual_file = BytesIO()
        np.save(virtual_file, self.reference_faces[face_name], allow_pickle=True)

        if save:
            if not os.path.isdir(REF_FACES_DIR):
                os.makedirs(REF_FACES_DIR)

            filename = os.path.join(REF_FACES_DIR, f"{face_name}.npz")
            if os.path.isfile(filename):
                print(f"Reference face {face_name} already exists. Overwriting.")

            with open(filename, "wb") as f:
                f.write(virtual_file.getbuffer())

        return face_name, virtual_file

    def swap_from_reference_face(self, face_name: str, target_image: Union[np.array, list]) -> np.array:
        """
        Changes the face(s) of the target image to the face of the reference image.
        :param face_name: the name of the reference face
        :param target_image: the target image in BGR format (read with cv2). Can be a list of images
        :return: the swapped image
        """
        face_name = encode_path_safe(face_name)

        # load reference face if not existing
        if face_name not in self.reference_faces:
            embedding = self.load_reference_face(face_name)
            if embedding is None:
                raise Exception(f"Reference face {face_name} not found. "
                            f"Please add the reference face first with add_reference_face")

        if type(target_image) == list:
            # if target_image is a list of images, swap all images
            return [self.swap_from_reference_face(face_name, img) for img in target_image]

        target_faces = self.get_many_faces(target_image)
        source_faces = [FileWriteableFace.to_face(fwf) for fwf in self.reference_faces[face_name]]

        return self._swap_detected_faces(source_faces, target_faces, target_image)
