# ordinary imports
from io import BytesIO
from typing import List, Union
import copy
import os

from insightface.app.common import Face
from tqdm import tqdm

# image processing imports
import numpy as np

# model imports
import insightface
import onnxruntime

import pickle

from .utils import encode_path_safe, get_files_in_dir
from .settings import MODELS_DIR, REF_FACES_DIR


class FileWriteableFace(dict):
    """
    Enables the insightface Face class to be saved.
    To recast the object simply use new_face = Face(pickleSafeFace)
    The problem was that Face overwrites __getattr__ which all the pickle methods use.
    In addition, some np types are not pickleable.
    """

    def __init__(self, face: Face):
        super().__init__()
        # copy all attributes
        for attr in dir(face):
            if not attr.startswith("__"):
                try:
                    self[attr] = face[attr]
                except:
                    try:
                        if type(face.attr) in [
                            np.float32,
                            np.float64,
                            np.int32,
                            np.int64,
                        ]:
                            self.attr = (float)(face[attr])
                        else:
                            self.attr = face.attr
                    except:
                        pass

    @staticmethod
    def to_face(pickle_safe_face):
        if type(pickle_safe_face) == Face:
            return pickle_safe_face
        f = Face(pickle_safe_face)
        return f


class FaceSwapper:
    def __init__(
        self,
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

        if reference_faces_folder is None:
            reference_faces_folder = REF_FACES_DIR

        self.providers = onnxruntime.get_available_providers()
        self.face_analyser = self._get_face_analyser(model_path, self.providers)
        self.face_swapper = insightface.model_zoo.get_model(
            os.path.join(model_path, inswapper_model_name)
        )

        # face swapper has the option to swap from image to image or
        # to have a reference images with reference faces and apply them to an image
        # they dict has structure {face_name: detected faces }
        self.reference_faces = self.__load_reference_faces_from_folder(
            reference_faces_folder
        )

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
            face = self.face_analyser.get(frame)
            return sorted(face, key=lambda x: x.bbox[0])
        except IndexError:
            return None

    def _swap_detected_faces(
        self, source_faces, target_faces, target_image: np.array
    ) -> np.array:
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
            result = self.face_swapper.get(
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

    @staticmethod
    def __load_reference_faces_from_folder(folder_path: str) -> dict:
        """
        Load reference faces from a folder. The folder must contain .npy files with the reference faces.
        The file name will be used as the reference face name.
        :param folder_path: the folder path
        :return:
        """
        files = get_files_in_dir(folder_path, [".npz"])
        reference_faces = {}
        for file in files:
            face_name = os.path.basename(file)[:-4]
            try:
                saved_faces = np.load(
                    os.path.join(folder_path, file), allow_pickle=True
                )
                reference_faces[face_name] = [
                    FileWriteableFace.to_face(fa) for fa in saved_faces
                ]
            except Exception as e:
                print(f"Error loading reference face {face_name}: {e}")
                continue

        return reference_faces

    def add_reference_face(
        self, face_name: str, ref_image: np.array, save=False
    ) -> Union[str, np.array]:
        """
        Add a reference face to the face swapper. The face swapper will use this face to swap it to other images.
        Use the method swap_from_ref_image to swap the reference face to other images.
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

    def swap_from_reference_face(
        self, face_name: str, target_image: Union[np.array, list]
    ) -> np.array:
        """
        Changes the face(s) of the target image to the face of the reference image.
        :param face_name: the name of the reference face
        :param target_image: the target image in BGR format (read with cv2). Can be a list of images
        :return: the swapped image
        """
        face_name = encode_path_safe(face_name)

        if face_name not in self.reference_faces:
            raise Exception(
                f"Reference face {face_name} not found. Please add the reference face first with add_reference_face"
            )

        if type(target_image) == list:
            # if target_image is a list of images, swap all images
            return [
                self.swap_from_reference_face(face_name, img) for img in target_image
            ]

        target_faces = self.get_many_faces(target_image)
        source_faces = [
            FileWriteableFace.to_face(fwf) for fwf in self.reference_faces[face_name]
        ]

        return self._swap_detected_faces(source_faces, target_faces, target_image)


# if __name__ == "__main__":
# test face swapper
# source_img_path = "D:\\Assets\\Potter_Redcliff\\redcliff_closeup.jpg"
# target_images_path = "D:\\youtube\\ue5_first_shot"
# output_path = f"{target_images_path}/swapped/"
#
# target_images = get_files_in_dir(target_images_path, [".jpeg"])
#
# faceswapper_redcliff = FaceSwapperFixedIndividuals(reference_image_path=source_img_path)
# faceswapper_redcliff.swap_many(target_images, output_path)
## create a video from images
# swapped_imgs = get_files_in_dir(output_path, [".jpeg"])

# make_video_from_images(swapped_imgs, output_path + "output.mp4", frame_rate=120)
