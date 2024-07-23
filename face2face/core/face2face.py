# ordinary imports
from typing import List, Union
import copy
import os
import numpy as np

import insightface
import onnxruntime
from insightface.app.common import Face

from face2face.core.mixins._face_embedding import _FaceEmbedding
from face2face.core.mixins._face_recognition import _FaceRecognition
from face2face.core.mixins._video_swap import _Video_Swap

from face2face.modules.storage.f2f_loader import get_face_analyser
from face2face.modules.utils.utils import encode_path_safe, download_model, load_image
from face2face.settings import MODELS_DIR, REF_FACES_DIR, DEVICE_ID
from face2face.modules.face_enhance.face_enhancer import enhance_face


class Face2Face(_FaceEmbedding, _FaceRecognition, _Video_Swap):
    def __init__(self, face_embedding_folder: str = None, device_id: int = None):
        """
        :param model_path: the folder where the models are stored and downloaded to.
            results in structure like models/insightface/inswapper_128.onnx model
            and models/face_enhancer/gfpgan_1.4.onnx
        :param inswapper_model_name:
        """
        # download inswapper model (roop) if not existing
        swapper_model_file_path = download_model("inswapper_128")
        face_analyzer_models_path = os.path.join(MODELS_DIR, 'insightface')

        self.providers = onnxruntime.get_available_providers()
        # Setting GPU number
        if device_id is None:
            device_id = DEVICE_ID

        if "CUDAExecutionProvider" in self.providers:
            self.providers.remove("CUDAExecutionProvider")
            self.providers.append(("CUDAExecutionProvider", {'device_id': device_id}))
            self.providers = [("CUDAExecutionProvider", {'device_id': device_id})]

        self._face_analyser = get_face_analyser(face_analyzer_models_path, self.providers)
        self._face_swapper = insightface.model_zoo.get_model(swapper_model_file_path, providers=self.providers)

        # face swapper has the option to swap images from previously stored faces as embeddings
        # they dict has structure {face_name: face_embedding }
        if face_embedding_folder is None:
            face_embedding_folder = REF_FACES_DIR

        self._face_embedding_folder = face_embedding_folder
        self._face_embeddings = {}

    def _swap_faces(
            self,
            source_faces: List[Face],
            target_faces: List[Face],
            target_image: np.array,
            enhance_face_model: str = 'gpen_bfr_512'
    ) -> np.array:
        """
        Changes the face(s) of the target image to the face(s) of the source image.
        if there are more target faces than source faces, the source face index is reset and starts again left->right.
        source_faces: the source faces from left to right [face1, None, face3, ... ]
        target_faces: the target faces from left to right [face1, face2, face3, ... ].
        target_image: the target image in BGR format (read with cv2)
        enhance_face_model: if str, the faces will be enhanced with the given face enhancer model.
            if none the faces will not be enhanced
        """
        if source_faces is None or len(source_faces) == 0:
            raise Exception("No source faces found!")

        if target_faces is None or len(target_faces) == 0:
            print(f"No face found in image. Return image as is")
            return target_image

        # make sure face enhance model is downloaded
        if enhance_face_model is not None:
            download_model(enhance_face_model)

        result = copy.deepcopy(target_image)

        # iter through all target faces and swap them with the source faces
        # if there are more target faces than source faces, the source face index is reset
        for target_index in range(len(target_faces)):
            source_index = target_index % len(source_faces)  # 6 % 5 = 1 and 1 % 5 = 1 ...

            # having none values in the array allows users to skip faces
            source_face = source_faces[source_index]
            if source_face is None:
                continue

            result = self._face_swapper.get(
                result,  # in place operation
                target_faces[target_index],
                source_face,
                paste_back=True,
            )
            if enhance_face_model is not None:
                try:
                    result = enhance_face(
                        target_face=target_faces[target_index], temp_vision_frame=result, model=enhance_face_model
                    )
                except Exception as e:
                    print(f"Error in enhancing face {target_index}: {e}. Returning lowres swap instead.")

        return result

    def swap_one_image(
        self,
        source_image: Union[np.array, str],
        target_image: Union[np.array, str],
        enhance_face_model: str = 'gpen_bfr_512'
    ) -> np.array:
        """
        Changes the face(s) of the target image to the face(s) of the source image.
        source_image: the source image in BGR format (read with cv2)
        target_image: the target image in BGR format (read with cv2)
        """
        source_image = load_image(source_image)
        target_image = load_image(target_image)

        # get the bounding box of the faces
        source_faces = self.detect_faces(source_image)

        if source_faces is None:
            raise Exception("No source faces found!")

        target_faces = self.detect_faces(target_image)
        return self._swap_faces(source_faces, target_faces, target_image, enhance_face_model)

    def detect_faces(self, frame: np.ndarray) -> Union[List | None]:
        """
        get faces from left to right by order
        """
        try:
            face = self._face_analyser.get(frame)
            return sorted(face, key=lambda x: x.bbox[0])
        except IndexError:
            return None

    def swap_to_face(
            self,
            face_name: str,
            target_image: Union[np.array, list],
            enhance_face_model: str | None = 'gpen_bfr_2048'
        ) -> np.array:
        """
        Changes the face(s) of the target image to the face of the reference image.
        :param face_name: the name of the reference face
        :param target_image: the target image in BGR format (read with cv2). Can be a list of images
        :return: the swapped image
        """
        face_name = encode_path_safe(face_name)

        # if target_image is a list of images, swap all images
        if isinstance(target_image, list):
            return list(self.swap_to_face_generator(face_name, target_image))

        # swap single image
        source_faces = self.load_face(face_name)
        target_faces = self.detect_faces(target_image)
        return self._swap_faces(source_faces, target_faces, target_image, enhance_face_model)

    def swap_to_face_generator(
            self,
            face_name: str,
            image_generator,
            enhance_face_model: str = 'gpen_bfr_2048'
    ):
        """
        Changes the face(s) of each image in the target_img_generator to the face of the reference image.
        :param face_name: the name of the reference face
        :param image_generator: a generator that yields images in BGR format (read with cv2).
            Or a video_stream that yields (image, audio) like VideoFile().to_video_stream() in media_toolkit.
        :return: a generator that yields the swapped images or tuples (image, audio)
        """
        face_name = encode_path_safe(face_name)
        source_faces = self.load_face(face_name)

        for i, target_image in enumerate(image_generator):
            # check if generator yields tuples (video, audio) or only images
            audio = None
            if isinstance(target_image, tuple) and len(target_image) == 2:
                target_image, audio = target_image

            try:
                target_faces = self.detect_faces(target_image)
                swapped = self._swap_faces(source_faces, target_faces, target_image, enhance_face_model)
                if audio is not None:
                    yield swapped, audio
                    continue

                yield swapped
            except Exception as e:
                print(f"Error in swapping frame {i} to {face_name}: {e}. Skipping image")
                if audio is not None:
                    yield target_image, audio
                    continue

                yield np.array(target_image)

    def swap_faces_to_faces_generator(
            self,
            swap_pairs: dict,
            image_generator,
            enhance_face_model: str = 'gpen_bfr_2048'
    ):
        """
        Swaps the reference faces in the target image.
        :param swap_pairs: a dict with the structure {source_face_name: target_face_name}
        :param image_generator: a generator that yields images in BGR format (read with cv2).
            Or a video_stream that yields (image, audio) like in media_toolkit.
        :return: a generator that yields the swapped images or tuples (image, audio)
        """
        for i, target_image in enumerate(image_generator):
            # check if generator yields tuples (video, audio) or only images
            audio = None
            if isinstance(target_image, tuple) and len(target_image) == 2:
                target_image, audio = target_image

            try:
                swapped = self.swap_faces_to_faces(swap_pairs, target_image, enhance_face_model)

                if audio is not None:
                    yield swapped, audio
                    continue

                yield swapped
            except Exception as e:
                print(f"Error in swapping frame {i}: {e}. Skipping image")
                if audio is not None:
                    yield target_image, audio
                    continue

                yield np.array(target_image)


