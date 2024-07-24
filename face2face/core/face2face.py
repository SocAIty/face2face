# ordinary imports
from typing import List, Union
import copy
import os
import numpy as np

import insightface
import onnxruntime
from insightface.app.common import Face

from face2face.core.mixins._face_embedding import _FaceEmbedding
from face2face.core.mixins._face_enhance import _FaceEnhancer
from face2face.core.mixins._face_recognition import _FaceRecognition
from face2face.core.mixins._video_swap import _Video_Swap
from face2face.core.modules.face_enhance.face_enhancer import enhance_face

from face2face.core.modules.storage.f2f_loader import get_face_analyser
from face2face.core.modules.utils.utils import encode_path_safe, download_model, load_image
from face2face.settings import MODELS_DIR, REF_FACES_DIR, DEVICE_ID


class Face2Face(_FaceEmbedding, _FaceRecognition, _Video_Swap, _FaceEnhancer):
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

    # universal method still hidden until the others are refactored and simplified as well.
    def _face2face(
        self,
        img_video_or_pair: Union[np.array, str, tuple, List[str]],
        faces: Union[str, dict, list, List[Face], Face],
        enhance_face_model: str = 'gpen_bfr_512',
        include_audio: bool = True,
    ):
        """
        This is a unified function to swap faces in images and videos.
        :param img_video_or_pair: the image or video to swap the faces in.
            if str -> path to image or video. Load from file
            if np.array -> image
            if tuple -> (image, image). Same as swap_one
            if List[str] -> list of file paths. Perform swap on all of those files
            if generator -> generator that yields images or tuples (image, audio)
            if VideoFile -> video file

        :param faces: defines what to swap to.
            if str -> the name of the face to swap to. All faces in the image will be swapped to this face.
            if list -> the list of face_names or face objects to swap to from left to right.
            if dict -> the swap_pairs with the structure {source_face_name: target_face_name}. Use face recognition
                to swap the source faces to the target faces.
            if Face -> the face object to swap to. All faces in the image will be swapped to this face.

        :param enhance_face_model: the face enhancement model to use. Use None for no enhancement
        :param include_audio: if True, the audio will be included in the output video if the input is a video.
        """
        if isinstance(img_video_or_pair, str):
            # Todo: check if it's a video or image
            # load image
            # img_video_or_pair = load_image(img_video_or_pair)
            raise NotImplementedError("Not implemented yet")
        elif isinstance(img_video_or_pair, np.array):
            return self.swap_to_face(image=img_video_or_pair, face_name=faces, enhance_face_model=enhance_face_model)
        elif isinstance(img_video_or_pair, tuple):
            return self.swap_one_image(img_video_or_pair[0], img_video_or_pair[1], enhance_face_model)
        elif isinstance(img_video_or_pair, list):
            return [self.face2face(inp, faces, enhance_face_model, include_audio) for inp in img_video_or_pair]
        elif "videofile" in img_video_or_pair.__class__.__name__.lower():
            return self.swap_to_face_in_video(face_name=faces, target_video=img_video_or_pair, include_audio=include_audio)

        raise ValueError("Invalid input")


    def _swap_faces(
            self,
            source_faces: List[Face],
            target_faces: List[Face],
            image: np.array,
            enhance_face_model: str = 'gpen_bfr_512'
    ) -> np.array:
        """
        Changes the face(s) of the target image to the face(s) of the source image.
        if there are more target faces than source faces, the source face index is reset and starts again left->right.
        :param source_faces: the source faces from left to right [face1, None, face3, ... ]
        :param target_faces: the target faces from left to right [face1, face2, face3, ... ].
        :param image: the target image in BGR format (read with cv2)
        :param enhance_face_model: if str, the faces will be enhanced with the given face enhancer model.
            if none the faces will not be enhanced
        """
        if source_faces is None or len(source_faces) == 0:
            raise Exception("No source faces found!")

        if target_faces is None or len(target_faces) == 0:
            print(f"No face found in image. Return image as is")
            return image

        # make sure face enhance model is downloaded
        if enhance_face_model is not None:
            download_model(enhance_face_model)

        result = copy.deepcopy(image)

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
        enhance_face_model: Union[str, None] = 'gpen_bfr_512'
    ) -> np.array:
        """
        Changes the face(s) of the target image to the face(s) of the source image.
        :param source_image: the source image in BGR format (read with cv2)
        :param target_image: the target image in BGR format (read with cv2)
        :param enhance_face_model: if str, the faces will be enhanced with the given face enhancer model.
            if none the faces will not be enhanced
        """
        source_image = load_image(source_image)
        target_image = load_image(target_image)

        # get the bounding box of the faces
        source_faces = self.detect_faces(source_image)

        if source_faces is None:
            raise Exception("No source faces found!")

        target_faces = self.detect_faces(target_image)
        return self._swap_faces(source_faces, target_faces, target_image, enhance_face_model)

    def detect_faces(self, image: Union[np.array, str]) -> Union[List | None]:
        """
        get faces from left to right by order
        """
        image = load_image(image)

        try:
            face = self._face_analyser.get(image)
            return sorted(face, key=lambda x: x.bbox[0])
        except IndexError:
            return None

    def swap_to_face(
            self,
            face_name: str,
            image: Union[np.array, list],
            enhance_face_model: Union[str, None] = 'gpen_bfr_2048'
        ) -> np.array:
        """
        Changes the face(s) of the target image to the face of the reference image.
        :param face_name: the name of the reference face
        :param image: the target image in BGR format (read with cv2). Can be a list of images
        :param enhance_face_model: if str, the faces will be enhanced with the given face enhancer model.
            if none the faces will not be enhanced
        :return: the swapped image
        """
        face_name = encode_path_safe(face_name)

        # if image is a list of images, swap all images
        if isinstance(image, list):
            return list(self.swap_to_face_generator(face_name, image))

        # swap single image
        source_faces = self.load_face(face_name)
        target_faces = self.detect_faces(image)
        return self._swap_faces(source_faces, target_faces, image, enhance_face_model)

    def swap_to_face_generator(
            self,
            face_name: str,
            image_generator,
            enhance_face_model: Union[str, None] = 'gpen_bfr_2048'
    ):
        """
        Changes the face(s) of each image in the target_img_generator to the face of the reference image.
        :param face_name: the name of the reference face
        :param image_generator: a generator that yields images in BGR format (read with cv2).
            Or a video_stream that yields (image, audio) like VideoFile().to_video_stream() in media_toolkit.
        :param enhance_face_model: if str, the faces will be enhanced with the given face enhancer model.
            if none the faces will not be enhanced
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

    def swap_pairs_generator(
            self,
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
