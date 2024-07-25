# ordinary imports
from typing import List, Union
import os
import numpy as np

import insightface
import onnxruntime
from insightface.app.common import Face

from face2face.core.mixins._image_swap import _ImageSwap
from media_toolkit import media_from_file, VideoFile, ImageFile

from face2face.core.mixins._face_embedding import _FaceEmbedding
from face2face.core.mixins._face_enhance import _FaceEnhancer
from face2face.core.mixins._face_recognition import _FaceRecognition
from face2face.core.mixins._video_swap import _Video_Swap

from face2face.core.modules.storage.f2f_loader import get_face_analyser
from face2face.core.modules.utils.utils import download_model
from face2face.settings import MODELS_DIR, REF_FACES_DIR, DEVICE_ID


class Face2Face(_ImageSwap, _FaceEmbedding, _FaceRecognition, _Video_Swap, _FaceEnhancer):
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
        # they dict has structure {faces: face_embedding }
        if face_embedding_folder is None:
            face_embedding_folder = REF_FACES_DIR

        self._face_embedding_folder = face_embedding_folder
        self._face_embeddings = {}

    def swap(
        self,
        media: Union[str, np.ndarray, tuple, List[str], ImageFile, VideoFile],
        faces: Union[str, dict, list, List[Face], Face, None] = None,
        enhance_face_model: str = 'gpen_bfr_512',
        include_audio: bool = True
    ) -> Union[np.array, list, VideoFile]:
        """
        This is a unified function to swap faces in images and videos.
        :param media: the image or video to swap the faces in.
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
        if isinstance(media, tuple):
            return self.swap_img_to_img(media[0], media[1], enhance_face_model)
        if isinstance(media, list):
            return [self.swap(inp, faces, enhance_face_model, include_audio) for inp in media]

        if faces is None:
            raise ValueError("Please provide faces to swap to.")

        # convert to image or video file
        file = None
        if type(media) in [str, ImageFile, VideoFile]:
            file = media_from_file(media)
        elif isinstance(media, np.ndarray):
            file = ImageFile().from_np_array(media)

        # perform swaps
        if isinstance(file, ImageFile):
            return self.swap_image(image=file, faces=faces, enhance_face_model=enhance_face_model)
        elif isinstance(file, VideoFile):
            return self.swap_video(
                faces=faces, video=file, include_audio=include_audio
            )

        raise ValueError(f"Wrong file type {media}. Check input.")

    def detect_faces(self, image: Union[np.array, str, ImageFile]) -> Union[List | None]:
        """
        get faces from left to right by order
        """
        image = ImageFile().from_any(image).to_np_array()
        try:
            face = self._face_analyser.get(image)
            return sorted(face, key=lambda x: x.bbox[0])
        except IndexError:
            return None


