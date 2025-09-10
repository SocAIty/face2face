# ordinary imports
from typing import List, Union
import numpy as np

import onnxruntime


from face2face.core.compatibility.Face import Face
from face2face.core.compatibility.FaceAnalysis import FaceAnalysis
from face2face.core.compatibility.INSwapper import INSwapper
from face2face.core.mixins._image_swap import _ImageSwap
from media_toolkit import VideoFile, ImageFile, MediaList

from face2face.core.mixins._face_embedding import _FaceEmbedding
from face2face.core.mixins._face_enhance import _FaceEnhancer
from face2face.core.mixins._face_recognition import _FaceRecognition
from face2face.core.mixins._video_swap import _Video_Swap

from face2face.core.modules.utils import load_image, download_model
from face2face.settings import EMBEDDINGS_DIR, DEVICE_ID


class Face2Face(_ImageSwap, _FaceEmbedding, _FaceRecognition, _Video_Swap, _FaceEnhancer):
    def __init__(self, face_embedding_folder: str = None, device_id: int = None):
        """
        :param model_path: the folder where the models are stored and downloaded to.
            results in structure like models/insightface/inswapper_128.onnx model
            and models/face_enhancer/gfpgan_1.4.onnx
        :param inswapper_model_name:
        """
        # download inswapper model (roop) if not existing and insightface models
        swapper_model_file_path = download_model("inswapper_128")
        face_analyiser_models_path = download_model("buffalo_l")

        self.providers = onnxruntime.get_available_providers()
        # Setting GPU number and creating onnx session
        if device_id is None:
            device_id = DEVICE_ID

        if "CUDAExecutionProvider" in self.providers:
            self.providers.remove("CUDAExecutionProvider")
            self.providers.append(("CUDAExecutionProvider", {'device_id': device_id}))
            self.providers = [("CUDAExecutionProvider", {'device_id': device_id})]

        self._face_analyser = FaceAnalysis(model_dir=face_analyiser_models_path, providers=self.providers)
        self._face_analyser.prepare(ctx_id=0, det_size=(320, 320))

        self._face_swapper = INSwapper(model_file=swapper_model_file_path, providers=self.providers)

        # face swapper has the option to swap images from previously stored faces as embeddings
        # they dict has structure {faces: face_embedding }
        if face_embedding_folder is None:
            face_embedding_folder = EMBEDDINGS_DIR

        self._face_embedding_folder = face_embedding_folder
        self._face_embeddings = {}

    def swap(
        self,
        media: Union[str, np.ndarray, tuple, List[str], ImageFile, VideoFile, List[ImageFile], MediaList],
        faces: Union[str, dict, list, List[Face], Face, None] = None,
        enhance_face_model: Union[str, None] = 'gpen_bfr_512',
        include_audio: bool = True
    ) -> Union[np.array, list, VideoFile]:
        """
        Perform a unified face swap operation on images, videos, or streams.
        Parameters
        ----------
        media : The input media where faces should be swapped. Supported formats:
            - str: Path or URL to an image or video file. The media will be loaded automatically.
            - np.ndarray: Image array (H, W, C) in RGB or BGR format.
            - tuple:
                * (image, image): Equivalent to swap_img_to_img.
                * (image, video): Equivalent to swap_img_to_video(image, video).
                * (video, image): Equivalent to swap_img_to_video(image, video).
            - list[str]: A list of file paths or URLs; swaps will be applied to each item.
            - ImageFile: Preloaded image object.
            - VideoFile: Preloaded video object.

        faces : The face(s) in the media will be swapped to the provided face(s). Variants:
            - str:
                * Path/URL to an image → faces are detected from the image and used as swap targets.
                * Path/URL to an face embedding file (.npy, etc.).
                * Registered face name → all detected faces in the media are swapped to this identity.
            - list:
                * List[Face]: Multiple Face embeddings.
                * List[str]: List of paths, URLs, registered names, or Face objects. → swaps are applied in order (left-to-right).
            - dict:
                * Mapping {source_face: face_embedding} → uses recognition to map specific source → target swaps.
            - Face: A single Face embedding.
            - list[Face]: Multiple Face embeddings.

        enhance_face_model : str | None, default="gpen_bfr_512"
            Optional face enhancement model to apply post-swap (e.g. GPEN, CodeFormer).
            Set to None to disable enhancement.

        include_audio : bool, default=True
            If True, preserves audio when processing videos. Ignored for images.

        Returns
        -------
        np.ndarray | list[np.ndarray] | VideoFile
            - Single image (np.ndarray) if input was an image.
            - List of images (list[np.ndarray]) if input was a list of files or a generator.
            - VideoFile object if input was a video.

        Notes
        -----
        - Supports both static (images) and dynamic (video/stream) swapping.
        - If multiple faces are present, swaps are applied in recognition order left to right.
        - Enhancement is applied after swapping, per frame in videos.
        """

        if media is None:
            raise ValueError("Please provide media to swap.")
        
        # TODO: the fucking faces in dict is a shit with swapping by pairs (with recognition)
        # Needs to be fixed.
        if faces is not None and not isinstance(faces, dict):
            faces = self.get_faces(faces)

        # read all the provided media
        media = MediaList(read_system_files=True, download_files=True).from_any(media)

        if len(media) == 0:
            raise ValueError("Please provide media to swap.")

        # DECIDE WHICH METHOD TO USE FOR FACE SWAPPING
        # media pair based swaps
        if len(media) == 2 and faces is None:
            if isinstance(media[0], ImageFile) and isinstance(media[1], ImageFile):
                return self.swap_img_to_img(media[0], media[1], enhance_face_model)
            elif isinstance(media[0], ImageFile) and isinstance(media[1], VideoFile):
                return self.swap_video(media[0], media[1], enhance_face_model)
            elif isinstance(media[0], VideoFile) and isinstance(media[1], ImageFile):
                return self.swap_video(media[1], media[0], enhance_face_model)

        # face based swaps
        if faces is None:
            raise ValueError("Please provide faces to swap to.")
        
        if len(media) >= 2:
            return [self.swap(inp, faces, enhance_face_model, include_audio) for inp in media]

        if isinstance(media[0], ImageFile):
            if not isinstance(faces, Face) and isinstance(faces, dict):
                return self.swap_pairs(media[0], faces, enhance_face_model)
            return self.swap_image(media[0], faces, enhance_face_model)
        elif isinstance(media[0], VideoFile):
            if not isinstance(faces, Face) and isinstance(faces, dict):
                return self.swap_pairs_in_video(faces, media[0], enhance_face_model, include_audio)
            return self.swap_video(media[0], faces, enhance_face_model, include_audio)

        raise ValueError(f"Wrong file type {media}. Check input.")

    def detect_faces(self, image: Union[np.array, str, ImageFile]) -> Union[List | None]:
        """
        get faces from left to right by order
        """
        image = load_image(image)
        try:
            face = self._face_analyser.get(image)
            return sorted(face, key=lambda x: x.bbox[0])
        except IndexError:
            return None
