# avoid circular dependency but provide type hints
from __future__ import annotations
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from face2face.core.face2face import Face2Face
# other imports
import numpy as np
from insightface.app.common import Face
from media_toolkit import ImageFile
from face2face.core.modules.face_enhance.face_enhancer import enhance_face
from face2face.core.modules.utils.utils import load_image, download_model


class _FaceEnhancer:
    def enhance_faces(self: Face2Face, image: Union[str, np.array, ImageFile], model='gfpgan_1.4'):
        """
        Method detects faces in the image and enhances them with the provided model.
        """
        image = load_image(image)
        # make sure face enhance model is downloaded
        download_model(model)

        faces = self.detect_faces(image)
        for face in faces:
            image = enhance_face(target_face=face, temp_vision_frame=image, model=model)
        return image

    @staticmethod
    def enhance_single_face(image: Union[str, np.array, ImageFile], target_face: Face, model='gfpgan_1.4'):
        """
        Method enhances a single face in the image with the provided model.
        To get faces use self.detect_faces(image)
        """
        image = load_image(image)
        # make sure face enhance model is downloaded
        download_model(model)

        return enhance_face(target_face=target_face, temp_vision_frame=image, model=model)