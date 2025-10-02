# avoid circular dependency but provide type hints
from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Union, List

from face2face.core.compatibility.Face import Face
from face2face.core.modules.face_enhance.face_enhancer import enhance_face

from media_toolkit import ImageFile

if TYPE_CHECKING:
    from face2face.core.face2face import Face2Face
    from media_toolkit import MediaFile, MediaList

# other imports
from face2face.core.modules.utils import load_image, download_model
import numpy as np

types_faces = Union[str, Face, list, 'ImageFile', 'MediaFile', 'MediaList']


class _ImageSwap:
    def swap_img_to_img(
        self: Face2Face,
        source_image: Union[np.array, str, ImageFile],
        target_image: Union[np.array, str, ImageFile],
        enhance_face_model: Union[str, None] = 'gpen_bfr_512'
    ) -> ImageFile:
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

        return self._swap_faces(
            source_faces=source_faces, target_faces=target_faces,
            image=target_image, enhance_face_model=enhance_face_model
        )

    def swap_to_faces(
            self: Face2Face,
            faces: types_faces,
            image: Union[np.array, list, ImageFile],
            enhance_face_model: Union[str, None] = 'gpen_bfr_2048'
    ) -> ImageFile:
        """
        Changes the face(s) of the target image to the face of the reference image.
        :param faces: the name of the reference face(s), reference face(s), image file or list of image files
        :param image: the target image in BGR format (read with cv2). Can be a list of images
        :param enhance_face_model: if str, the faces will be enhanced with the given face enhancer model.
            if none the faces will not be enhanced
        :return: the swapped image
        """
        # if image is a list of images, swap all images
        if isinstance(image, list):
            gen = self.swap_to_face_generator(
                faces=faces,
                image_generator=image,
                enhance_face_model=enhance_face_model
            )
            return list(gen)

        # swap single image
        source_faces = self.get_faces(faces)
        source_faces = list(source_faces.values())
        target_faces = self.detect_faces(image)
        return self._swap_faces(
            source_faces=source_faces, target_faces=target_faces,
            image=image, enhance_face_model=enhance_face_model
        )

    def _swap_faces(
        self: Face2Face,
        source_faces: List[Face],
        target_faces: List[Face],
        image: Union[np.array, str, ImageFile],
        enhance_face_model: str = 'gpen_bfr_512',
    ) -> ImageFile:
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
            print("No face found in image. Return image as is")
            return image

        # make sure it is a numpy array
        image = load_image(image)
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
            if enhance_face_model is not None and isinstance(enhance_face_model, str) and len(enhance_face_model) > 0:
                try:
                    download_model(enhance_face_model)  # make sure face enhance model is downloaded
                    result = enhance_face(
                        target_face=target_faces[target_index], temp_vision_frame=result, model=enhance_face_model
                    )
                except Exception as e:
                    print(f"Error in enhancing face {target_index}: {e}. Returning lowres swap instead.")

        return np.array(result)

    def swap_to_face_generator(
        self: Face2Face,
        faces: types_faces,
        image_generator,
        enhance_face_model: Union[str, None] = 'gpen_bfr_2048',
    ):
        """
        Changes the face(s) of each image in the target_img_generator to the face of the reference image.
        :param faces: the name of the reference face
        :param image_generator: a generator that yields images in BGR format (read with cv2).
            Or a video_stream that yields (image, audio) like VideoFile().to_video_stream() in media_toolkit.
        :param enhance_face_model: if str, the faces will be enhanced with the given face enhancer model.
            if none the faces will not be enhanced
        :return: a generator that yields the swapped images or tuples (image, audio)
        """
        source_faces = self.get_faces(faces)
        source_faces = list(source_faces.values())

        if len(source_faces) == 0:
            print("No source faces found. Returning image_generator as is.")
            return image_generator

        for i, target_image in enumerate(image_generator):
            try:
                target_faces = self.detect_faces(target_image)
                swapped = self._swap_faces(source_faces, target_faces, target_image, enhance_face_model)
                yield swapped
            except Exception as e:
                print(f"Error in swapping frame {i} to {faces}: {e}. Skipping image")
                frm = np.array(target_image)
                yield frm
