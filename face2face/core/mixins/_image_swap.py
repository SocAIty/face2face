# avoid circular dependency but provide type hints
from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Union, List, Dict

from face2face.core.modules.face_enhance.face_enhancer import enhance_face

if TYPE_CHECKING:
    from face2face.core.face2face import Face2Face

# other imports
from face2face.core.modules.utils.utils import load_image
from insightface.app.common import Face
from media_toolkit import ImageFile
import numpy as np


class _ImageSwap:
    def swap_image(
            self: Face2Face,
            image: Union[str, np.array, ImageFile],
            faces: Union[str, dict, list, List[Face], Face],
            enhance_face_model: str = 'gpen_bfr_512'
    ) -> np.array:
        """
        Swaps the faces in the image.
        :param image: the image to swap the faces in
        :param faces: defines what to swap to.
            if str -> the name of the face to swap to. All faces in the image will be swapped to this face.
            if list -> the list of face_names or face objects to swap to from left to right.
            if dict -> the swap_pairs with the structure {source_face_name: target_face_name}. Use face recognition
                to swap the source faces to the target faces.
            if Face -> the face object to swap to. All faces in the image will be swapped to this face.
        :param enhance_face_model: the face enhancement model to use. Use None for no enhancement
        """
        if isinstance(faces, dict):
            return self.swap_pairs(image=image, swap_pairs=faces, enhance_face_model=enhance_face_model)

        return self.swap_to_faces(faces=faces, image=image, enhance_face_model=enhance_face_model)

    def swap_img_to_img(
        self: Face2Face,
        source_image: Union[np.array, str, ImageFile],
        target_image: Union[np.array, str, ImageFile],
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

        return self._swap_faces(
            source_faces=source_faces, target_faces=target_faces,
            image=target_image, enhance_face_model=enhance_face_model
        )

    def swap_to_faces(
            self: Face2Face,
            faces: Union[str, list, Face, List[Face]],
            image: Union[np.array, list],
            enhance_face_model: Union[str, None] = 'gpen_bfr_2048'
        ) -> np.array:
        """
        Changes the face(s) of the target image to the face of the reference image.
        :param faces: the name of the reference face
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
        source_faces = self.load_faces(faces)
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

    def swap_to_face_generator(
            self: Face2Face,
            faces: Union[str, list, Face, List[Face]],
            image_generator,
            enhance_face_model: Union[str, None] = 'gpen_bfr_2048'
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
        source_faces = self.load_faces(faces)
        source_faces = list(source_faces.values())

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
                print(f"Error in swapping frame {i} to {faces}: {e}. Skipping image")
                if audio is not None:
                    yield target_image, audio
                    continue

                yield np.array(target_image)