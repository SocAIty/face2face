# avoid circular dependency but provide type hints
from __future__ import annotations
from typing import TYPE_CHECKING, Union, List, Dict
if TYPE_CHECKING:
    from face2face.core.face2face import Face2Face

# other imports
from insightface.app.common import Face
from media_toolkit import VideoFile




class _Video_Swap:

    def swap_video(
            self: Face2Face,
            video: Union[str, VideoFile],
            faces: Union[str, dict, list, List[Face], Face],
            enhance_face_model: str = 'gpen_bfr_2048',
            include_audio: bool = True
    ):
        """
        Swaps the faces in the video.
        :param video: the video to swap the faces in
        :param faces: the faces to swap in the video
        :param enhance_face_model: the face enhancement model to use. Use None for no enhancement
        :param include_audio: if True, the audio will be included in the output video
        """
        video = VideoFile().from_any(video)
        if isinstance(faces, dict):
            return self.swap_pairs_in_video(
                swap_pairs=faces, video=video, enhance_face_model=enhance_face_model, include_audio=include_audio
            )
        elif type(faces) in [list, str, Face]:
            return self.swap_to_face_in_video(
                faces=faces, video=video, enhance_face_model=enhance_face_model, include_audio=include_audio
            )

        raise NotImplementedError

    def swap_to_face_in_video(
            self: Face2Face,
            faces: str,
            video: Union[str, VideoFile],
            include_audio: bool = True,
            enhance_face_model: str = 'gpen_bfr_2048'
        ):
        """
        Swaps the face of the target video to the face of the reference image.
        :param face_name: the name of the reference face embedding
        :param video: the target video. Path to the file or VideoFile object
        :param include_audio: if True, the audio will be included in the output video
        :param enhance_face_model: the face enhancement model to use. Use None for no enhancement
        """
        video = VideoFile().from_any(video)
        if not isinstance(video, VideoFile):
            raise ValueError("Video must be a path or a VideoFile object")

        gen = video.to_video_stream(include_audio=include_audio)
        swap_gen = self.swap_to_face_generator(faces=faces, image_generator=gen, enhance_face_model=enhance_face_model)

        new_video = VideoFile().from_video_stream(
            video_audio_stream=swap_gen,
            frame_rate=video.frame_rate,
            audio_sample_rate=video.audio_sample_rate
        )
        return new_video

    def swap_pairs_in_video(
            self: Face2Face,
            swap_pairs: dict,
            video: Union[str, VideoFile],
            include_audio: bool = True,
            enhance_face_model: Union[str, None] = 'gpen_bfr_256',
            recognition_threshold: float = 0.5
    ):
        """
        Swaps the reference faces in the target video.
        :param swap_pairs: a dict with the structure {source_face_name: target_face_name}
        :param video: the target video. Path to the file or VideoFile object
        :param enhance_face_model: the face enhancement model to use. Use None for no enhancement
        :param recognition_threshold: the threshold for face-recognition. Lower value -> more false positives
        :param include_audio: if True, the audio will be included in the output video
        """
        video = VideoFile().from_any(video)
        if not isinstance(video, VideoFile):
            raise ValueError("Video must be a path or a VideoFile object")

        gen = video.to_video_stream(include_audio=include_audio)

        swapper_gen = self.swap_pairs_generator(
                swap_pairs=swap_pairs,
                image_generator=gen,
                enhance_face_model=enhance_face_model,
                recognition_threshold=recognition_threshold
        )

        new_video = VideoFile().from_video_stream(
            video_audio_stream=swapper_gen,
            frame_rate=video.frame_rate,
            audio_sample_rate=video.audio_sample_rate
        )
        return new_video
