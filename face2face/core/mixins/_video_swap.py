# avoid circular dependency but provide type hints
from __future__ import annotations
from typing import TYPE_CHECKING, Union

from face2face.core.compatibility.Face import Face
from media_toolkit import VideoFile

from tqdm import tqdm

if TYPE_CHECKING:
    from face2face.core.face2face import Face2Face
    from media_toolkit import MediaFile, MediaList, ImageFile

types_faces = Union[str, Face, list, 'ImageFile', 'MediaFile', 'MediaList']


class _Video_Swap:
    def swap_video(
        self: Face2Face,
        video: Union[str, VideoFile],
        faces: types_faces,
        enhance_face_model: Union[str, None] = None,
        include_audio: bool = True
    ) -> VideoFile:
        """
        Swaps the faces in the video.
        :param video: the video to swap the faces in
        :param faces: the faces to swap in the video
        :param enhance_face_model: the face enhancement model to use. Use None for no enhancement
        :param include_audio: if True, the audio will be included in the output video
        """
        video = VideoFile().from_any(video)
        if not isinstance(faces, Face) and isinstance(faces, dict):
            return self.swap_pairs_in_video(
                swap_pairs=faces, video=video, enhance_face_model=enhance_face_model, include_audio=include_audio
            )
    
        return self.swap_to_face_in_video(
            faces=faces, video=video, enhance_face_model=enhance_face_model, include_audio=include_audio
        )

    def swap_to_face_in_video(
        self: Face2Face,
        faces: types_faces,
        video: Union[str, VideoFile],
        include_audio: bool = True,
        enhance_face_model: Union[str, None] = None
    ) -> VideoFile:
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

        video_stream = video.to_stream()
        if video.video_info.frame_count is not None:
            gen = tqdm(video_stream, total=video.video_info.frame_count, desc="Video face swap")

        swap_gen = self.swap_to_face_generator(faces=faces, image_generator=gen, enhance_face_model=enhance_face_model)

        # allow tqdm to show better progress bar
        audio_type = None
        audio_sample_rate = None
        if video.video_info.audio_info:
            audio_type = video.video_info.audio_info.codec_name
            audio_sample_rate = video.video_info.audio_info.sample_rate

        new_video = VideoFile().from_generators(
            frame_generator=swap_gen,
            audio_generator=video_stream.audio_frames(output_format="av") if include_audio else None,
            frame_rate=video.video_info.frame_rate,
            px_fmt=video.video_info.pix_fmt,
            audio_output_format=audio_type,
            audio_sample_rate=audio_sample_rate
        )
        return new_video

    def swap_pairs_in_video(
        self: Face2Face,
        swap_pairs: dict,
        video: Union[str, VideoFile],
        include_audio: bool = True,
        enhance_face_model: Union[str, None] = None,
        recognition_threshold: float = 0.5
    ) -> VideoFile:
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

        video_stream = video.to_stream()
        if video.video_info.frame_count is not None:
            gen = tqdm(video_stream, total=video.video_info.frame_count, desc="Video face swap with recognition")

        swapper_gen = self.swap_pairs_generator(
            swap_pairs=swap_pairs,
            image_generator=gen,
            enhance_face_model=enhance_face_model,
            recognition_threshold=recognition_threshold
        )

        audio_type = None
        audio_sample_rate = None
        if video.video_info.audio_info:
            audio_type = video.video_info.audio_info.codec_name
            audio_sample_rate = video.video_info.audio_info.sample_rate

        new_video = VideoFile().from_generators(
            frame_generator=swapper_gen,
            audio_generator=video_stream.audio_frames(output_format="av") if include_audio else None,
            frame_rate=video.video_info.frame_rate,
            px_fmt=video.video_info.pix_fmt,
            audio_output_format=audio_type,
            audio_sample_rate=audio_sample_rate
        )
        return new_video
