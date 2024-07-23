from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from face2face.core.face2face import Face2Face

class _Video_Swap:
    def swap_to_face_in_video(self: Face2Face,
                              face_name: str,
                              target_video,
                              include_audio: bool = True,
                              enhance_face_model: str = 'gpen_bfr_2048'
                              ):
        """
        Swaps the face of the target video to the face of the reference image.
        :param face_name: the name of the reference face embedding
        :param target_video: the target video. Path to the file or VideoFile object
        :param include_audio: if True, the audio will be included in the output video
        """
        try:
            from media_toolkit import VideoFile
        except:
            raise ImportError("Please install socaity media_toolkit to use this function")

        if isinstance(target_video, str):
            target_video = VideoFile().from_file(target_video)

        if not isinstance(target_video, VideoFile):
            raise ValueError("target_video must be a path or a VideoFile object")

        gen = target_video.to_video_stream(include_audio=include_audio)

        new_video = VideoFile().from_video_stream(
            video_audio_stream=self.swap_to_face_generator(face_name, gen, enhance_face_model=enhance_face_model),
            frame_rate=target_video.frame_rate,
            audio_sample_rate=target_video.audio_sample_rate
        )
        return new_video

    def swap_faces_to_faces_in_video(self: Face2Face, swap_pairs: dict, target_video, include_audio: bool = True):
        """
        Swaps the reference faces in the target video.
        :param swap_pairs: a dict with the structure {source_face_name: target_face_name}
        :param target_video: the target video. Path to the file or VideoFile object
        :param include_audio: if True, the audio will be included in the output video
        """
        try:
            from media_toolkit import VideoFile
        except:
            raise ImportError("Please install socaity media_toolkit to use this function")

        if isinstance(target_video, str):
            target_video = VideoFile().from_file(target_video)

        if not isinstance(target_video, VideoFile):
            raise ValueError("target_video must be a path or a VideoFile object")

        gen = target_video.to_video_stream(include_audio=include_audio)

        swap_faces_to_faces

        #new_video = VideoFile().from_video_stream(
        #    video_audio_stream=self.swap_generator(swap_pairs, gen),
        #    frame_rate=target_video.frame_rate,
        #    audio_sample_rate=target_video.audio_sample_rate
        #)
        #return new_video


