import os
from typing import Literal, Union, List

from fast_task_api import FastTaskAPI, ImageFile, JobProgress, MediaFile, VideoFile, MediaList, MediaDict

import numpy as np


from face2face.core.face2face import Face2Face
from media_toolkit.utils.generator_wrapper import SimpleGeneratorWrapper

from face2face.settings import ALLOW_EMBEDDING_SAVE_ON_SERVER


f2f = Face2Face()
app = FastTaskAPI(
    title="Face2Face",
    summary="Swap faces from images and videos. Create face embeddings.",
    version="0.0.8",
    contact={
        "name": "SocAIty",
        "url": "https://github.com/SocAIty/face2face",
    }
)

FACE_ENHANCE_MODELS = Literal['', 'gpen_bfr_512', 'gpen_bfr_1024', 'gpen_bfr_2048']


@app.task_endpoint("/swap_img_to_img", queue_size=500)
def swap_img_to_img(source_img: ImageFile, target_img: ImageFile, enhance_face_model: FACE_ENHANCE_MODELS = 'gpen_bfr_512'):
    """
    Swap faces between two images.

    Args:
        source_img: Source image containing the face(s) to swap from
        target_img: Target image containing the face(s) to swap to
        enhance_face_model: Face enhancement model to use. Defaults to 'gpen_bfr_512'

    Returns:
        ImageFile: The resulting image with swapped faces
    """
    swapped_img = f2f.swap_img_to_img(np.array(source_img), np.array(target_img), enhance_face_model=enhance_face_model)
    return ImageFile(file_name="swapped_img.png").from_np_array(swapped_img)


@app.task_endpoint("/add_face", queue_size=500)
def add_face(
    face_name: Union[str, List[str]],
    image: ImageFile,
    save: bool = ALLOW_EMBEDDING_SAVE_ON_SERVER
):
    """
    Add one or multiple reference face(s) to the face swapper.
    
    Args:
        face_name: Name(s) for the reference face(s).
            - If a single string, creates one face embedding
            - If a list of strings, creates embeddings for each face from left to right in the image
        image: The image from which to extract the face(s).
            - ImageFile: Standard image file
        save: Whether to save the face embeddings to disk.
            Note: This is controlled by ALLOW_EMBEDDING_SAVE_ON_SERVER setting
    
    Returns:
        Union[MediaFile, MediaDict]:
            - For single face: MediaFile containing the face embedding
            - For multiple faces: MediaDict mapping face names to their embeddings
    
    Raises:
        ValueError: If no face name is provided or no faces are detected in the image
    """
    # don't save embeddings on the server if the setting is False
    # this is useful in "multi-user" scenarios or if the server is not supposed to store any data
    save = save and ALLOW_EMBEDDING_SAVE_ON_SERVER

    faces = f2f.add_face(face_name=face_name, image=image, save=save)

    if isinstance(faces, dict):
        return MediaDict({
            face_name: MediaFile(file_name=f"{face_name}.npy").from_bytesio(face.to_bytes_io())
            for face_name, face in faces.items()
        })

    if isinstance(faces, tuple):
        return MediaFile(file_name=f"{face_name}.npy").from_bytesio(faces[1].to_bytes_io())


@app.task_endpoint("/swap", queue_size=500)
def swap(
    job_progress: JobProgress,
    faces: Union[List[str], dict, MediaFile, MediaList],
    media: Union[ImageFile, VideoFile],
    enhance_face_model: FACE_ENHANCE_MODELS = 'gpen_bfr_512'
):
    """
    Swap faces in an image or video.

    Args:
        faces: The face(s) to swap to. Can be:
            - str: Name of a reference face
            - dict: Swap pairs with structure {source_face_name: target_face_name}
            - list: List of face names or Face embeddings
            - MediaFile: Single face embedding file
            - MediaList: Multiple face embedding files
        media: The image or video to swap faces in
        enhance_face_model: Face enhancement model to use. Defaults to 'gpen_bfr_512'

    Returns:
        Union[ImageFile, VideoFile]: The resulting media with swapped faces

    Raises:
        ValueError: If no faces are provided or media type is unsupported
    """
    if isinstance(media, VideoFile):
        return swap_video(job_progress=job_progress, faces=faces, target_video=media, enhance_face_model=enhance_face_model, include_audio=True)

    job_progress.set_status(progress=0.01, message="Started swapping faces")
    swapped_media = f2f.swap(faces=faces, media=media, enhance_face_model=enhance_face_model)

    if isinstance(swapped_media, np.ndarray):
        return ImageFile(file_name="swapped.png").from_np_array(swapped_media)

    return swapped_media


@app.task_endpoint("/swap_video", queue_size=10)
def swap_video(
        job_progress: JobProgress,
        faces: Union[List[str], dict, MediaFile, MediaList],
        target_video: VideoFile,
        include_audio: bool = True,
        enhance_face_model: FACE_ENHANCE_MODELS = 'gpen_bfr_512'
):
    """
    Swap faces in a video file.
    
    Args:
        face_name: The face(s) to swap to. Can be:
            - str: Name of a reference face
            - list: List of face names or Face objects
            - MediaFile: Single face embedding file
            - MediaList: Multiple face embedding files
        target_video: The video to swap faces in
        include_audio: Whether to include audio in the output video
        enhance_face_model: Face enhancement model to use. Defaults to 'gpen_bfr_512'

    Returns:
        VideoFile: The resulting video with swapped faces

    Raises:
        ValueError: If no faces are provided or video cannot be processed
    """
    def video_stream_gen():
        # generator reads the video stream and swaps the faces frame by frame
        gen = target_video.to_video_stream(include_audio=include_audio)
        swap_gen = f2f.swap_to_face_generator(faces, gen, enhance_face_model=enhance_face_model)
        # Swap the images one by one
        for i, swapped_audio_tuple in enumerate(swap_gen):
            audio = None
            if include_audio and len(swapped_audio_tuple) == 2:
                swapped_img, audio = swapped_audio_tuple
            else:
                swapped_img = swapped_audio_tuple

            # update progress. Swapping is 90% of the work
            if target_video.frame_count:
                percent_converted = float(i / target_video.frame_count)
                percent_total = round(percent_converted * 0.9, 2)
                if percent_total > 0.9:  # this is case if the estimation of cv2 is smaller than the actual video size
                    percent_total = 0.9

            else:
                percent_total = 0.5  # arbitrary number
            job_progress.set_status(progress=percent_total, message=f"swapping frame {i}")

            # Yield the image
            if include_audio and audio is not None:
                yield swapped_img, audio
            else:
                yield swapped_img

    # allow tqdm to show better progress bar
    streamer = SimpleGeneratorWrapper(video_stream_gen(), length=target_video.frame_count)

    # Create video
    output_video = VideoFile().from_video_stream(
        video_audio_stream=streamer,
        frame_rate=target_video.frame_rate,
        audio_sample_rate=target_video.audio_sample_rate
    )
    return output_video


# start the server on provided port
if __name__ == "__main__":
    # setting port to 8020 (referenced as so in socaity sdk for face2face) if not specified differently
    port = int(os.environ.get("FTAPI_PORT", 8020))
    app.start(port=port)
