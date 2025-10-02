import os
import numpy as np
from typing import Literal, Union, List

from fast_task_api import FastTaskAPI, ImageFile, JobProgress, MediaFile, VideoFile, MediaList
import fast_task_api
import media_toolkit
import face2face
from face2face.core.face2face import Face2Face
from media_toolkit.utils.generator_wrapper import SimpleGeneratorWrapper

from face2face.settings import ALLOW_EMBEDDING_SAVE_ON_SERVER

# print the fasttaskapi version, media-toolkit version and face2face version
print(f"FastTaskAPI version: {fast_task_api.__version__}")
print(f"Media-toolkit version: {media_toolkit.__version__}")
print(f"Face2Face version: {face2face.__version__}")


f2f = Face2Face()
app = FastTaskAPI(
    title="Face2Face",
    summary="Swap faces from images and videos. Create face embeddings.",
    version="1.2.4",
    contact={
        "name": "SocAIty",
        "url": "https://github.com/SocAIty/face2face",
    }
)

FACE_ENHANCE_MODELS = Literal['', 'gpen_bfr_512', 'gpen_bfr_1024', 'gpen_bfr_2048', 'gfpgan_1.4.onnx']


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
    return ImageFile(file_name="swapped_img.png").from_any(swapped_img)


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

    faces = f2f.add_face(face_name=face_name, media=image, save=save)

    if isinstance(faces, dict):
        return MediaList([
            MediaFile(file_name=f"{face_name}.npy").from_bytesio(face.to_bytes_io())
            for face_name, face in faces.items()
        ])

    if isinstance(faces, tuple):
        return MediaFile(file_name=f"{face_name}.npy").from_bytesio(faces[1].to_bytes_io())


@app.task_endpoint("/swap", queue_size=500)
def swap(
    job_progress: JobProgress,
    faces: Union[List[str], dict, ImageFile, MediaFile, MediaList],
    media: Union[ImageFile, VideoFile, MediaList],
    enhance_face_model: FACE_ENHANCE_MODELS = 'gpen_bfr_512'
):
    """
    Swap faces in an image or video.

    Args:
        faces: The face(s) to swap to. Can be:
            - str: Name of a reference face
            - dict: Swap pairs with structure {source_face_name: target_face_name}
            - list: List of face names or Face embeddings
            - ImageFile: Use face detection to create a face embedding from the image.
            - MediaFile: Single face embedding file
            - MediaList: Multiple face embeddings
        media: The image(s) or video(s) to swap faces in
        enhance_face_model: Face enhancement model to use. Defaults to 'gpen_bfr_512'

    Returns:
        Union[ImageFile, VideoFile]: The resulting media with swapped faces

    Raises:
        ValueError: If no faces are provided or media type is unsupported
    """

    media = MediaList(read_system_files=False, download_files=True).from_any(media)

    processsable_media = media.get_processable_files()
    if len(processsable_media) == 0:
        raise ValueError("No processable media files found")

    progress = 0.01
    errors = []
    swapped_media = MediaList()
    for i, media_file in enumerate(processsable_media, start=1):
        progress = 0.01 + (i / len(processsable_media)) * 0.98  # do not set to 1.0 because then runpod already finishes the job and does delete the job
        message = f"Swapping {i} of {len(processsable_media)} media files."
        if errors:
            message += f" Errors: {errors}"

        job_progress.set_status(progress=progress, message=f"Swapping {i} of {len(processsable_media)} media files")
        try:
            if isinstance(media_file, VideoFile):
                result = swap_video(job_progress=job_progress, faces=faces, target_video=media_file, enhance_face_model=enhance_face_model, include_audio=True)
                swapped_media.append(result)
            elif isinstance(media_file, ImageFile):
                result = f2f.swap(faces=faces, media=media_file, enhance_face_model=enhance_face_model)
                result = ImageFile().from_np_array(result)
                swapped_media.append(result)
        except Exception as e:
            errors_message = f"Error swapping media {i}: {e}"
            errors.append(errors_message)
            continue

    if len(swapped_media) == 0:
        job_progress.set_status(progress=0.99, message=f"Errors: {errors}")
        raise ValueError(f"Errors swapping your media: {errors}")

    return swapped_media


@app.task_endpoint("/swap_video", queue_size=10)
def swap_video(
        job_progress: JobProgress,
        faces: Union[List[str], dict, MediaFile, MediaList, ImageFile],
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

    # get the frame count and sample rate from the target video
    frame_count = None
    if hasattr(target_video, "video_info") and target_video.video_info.frame_count:
        frame_count = target_video.video_info.frame_count

    sample_rate = None
    if include_audio and hasattr(target_video, "video_info") and hasattr(target_video.video_info, "audio_info") and target_video.video_info.audio_info.sample_rate:
        sample_rate = target_video.video_info.audio_info.sample_rate

    gen = target_video.to_stream()

    def video_stream_gen():
        swap_gen = f2f.swap_to_face_generator(faces, gen, enhance_face_model=enhance_face_model)
        # Swap the images one by one
        for i, swapped_img in enumerate(swap_gen):
            # update progress. Swapping is 90% of the work
            if frame_count:
                percent_converted = float(i / frame_count)
                percent_total = round(percent_converted * 0.9, 2)
                if percent_total > 0.9:  # this is case if the estimation of cv2 is smaller than the actual video size
                    percent_total = 0.9

            else:
                percent_total = 0.5  # arbitrary number
            job_progress.set_status(progress=percent_total, message=f"swapping frame {i}")

            # Yield the image
            yield swapped_img

    # allow tqdm to show better progress bar
    streamer = SimpleGeneratorWrapper(video_stream_gen(), length=frame_count)

    # Create video
    output_video = VideoFile().from_generators(
        frame_generator=streamer,
        audio_generator=gen.audio_frames("av"),
        frame_rate=target_video.video_info.frame_rate,
        audio_sample_rate=sample_rate
    )

    return output_video


@app.task_endpoint("/enhance_face", queue_size=500)
def enhance_face(
    face_image: Union[ImageFile, MediaList],
    enhance_face_model: FACE_ENHANCE_MODELS = 'gpen_bfr_512'
):
    """
    Enhance a face image.
    """
    if not isinstance(face_image, MediaList):
        face_image = MediaList([face_image], read_system_files=False, download_files=True)

    results = MediaList()
    for i, face in enumerate(face_image):
        try:
            enhanced_faces = f2f.enhance_faces([face], enhance_face_model=enhance_face_model)
            result = ImageFile().from_np_array(enhanced_faces)
            results.append(result)
        except Exception as e:
            print(f"Error enhancing face {i}: {e}")
            continue
    
    return results


# start the server on provided port
if __name__ == "__main__":
    # setting port to 8020 (referenced as so in socaity sdk for face2face) if not specified differently
    port = int(os.environ.get("FTAPI_PORT", 8020))
    app.start(port=port)
