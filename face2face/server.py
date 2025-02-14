import os
from typing import Union

from fast_task_api import FastTaskAPI, ImageFile, JobProgress, MediaFile, VideoFile

import numpy as np


from face2face.core.face2face import Face2Face
from media_toolkit.utils.generator_wrapper import SimpleGeneratorWrapper

from face2face.settings import ALLOW_EMBEDDING_SAVE_ON_SERVER


f2f = Face2Face()
app = FastTaskAPI(
    title="Face2Face service",
    summary="Swap faces from images and videos. Create face embeddings.",
    version="0.0.8",
    contact={
        "name": "SocAIty",
            "url": "https://github.com/SocAIty",
    }
)

@app.task_endpoint("/swap_img_to_img", queue_size=100)
def swap_img_to_img(source_img: ImageFile, target_img: ImageFile, enhance_face_model: str = 'gpen_bfr_512'):
    swapped_img = f2f.swap_img_to_img(np.array(source_img), np.array(target_img), enhance_face_model=enhance_face_model)
    return ImageFile(file_name="swapped_img.png").from_np_array(swapped_img)

@app.task_endpoint("/add_face", queue_size=100)
def add_face(face_name: str, image: ImageFile = None, save: bool = ALLOW_EMBEDDING_SAVE_ON_SERVER):
    """
    Add one or multiple reference face(s) to the face swapper. This face(s) can be used for swapping in other images.

    :param face_name: The name for the reference face.
        In case you provide a list of face names, an embedding is stored for each face from left to right in the provided image.

    :param image: The image from which to extract the face(s) (can be a numpy array, file path, or ImageFile).
        If there are multiple faces in the image, an embedding will be created for each name from left to right.
        If you only provide one name, only the first face will be stored.
    :param save:
        If True, the reference face will be saved to the _face_embeddings folder for future use.
        If False, the reference face will only be stored in memory.
        Note:
    :return: a file containing the face embedding. You can send this to the swap endpoint to swap the face with this reference face.
    """
    # don't save embeddings on the server if the setting is False
    # this is useful in "multi-user" scenarios or if the server is not supposed to store any data
    save = save and ALLOW_EMBEDDING_SAVE_ON_SERVER

    face_name, face_embedding = f2f.add_face(face_name=face_name, image=image, save=save)
    bytes_io = face_embedding.to_bytes_io()
    return MediaFile(file_name=f"{face_name}.npy").from_bytesio(bytes_io)

@app.task_endpoint("/swap", queue_size=100)
def swap(
    faces: Union[str, dict, list],
    media: ImageFile, #Union[ImageFile, VideoFile] = None,
    enhance_face_model: str = 'gpen_bfr_512'
):
    swapped_media = f2f.swap(faces=faces, media=media, enhance_face_model=enhance_face_model)

    if isinstance(swapped_media, np.ndarray):
        return ImageFile(file_name="swapped.png").from_np_array(swapped_media)

    return swapped_media


@app.task_endpoint("/swap_video", queue_size=1)
def swap_video(
        job_progress: JobProgress,
        face_name: str,
        target_video: VideoFile,
        include_audio: bool = True,
        enhance_face_model: str = 'gpen_bfr_512'
    ):
    def video_stream_gen():
        # generator reads the video stream and swaps the faces frame by frame
        gen = target_video.to_video_stream(include_audio=include_audio)
        swap_gen = f2f.swap_to_face_generator(face_name, gen, enhance_face_model=enhance_face_model)
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
                if percent_total > 0.9: # this is case if the estimation of cv2 is smaller than the actual video size
                    percent_total = 0.9

            else:
                percent_total = 0.5  # arbitrary number
            job_progress.set_status(message=f"swapping frame {i}", progress=percent_total)

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
