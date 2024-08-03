import argparse
from typing import Union

from fast_task_api import FastTaskAPI, ImageFile, JobProgress, MediaFile, VideoFile

import numpy as np

from face2face.settings import PORT
from face2face.core.face2face import Face2Face
from media_toolkit.utils.generator_wrapper import SimpleGeneratorWrapper

try:
    import fastapi
    fapi_app = fastapi.FastAPI(
        title="Face2Face service",
        summary="Swap faces from images and videos. Create face embeddings.",
        version="0.0.6",
        contact={
            "name": "SocAIty",
            "url": "https://github.com/SocAIty",
        }
    )
except ImportError:
    fapi_app = None

f2f = Face2Face()
app = FastTaskAPI(app=fapi_app)

@app.task_endpoint("/swap_img_to_img", queue_size=100)
def swap_img_to_img(source_img: ImageFile, target_img: ImageFile, enhance_face_model: str = 'gpen_bfr_512'):
    swapped_img = f2f.swap_img_to_img(np.array(source_img), np.array(target_img), enhance_face_model=enhance_face_model)
    return ImageFile(file_name="swapped_img.png").from_np_array(swapped_img)

@app.task_endpoint("/add_face", queue_size=100)
def add_face(face_name: str, image: ImageFile = None, save: bool = True):
    face_name, face_embedding = f2f.add_face(face_name=face_name, image=image, save=save)
    return MediaFile(file_name=f"{face_name}.npz").from_bytesio(face_embedding)

@app.task_endpoint("/swap", queue_size=100)
def swap(
    faces: Union[str, dict, list],
    media: ImageFile, #Union[ImageFile, VideoFile] = None,
    enhance_face_model: str = 'gpen_bfr_512'
):
    swapped_media = f2f.swap(faces=faces, media=media, enhance_face_model=enhance_face_model)

    if isinstance(swapped_media, np.ndarray):
        return ImageFile(file_name="swapped.png").from_np_array(swapped_media)
    #elif isinstance(swapped_media, list):
    #    return [ImageFile(file_name="swapped.png").from_np_array(img) for img in swapped_media]
    # ToDo: re-add progress bar for video files. But maybe more under the hood
    #elif isinstance(swapped_media, VideoFile):
    #    return VideoFile

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


def start_server(port: int = PORT):
    app.start(port=port)

# start the server on provided port
if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--port", type=int, default=PORT)
    args = arg_parser.parse_args()
    start_server(port=args.port)
