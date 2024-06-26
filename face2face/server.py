import argparse
import fastapi

from fast_task_api import FastTaskAPI, ImageFile, JobProgress, MediaFile, VideoFile

import numpy as np

from face2face.settings import PORT, PROVIDER
from face2face.core.face2face import Face2Face
f2f = Face2Face()


#f2f = Face2Face()
app = FastTaskAPI(
    provider=PROVIDER,
    app=fastapi.FastAPI(
        title="Face2Face service",
        summary="Swap faces from images and videos. Create face embeddings.",
        version="0.0.2",
        contact={
            "name": "SocAIty",
            "url": "https://github.com/SocAIty",
        }
    ),
)

@app.task_endpoint("/swap_one", queue_size=100)
def swap_one(source_img: ImageFile, target_img: ImageFile):
    swapped_img = f2f.swap_one_image(np.array(source_img), np.array(target_img))
    return ImageFile(file_name="swapped_img.png").from_np_array(swapped_img)

@app.task_endpoint("/add_reference_face", queue_size=100)
def add_reference_face(face_name: str, source_img: ImageFile = None, save: bool = True):
    face_name, face_embedding = f2f.add_reference_face(face_name, np.array(source_img), save=save)
    return MediaFile(file_name=f"{face_name}.npz").from_bytesio(face_embedding)

@app.task_endpoint("/swap_from_reference_face", queue_size=100)
def swap_from_reference_face(face_name: str, target_img: ImageFile = None):
    swapped_img = f2f.swap_from_reference_face(face_name, np.array(target_img))
    return ImageFile(file_name=f"swapped_to_{face_name}.png").from_np_array(swapped_img)


def _swapped_video_stream_gen(face_name: str, target_video: VideoFile):
    for image, audio in target_video.to_video_stream():
        # Swap
        try:
            swapped_img = f2f.swap_from_reference_face(face_name=face_name, target_image=np.array(image))
        except Exception as e:
            print(f"Error in swapping to {face_name}: {e} ")
            swapped_img = np.array(image)

        yield swapped_img, audio

@app.task_endpoint("/swap_video_from_reference_face", queue_size=1)
def swap_video_from_reference_face(job_progress: JobProgress, face_name: str, target_video: VideoFile):
    # generator reads the video stream and swaps the faces frame by frame
    def video_stream_gen():
        # Swap the images one by one
        for i, (swapped_img, audio) in enumerate(_swapped_video_stream_gen(face_name, target_video)):
            # update progress. Swapping is 90% of the work
            percent_converted = float(i / target_video.frame_count)
            percent_total = round(percent_converted * 0.9, 2)
            job_progress.set_status(message=f"swapping frame {i}", progress=percent_total)
            # Yield the image
            yield swapped_img, audio

    # Create video
    output_video = VideoFile().from_video_stream(
        video_audio_stream=video_stream_gen(),
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
