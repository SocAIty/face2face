import os

import cv2
from tqdm import tqdm

from face2face.settings import OUTPUT_DIR, MODELS_DIR
from face2face.utils import get_files_in_dir


def make_video_from_images(
    image_paths: list[str], outpath: str = None, frame_rate: int = 60
) -> None:
    """creates a video from a list of images"""

    if outpath is None:
        outpath = os.path.join(OUTPUT_DIR, "output.mp4")

    # get image dimensions
    firstimg = cv2.imread(image_paths[0])
    height, width, layers = firstimg.shape

    # create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(outpath, fourcc, frame_rate, (width, height))

    # write images to video
    for image_path in tqdm(image_paths, desc="writing images to video"):
        image = cv2.imread(image_path)
        video.write(image)

    # close video writer
    video.release()


def make_video_from_image_folder(
    image_folder: str, outpath: str = None, frame_rate: int = 60
) -> None:
    """creates a video from a folder of images"""
    image_paths = get_files_in_dir(image_folder, [".jpeg", ".jpg", ".png"])
    # order images by creation date otherwise the video will be out of order
    image_paths.sort(key=os.path.getmtime)
    make_video_from_images(image_paths, outpath, frame_rate)


def video2images(video_path: str, outpath: str = None) -> None:
    """creates a video from a list of images
    video_path: path to video file in mp4
    outpath: folder to save the images (no trailing /)
    """

    if outpath is None:
        outpath = OUTPUT_DIR

    if not os.path.isdir(outpath):
        os.makedirs(outpath)

    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0
    while success:
        cv2.imwrite(outpath + "/frame%d.jpg" % count, image)  # save frame as JPEG file
        success, image = vidcap.read()
        print('Read a new frame: ', success)
        count += 1

    # write framerate to file
    framerate = vidcap.get(cv2.CAP_PROP_FPS)
    with open(os.path.join(outpath, "_framerate.txt"), "w") as f:
        f.write(str(framerate))


def extract_audio_from_video(video_path: str, outpath: str = None) -> None:
    """extracts audio from a video
    video_path: path to video file in mp4
    outpath: path to save the audio file
    """

    if outpath is None:
        outpath = OUTPUT_DIR

    if not os.path.isdir(outpath):
        os.makedirs(outpath)

    # ffmpeg -i input.mp4 -vn -acodec copy output-audio.aac
    # command = "ffmpeg -i C:/test.mp4 -ab 160k -ac 2 -ar 44100 -vn audio.wav"
    # subprocess.call(command, shell=True)
    os.system(f"ffmpeg -i {video_path} -vn -acodec copy {outpath}/audio.wav")


def upscale_images_in_folder(image_folder: str, outpath: str = None):
    print(f"upscaling images in {image_folder}")
    path_to_real_esrgan = (
        MODELS_DIR + "/upscaling/realesrgan-ncnn-vulkan/realesrgan-ncnn-vulkan.exe"
    )

    # get imags
    lowres_imgs = get_files_in_dir(image_folder, [".jpeg", ".jpg", ".png"])
    # order images by creation date otherwise the video will be out of order
    lowres_imgs.sort(key=os.path.getmtime)

    # create output dir
    if outpath is None:
        outpath = os.path.join(image_folder, "upscaled")
        if not os.path.isdir(outpath):
            os.makedirs(outpath)

    # convert images with realesrgan
    for limg in tqdm(lowres_imgs):
        out_img = outpath + "/" + os.path.basename(limg)
        cmd = f"{path_to_real_esrgan} -i {limg} -o {out_img}"
        os.system(cmd)


def upscale_video(video_path: str, outpath: str = None):
    """
    Uses ESRGAN to upscale video. The audio is reaplied to the upscaled video.
    """
    # video2images(video_path, outpath)
    # extract_audio_from_video(video_path, outpath)
    # upscale images
    outupscaled = outpath + "/upscaled"
    # upscale_images_in_folder(outpath, outupscaled)

    # make video from images
    image_paths = get_files_in_dir(outupscaled, [".jpeg", ".jpg", ".png"])
    # get framerate from file if it exists
    if os.path.isfile(os.path.join(outpath, "_framerate.txt")):
        with open(os.path.join(outpath, "_framerate.txt"), "r") as f:
            frame_rate = int(float(f.readline()))
    else:
        print("Warning: Could not find framerate file. Using default framerate 60.")
        frame_rate = 60

    make_video_from_images(
        image_paths, outupscaled + "/upscaled.mp4", frame_rate=frame_rate
    )
