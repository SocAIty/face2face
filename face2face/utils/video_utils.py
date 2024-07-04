import os

import cv2
from tqdm import tqdm

from face2face.settings import OUTPUT_DIR, MODELS_DIR
from face2face.utils import get_files_in_dir



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
