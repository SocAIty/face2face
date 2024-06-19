import argparse
from io import BytesIO

import fastapi

from multimodal_files import MultiModalFile
from socaity_router import SocaityRouter, ImageFile
from fastapi.responses import StreamingResponse
import cv2

import numpy as np

from face2face.settings import PORT, PROVIDER
from face2face.core.face2face import Face2Face

f2f = Face2Face()
router = SocaityRouter(
    provider=PROVIDER,
    app=fastapi.FastAPI(
        title="Face2Face service",
        summary="Swap faces from images. Create face embeddings. Integrate into hosted environments.",
        version="0.0.2",
        contact={
            "name": "SocAIty",
            "url": "https://github.com/SocAIty",
        }
    ),
)

@router.add_route("/swap_one")
def swap_one(source_img: ImageFile, target_img: ImageFile):
    swapped_img = f2f.swap_one_image(np.array(source_img), np.array(target_img))
    return ImageFile(file_name="swapped_img.png").from_np_array(swapped_img)

@router.add_route("/add_reference_face")
def add_reference_face(face_name: str, source_img: ImageFile = None, save: bool = True):
    face_name, face_embedding = f2f.add_reference_face(face_name, np.array(source_img), save=save)
    return MultiModalFile(file_name=f"{face_name}.npz").from_np_array(face_embedding)

@router.add_route("/swap_from_reference_face")
async def swap_from_reference_face(face_name: str, target_img: ImageFile = None):
    swapped_img = f2f.swap_from_reference_face(face_name, np.array(target_img))
    return ImageFile(file_name=f"swapped_to_{face_name}.png").from_np_array(swapped_img)


def start_server(port: int = PORT):
    router.start(port=port)

# start the server on provided port
if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--port", type=int, default=PORT)
    args = arg_parser.parse_args()
    start_server(port=args.port)
