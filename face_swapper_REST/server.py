import argparse
from io import BytesIO

import fastapi
import uvicorn
from fastapi.responses import StreamingResponse
import cv2

import numpy as np

from face_swapper_REST.FaceSwapper import FaceSwapper
from face_swapper_REST.model_downloader import download_face_swap_model
from face_swapper_REST.settings import DEFAULT_PORT


face_swapper = None
def init_models():
    global face_swapper
    if face_swapper is None:
        download_face_swap_model()
        face_swapper = FaceSwapper()



app = fastapi.FastAPI(
    title="Face Swapper FastAPI",
    summary="Swap faces from images.",
    version="0.0.1",
    contact={
        "name": "w4hns1nn",
        "url": "https://github.com/w4hns1nn",
    }
)

async def upload_file_to_cv2(file: fastapi.UploadFile):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def cv2_to_bytes(img: np.ndarray):
    is_success, buffer = cv2.imencode(".png", img)
    io_buf = BytesIO(buffer)
    return io_buf


@app.post("/swap_one")
async def swap_one(source_img: fastapi.UploadFile, target_img: fastapi.UploadFile):
    # init face swapper class if not existing
    init_models()

    source_img = await upload_file_to_cv2(source_img)
    target_img = await upload_file_to_cv2(target_img)

    swapped_img = face_swapper.swap_one_image(source_img, target_img)
    swapped_img = cv2_to_bytes(swapped_img)

    out_file_name = "swapped_img.png"

    return StreamingResponse(
        swapped_img, media_type="png",
        headers={"Content-Disposition": f"attachment; filename={out_file_name}"}
    )


@app.post("/add_reference_face")
async def add_reference_face(face_name: str, source_img: fastapi.UploadFile, save: bool = True):
    # init face swapper class if not existing
    init_models()
    source_img = await upload_file_to_cv2(source_img)
    face_name, face_embedding = face_swapper.add_reference_face(face_name, source_img, save=save)

    return StreamingResponse(
        face_embedding,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f"attachment; filename={face_name}.npz"}
    )


@app.post("/swap_from_reference_face")
async def swap_from_reference_face(face_name: str, target_img: fastapi.UploadFile):
    # init face swapper class if not existing
    init_models()
    target_img = await upload_file_to_cv2(target_img)
    swapped_img = face_swapper.swap_from_reference_face(face_name, target_img)
    swapped_img = cv2_to_bytes(swapped_img)

    return StreamingResponse(
        swapped_img, media_type="png",
        headers={"Content-Disposition": f"attachment; filename={face_name}_swapped.png"}
    )


@app.get("/status")
def status():
    return {"status": "ok"}


# start the server on provided port
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--port", type=int, default=DEFAULT_PORT)
args = arg_parser.parse_args()
uvicorn.run(app, host="localhost", port=args.port)