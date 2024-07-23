import threading
from typing import Tuple

import cv2
import onnxruntime
from cv2.typing import Size
import numpy as np
from insightface.app.common import Face
from .face_enhance_models import get_model_config, WARP_TEMPLATES
from face2face.settings import EXECUTION_PROVIDER, DEVICE_ID

# Thread Lock required for onnx engine
THREAD_LOCK = threading.Lock()


def create_static_box_mask(
        crop_size: Size,
        face_mask_blur: float,
        face_mask_padding: Tuple[int, int, int, int]
) -> np.array:
    blur_amount = int(crop_size[0] * 0.5 * face_mask_blur)
    blur_area = max(blur_amount // 2, 1)
    box_mask = np.ones(crop_size, np.float32)
    box_mask[:max(blur_area, int(crop_size[1] * face_mask_padding[0] / 100)), :] = 0
    box_mask[-max(blur_area, int(crop_size[1] * face_mask_padding[2] / 100)):, :] = 0
    box_mask[:, :max(blur_area, int(crop_size[0] * face_mask_padding[3] / 100))] = 0
    box_mask[:, -max(blur_area, int(crop_size[0] * face_mask_padding[1] / 100)):] = 0
    if blur_amount > 0:
        box_mask = cv2.GaussianBlur(box_mask, (0, 0), blur_amount * 0.25)
    return box_mask


def prepare_crop_frame(crop_vision_frame: np.array) -> np.array:
    crop_vision_frame = crop_vision_frame[:, :, ::-1] / 255.0
    crop_vision_frame = (crop_vision_frame - 0.5) / 0.5
    crop_vision_frame = np.expand_dims(crop_vision_frame.transpose(2, 0, 1), axis=0).astype(np.float32)
    return crop_vision_frame


def estimate_matrix_by_face_landmark_5(
        face_landmark_5: np.array,
        warp_template: str,
        crop_size: Size
) -> np.array:
    normed_warp_template = WARP_TEMPLATES.get(warp_template) * crop_size
    affine_matrix = cv2.estimateAffinePartial2D(
            face_landmark_5, normed_warp_template, method=cv2.RANSAC, ransacReprojThreshold=100
    )[0]
    return affine_matrix


def warp_face_by_face_landmark_5(
        temp_vision_frame: np.array,
        face_landmark_5: np.array,
        warp_template: str,  # the ones defined in model templates
        crop_size: Size
) -> Tuple[np.array, np.array]:
    affine_matrix = estimate_matrix_by_face_landmark_5(face_landmark_5, warp_template, crop_size)
    crop_vision_frame = cv2.warpAffine(temp_vision_frame, affine_matrix, crop_size, borderMode=cv2.BORDER_REPLICATE,
                                       flags=cv2.INTER_AREA)
    return crop_vision_frame, affine_matrix


def normalize_crop_frame(crop_vision_frame: np.array) -> np.array:
    crop_vision_frame = np.clip(crop_vision_frame, -1, 1)
    crop_vision_frame = (crop_vision_frame + 1) / 2
    crop_vision_frame = crop_vision_frame.transpose(1, 2, 0)
    crop_vision_frame = (crop_vision_frame * 255.0).round()
    crop_vision_frame = crop_vision_frame.astype(np.uint8)[:, :, ::-1]
    return crop_vision_frame


def get_onnx_inference_session(
        model_path: str,
        execution_provider: tuple = None  # tuple in form ov (execution_provider_name, settings)
):
    if execution_provider is None:
        providers = [(EXECUTION_PROVIDER, {"device_id": DEVICE_ID})]

    with THREAD_LOCK:
        session = onnxruntime.InferenceSession(model_path, providers=providers)
    return session


def apply_enhance(crop_vision_frame: np.array, model_path: str) -> np.array:
    inference_session = get_onnx_inference_session(model_path=model_path)
    frame_processor_inputs = {}
    for frame_processor_input in inference_session.get_inputs():
        if frame_processor_input.name == 'input':
            frame_processor_inputs[frame_processor_input.name] = crop_vision_frame
        if frame_processor_input.name == 'weight':
            weight = np.array([1]).astype(np.double)
            frame_processor_inputs[frame_processor_input.name] = weight

    with THREAD_LOCK:
        crop_vision_frame = inference_session.run(None, frame_processor_inputs)[0][0]

    return crop_vision_frame


def paste_back(
        temp_vision_frame: np.array,
        crop_vision_frame: np.array,
        crop_mask: np.array,
        affine_matrix: np.array
) -> np.array:
    inverse_matrix = cv2.invertAffineTransform(affine_matrix)
    temp_size = temp_vision_frame.shape[:2][::-1]
    inverse_mask = cv2.warpAffine(crop_mask, inverse_matrix, temp_size).clip(0, 1)
    inverse_vision_frame = cv2.warpAffine(crop_vision_frame, inverse_matrix, temp_size, borderMode=cv2.BORDER_REPLICATE)
    paste_vision_frame = temp_vision_frame.copy()
    paste_vision_frame[:, :, 0] = inverse_mask * inverse_vision_frame[:, :, 0] + (1 - inverse_mask) * temp_vision_frame[
                                                                                                      :, :, 0]
    paste_vision_frame[:, :, 1] = inverse_mask * inverse_vision_frame[:, :, 1] + (1 - inverse_mask) * temp_vision_frame[
                                                                                                      :, :, 1]
    paste_vision_frame[:, :, 2] = inverse_mask * inverse_vision_frame[:, :, 2] + (1 - inverse_mask) * temp_vision_frame[
                                                                                                      :, :, 2]
    return paste_vision_frame


def blend_frame(
        temp_vision_frame: np.array,
        paste_vision_frame: np.array,
        blend_strength: float = 0.5  # value between 0 and 1
) -> np.array:
    face_enhancer_blend = 1 - blend_strength
    temp_vision_frame = cv2.addWeighted(
        temp_vision_frame, face_enhancer_blend, paste_vision_frame, 1 - face_enhancer_blend, 0
    )
    return temp_vision_frame


def enhance_face(
        target_face: Face,
        temp_vision_frame: np.array,
        model='gfpgan_1.4'
) -> np.array:
    model_template = get_model_config(model).get('template')
    model_size = get_model_config(model).get('size')
    model_path = get_model_config(model).get('path')

    landmark = target_face.get("kps") # get('landmark_2d_106')

    crop_vision_frame, affine_matrix = warp_face_by_face_landmark_5(
        temp_vision_frame,
        landmark,
        model_template,
        model_size
    )
    box_mask = create_static_box_mask(
        crop_vision_frame.shape[:2][::-1],
        face_mask_blur=0.0,#(0, 0, 0, 0),  # facefusion.globals.face_mask_blur,
        face_mask_padding=(0, 0, 0, 0)
    )
    crop_mask_list = [box_mask]

    # if 'occlusion' in facefusion.globals.face_mask_types:
    #    occlusion_mask = create_occlusion_mask(crop_vision_frame)
    #    crop_mask_list.append(occlusion_mask)

    crop_vision_frame = prepare_crop_frame(crop_vision_frame)
    crop_vision_frame = apply_enhance(crop_vision_frame=crop_vision_frame, model_path=model_path)
    crop_vision_frame = normalize_crop_frame(crop_vision_frame)
    crop_mask = np.minimum.reduce(crop_mask_list).clip(0, 1)
    paste_vision_frame = paste_back(temp_vision_frame, crop_vision_frame, crop_mask, affine_matrix)
    temp_vision_frame = blend_frame(temp_vision_frame, paste_vision_frame)
    return temp_vision_frame
