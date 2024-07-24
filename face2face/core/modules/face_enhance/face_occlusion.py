

def create_occlusion_mask(crop_vision_frame: VisionFrame) -> Mask:
    face_occluder = get_face_occluder()
    prepare_vision_frame = cv2.resize(crop_vision_frame, face_occluder.get_inputs()[0].shape[1:3][::-1])
    prepare_vision_frame = numpy.expand_dims(prepare_vision_frame, axis=0).astype(numpy.float32) / 255
    prepare_vision_frame = prepare_vision_frame.transpose(0, 1, 2, 3)
    with conditional_thread_semaphore(facefusion.globals.execution_providers):
        occlusion_mask: Mask = face_occluder.run(None,
                                                 {
                                                     face_occluder.get_inputs()[0].name: prepare_vision_frame
                                                 })[0][0]
    occlusion_mask = occlusion_mask.transpose(0, 1, 2).clip(0, 1).astype(numpy.float32)
    occlusion_mask = cv2.resize(occlusion_mask, crop_vision_frame.shape[:2][::-1])
    occlusion_mask = (cv2.GaussianBlur(occlusion_mask.clip(0, 1), (0, 0), 5).clip(0.5, 1) - 0.5) * 2
    return occlusion_mask
