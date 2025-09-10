import os.path

from face2face import Face2Face
import cv2
from media_toolkit import VideoFile, ImageFile


f2f = Face2Face(device_id=0)


test_file_folder = os.path.dirname(os.path.abspath(__file__))

test_folder = f"{test_file_folder}/test_media"
output_folder = f"{test_file_folder}/output"


def test_single_face_swap():
    print("test_single_face_swap")
    source_img = f"{test_folder}/test_face_1.jpg"
    target_img = f"{test_folder}/test_face_2.jpg"
    swapped = f2f.swap_img_to_img(source_img, target_img, enhance_face_model=None)
    ImageFile().from_np_array(swapped).save(f"{output_folder}/test_swap.jpg")


def test_multi_face_swap():
    print("test_multi_face_swap")
    source_img = f"{test_folder}/three.jpeg"
    target_img = f"{test_folder}/three_men.jpeg"
    swapped = f2f.swap_img_to_img(source_img, target_img, enhance_face_model='gpen_bfr_2048')
    cv2.imwrite(f"{output_folder}/multi_swap.png", swapped)


def test_embedding_face_swap():
    print("test_embedding_face_swap")
    source_img = f"{test_folder}/test_face_1.jpg"
    target_img = f"{test_folder}/test_face_2.jpg"
    face_name, face = f2f.add_face("hagrid", source_img)
    swapped = f2f.swap(media=target_img, faces=face, enhance_face_model=None)
    cv2.imwrite(f"{output_folder}/test_swap.png", swapped)


def test_image_to_video_swap():
    print("test_image_to_video_swap")
    source_img = f"{test_folder}/test_face_1.jpg"
    target_video = f"{test_folder}/test_video_1.mp4"
    swapped = f2f.swap(media=target_video, faces=source_img, enhance_face_model=None)
    cv2.imwrite(f"{output_folder}/test_swap.png", swapped)


def test_multi_face_with_face_recognition():
    print("test_multi_face_from_reference")
    f1 = f"{test_folder}/test_face_1.jpg"
    f2 = f"{test_folder}/test_face_2.jpg"
    f3 = f"{test_folder}/test_face_3.png"
    f4 = f"{test_folder}/test_face_4.jpg"
    f2f.add_face("hagrid", f1, save=True)
    f2f.add_face("ron", f2, save=True)
    f2f.add_face("biden", f3, save=True)
    f2f.add_face("trump", f4, save=True)

    test_multi_face_img = f"{test_folder}/test_multi_swap_from_reference.jpg"
    swapped = f2f.swap(media=test_multi_face_img, faces={
        "trump": "hagrid",
        "biden": "ron"
    })
    cv2.imwrite("../docs/swap_with_recognition.jpg", swapped)


def test_face_enhancing():
    print("test_face_enhancing")
    source_img = f"{test_folder}/test_face_1.jpg"
    enhanced = f2f.enhance_faces(image=source_img, model='gpen_bfr_2048')
    cv2.imwrite(f"{output_folder}/enhance_test_gpen_bfr_2048.png", enhanced)


def test_face_enhancing_single_face():
    print("test_face_enhancing_single_face")
    source_img = f"{test_folder}/three.jpeg"
    f = f2f.detect_faces(source_img)
    enhanced = f2f.enhance_single_face(image=source_img, target_face=f[0], model='gfpgan_1.4')
    cv2.imwrite(f"{output_folder}/enhance_single_face_gfpgan_1.4.png", enhanced)


def test_video_face_swap():
    print("test_video_face_swap")
    # add ref face
    # source_img = cv2.imread(f"{test_folder}/test_face_4.jpg")
    # f2f.add_face("caprio", source_img, save=True)
    # swap it
    fn = f"{test_folder}/test_video_ultra_short_short.mp4"
    vf = VideoFile().from_file(fn)
    swapped = f2f.swap(media=vf, faces="caprio", enhance_face_model=None)
    swapped.save(f"{output_folder}/{os.path.basename(fn)}_swapped.mp4")


def test_multi_face_video_swap():
    print("test_multi_face_video_swap")
    
    # add ref face
    f2f.add_face("biden", f"{test_folder}/test_face_3.png", save=True)
    f2f.add_face("harris", f"{test_folder}/test_face_5.jpg", save=True)
    f2f.add_face("murphy", f"{test_folder}/test_face_6.jpg", save=True)
    f2f.add_face("warren", f"{test_folder}/test_face_7.jpg", save=True)
    # swap it
    vf = VideoFile().from_file(f"{test_folder}/test_video_1.mp4")
    swapped = f2f.swap(media=vf, enhance_face_model=None, faces={
        "murphy": "biden",
        "warren": "harris",
    })
    swapped.save(f"{output_folder}/", f"{test_folder}/z.mp4")


if __name__ == "__main__":
    #test_single_face_swap()
    #test_multi_face_swap()
    #test_multi_face_with_face_recognition()
    test_embedding_face_swap()
    test_image_to_video_swap()
    test_face_enhancing()
    test_face_enhancing_single_face()
    test_video_face_swap()
    test_multi_face_video_swap()

