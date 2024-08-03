from face2face import Face2Face
import cv2
from media_toolkit import VideoFile


f2f = Face2Face(device_id=0)
def test_single_face_swap():
    source_img = "test_media/test_face_1.jpg"
    target_img = "test_media/test_face_2.jpg"
    swapped = f2f.swap_img_to_img(source_img, target_img, enhance_face_model=None)
    cv2.imwrite("test_swap.png", swapped)

def test_multi_face_swap():
    source_img = "test_media/three.jpeg"
    target_img = "test_media/three_men.jpeg"
    swapped = f2f.swap_img_to_img(source_img, target_img, enhance_face_model='gpen_bfr_2048')
    cv2.imwrite("multi_swap.png", swapped)

def test_embedding_face_swap():
    source_img = "test_media/test_face_1.jpg"
    target_img = "test_media/test_face_2.jpg"
    f2f.add_face("hagrid", source_img, save=True)
    #swapped = f2f.swap_to_faces("hagrid", target_img)
    swapped = f2f.swap(media=target_img, faces="hagrid")
    cv2.imwrite("test_swap.png", swapped)

def test_multi_face_from_reference():
    f1 = "test_media/test_face_1.jpg"
    f2 = "test_media/test_face_2.jpg"
    f3 = "test_media/test_face_3.png"
    f4 = "test_media/test_face_4.jpg"
    f2f.add_face("hagrid", f1, save=True)
    f2f.add_face("ron", f2, save=True)
    f2f.add_face("biden", f3, save=True)
    f2f.add_face("trump", f4, save=True)

    test_multi_face_img = "test_media/test_multi_swap_from_reference.jpg"
    #swapped = f2f.swap_pairs(test_multi_face_img, swap_pairs={
    #    "trump": "hagrid",
    #    "biden": "ron"
    #})
    swapped = f2f.swap(media=test_multi_face_img, faces={
        "trump": "hagrid",
        "biden": "ron"
    })
    cv2.imwrite("../docs/swap_with_recognition.jpg", swapped)


def test_face_enhancing():
    source_img = "test_media/test_face_1.jpg"
    enhanced = f2f.enhance_faces(image=source_img, model='gpen_bfr_2048')
    cv2.imwrite("enhance_test_gpen_bfr_2048.png", enhanced)

def test_face_enhancing_single_face():
    source_img = "test_media/three.jpeg"
    f = f2f.detect_faces(source_img)
    enhanced = f2f.enhance_single_face(image=source_img, target_face=f[0], model='gfpgan_1.4')
    cv2.imwrite("enhance_single_face_gfpgan_1.4.png", enhanced)

def test_video_face_swap():
    # add ref face
    source_img = cv2.imread("test_media/test_face_4.jpg")
    f2f.add_face("caprio", source_img, save=True)
    # swap it
    vf = VideoFile().from_file("test_media/test_video_1.mp4")
    #swapped = f2f.swap_to_face_in_video(face_name="caprio", video=vf)
    swapped = f2f.swap(media=vf, faces="caprio")
    swapped.save("test_media/test_video_2_swapped_swapped_smithy.mp4")

def test_multi_face_video_swap():
    # add ref face
    f2f.add_face("biden", "test_media/test_face_3.png", save=True)
    f2f.add_face("harris", "test_media/test_face_5.jpg", save=True)
    f2f.add_face("murphy", "test_media/test_face_6.jpg", save=True)
    f2f.add_face("warren", "test_media/test_face_7.jpeg", save=True)
    # swap it
    vf = VideoFile().from_file("test_media/test_video_1.mp4")
    #swapped = f2f.swap_pairs_in_video(swap_pairs={
    #    "murphy": "biden",
    #    "warren": "harris"
    #}, video=vf)
    swapped = f2f.swap(media=vf, enhance_face_model=None, faces={
        "murphy": "biden",
        "warren": "harris",
    })
    swapped.save("test_media/z.mp4")


if __name__ == "__main__":
    #test_multi_face_swap()
    #test_multi_face_from_reference()
    #test_face_enhancing()
    #test_face_enhancing_single_face()
    #test_multi_face_video_swap()
    test_video_face_swap()
    a = 1