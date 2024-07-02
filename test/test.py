from face2face import Face2Face
from media_toolkit import VideoFile
import cv2

f2f = Face2Face()

# test single face swap
#source_img = cv2.imread("test_imgs/test_face_1.jpg")
#target_img = cv2.imread("test_imgs/test_face_2.jpg")
#
#swapped = f2f.swap_one_image(source_img, target_img)
#cv2.imshow("swapped_face", swapped)
#
## test embedding face swap
#f2f.add_reference_face("hagrid", source_img, save=True)
#swapped = f2f.swap_from_reference_face("hagrid", target_img)

# test video face swap
#source_img = cv2.imread("test_imgs/test_face_3.jpg")
#f2f.add_reference_face("black_woman", source_img, save=True)
vf = VideoFile().from_file("test_imgs/test_vid_1.mp4")
swapped = f2f.swap_video(face_name="hagrid", target_video=vf)
swapped.save("test_imgs/swapped_vid_1.mp4")

# debug point
a = 1
