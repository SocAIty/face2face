  <h1 align="center" style="margin-top:-25px">Face2Face</h1>

<p align="center">
  <img align="center" src="docs/f2f_icon.png" height="200" />
</p>
  <h3 align="center" style="margin-top:-10px">Instantly swap faces in images and videos</h3>
<br/>
Face2Face is a generative AI technology to swap faces (aka Deep Fake) in images from one to another. 
For example, you can swap your face with Mona Lisa or your favorite celebrity.

With this repository you can:

- Swap faces from one image to another. 
- Swap faces in an entire video.
- Create face embeddings. With these embeddings you can later swap faces without running the whole stack again.
- Run face swapping as a service.

All of this is wrapped into a convenient web (openAPI) API with [FastTaskAPI](https://github.com/SocAIty/FastTaskAPI).
The endpoint allows you to easily deploy face swapping as a service.
The face swapping model itself was created by [Insightface](https://github.com/deepinsight/insightface)
This is a one shot model; for this reason only one face is needed to swap. It should work for all kinds of content, also for anime.
The model is fa


## Example swaps

<table>
<td width="55%"><img src="docs/juntos.jpg"/></td>
<td><img src="docs/pig.jpg" /></td>
</table>


https://github.com/SocAIty/face2face/assets/7961324/f3990fa6-a7b0-463c-a81a-486f658b3c4f

Watch the [hq video](https://www.youtube.com/watch?v=dE-d8DIndco) on youtube

# Setup

### Install via pip
Depending on your use case you can install the package with or without the service.
```bash
# face2face without service (only for inference from script)
pip install socaity-face2face 
# full package with service
pip install socaity-face2face[service]
# or from GitHub for the newest version.
pip install git+https://github.com/SocAIty/face2face
```

Additional dependencies:
- For GPU acceleration also install
pytorch gpu version (with `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`)
- For VideoFile support in the webservice you also need to install [ffmpeg](https://ffmpeg.org/download.html)

### Install and work with the GitHub repository
1. Clone the repository.
2. (Optional) Create a virtual environment. With `python -m venv venv` and activate it with `venv/Scripts/activate`.
3. Install the requirements.
`pip install -r requirements.txt`
4. Install additional dependencies as mentioned above

# Usage

We provide three ways to use the face swapping functionality.
1. [Direct module import and inference](#Inference-from-script) 
2. [By deploying and calling the web service](#Web Service)
3. As part of the [socaity SDK](https://github.com/SocAIty/socaity).  # coming soon


## Inference from script
Use the Face2Face class to swap faces from one image to another.
First create an instance of the class.

```python
from face2face import Face2Face
f2f = Face2Face()
```

### Easy face swapping
Swap faces from one image to another.
```python
swapped_img = f2f.swap_one(cv2.imread("src.jpg"), cv2.imread("target.jpg"))
```

### Face swapping with saved reference faces

Create a face embedding with the add_reference_face function and later swap faces with the swap_from_reference_face function.

If argument save=true is set, the face embedding is persisted and the f2f.swap_from_reference_face function can be used later with the same face_name, even after restarting the project.
```python
embedding = f2f.add_reference_face("my_embedding", source_img, save=True)
swapped = f2f.swap_from_reference_face("my_embedding", target_img)
```

### Swap the faces in a video
Swap faces in a video. The video is read frame by frame and the faces are swapped.
```python
swapped_video = f2f.swap_video(face_name="my_embedding", target_video="my_video.mp4")
```
To use this function you need to install ```socaity-face2face[service]``` or the media_toolkit package.

### Face swapping with a generator
Iteratively swapping from a list of images 
```python
def my_image_generator():
    for i in range(100):
        yield cv2.imread(f"image_{i}.jpg")

for swapped_img in f2f.swap_generator(face_name="my_embedding", target_img_generator=my_image_generator()):
    cv2.imshow("swapped", swapped_img)
    cv2.waitKey(1)
```

## Web Service

1. Start the server by running the provided .bat file "start_server.bat" 
   2. or by using `python face2face/server.py --port 8020` make sure the python PYTHONPATH is set to the root of this repository.
   3. or if module was installed via pypi by running `from face2face.server import start_server` and then `start_server(port=8020)`
2. To test the server, open `http://localhost:8020/docs` in your browser. You should see the openapi documentation.

![image of openapi server](docs/example_server.png)

Note: The first time you start the server, it will download the models. This can take a while.
If this fails, you can download the files manually and store them in models/ or models/insightface/inswapper_128.onnx

The Webservice is built with [FastTaskAPI](https://github.com/SocAIty/FastTaskAPI). 
In this regard, for each request it will create a task and return a job id

### Face2Face (aka swapping) 


```python
import requests

# load images from disc
with open("src.jpg", "rb") as image:
    src_img = image.read()
with open("target.jpg", "rb") as image:
    target_img = image.read()

# send post request
job = requests.post("http://localhost:8020/api/swap_one", files={"source_img": src_img, "target_img": target_img})
```

### For face embedding generation

```python
import requests

with open("src.jpg", "rb") as image:
    src_img = image.read()

response = requests.post("http://localhost:8020/api/add_reference_face", params={ "face_name": "myface", "save": True}, files={"source_img": src_img})
```
The response is a .npz file as bytes. 
After the embedding was created it can be used in the next swapping with the given face_name.

### For face swapping with saved reference faces

```python
import requests
with open("target.jpg", "rb") as image:
    target_img = image.read()

response = requests.post(
   "http://localhost:8020/api/swap_from_reference_face", 
    params={ "face_name" : "myface"}, files={"target_img": target_img}
)
```
In this example it is assumed that previously a face embedding with name "myface" was created with the add_reference_face endpoint.

### Swap faces in an entire video

```python
import httpx
from media_toolkit import VideoFile
my_video = VideoFile("my_video.mp4")
request = httpx.post(
   "http://localhost:8020/swap_video", params={ "face_name" : "myface"}, 
    files={"target_video": my_video.to_httpx_send_able_tuple()}
)
```

### Parse the results

The response is a json that includes the job id and meta information.
By sending then a request to the job endpoint you can check the status and progress of the job.
If the job is finished, you will get the result, including the swapped image.
```python
import cv2
from io import BytesIO
# check status of job
response = requests.get(f"http://localhost:8020/api/job/{job.json()['job_id']}")
# convert result to image file
swapped = cv2.imread(BytesIO(response.json()['result']))
```
If you want it more convenient use [fastSDK](https://github.com/SocAIty/fastSDK) to built your client,
or the [socaity SDK](https://github.com/SocAIty/socaity).

## Disclaimer

The author is not responsible of any misuse of the repository. Face swapping is a powerful technology that can be used for good and bad purposes.
Please use it responsibly and do not harm others. Do not publish any images without the consent of the people in the images.
The credits for face swapping technology go to the great Insightface Team thank you [insightface.ai](https://insightface.ai/). 
This project uses their pretrained models and parts of their code. Special thanks goes to their work around [ROOP](https://github.com/s0md3v/sd-webui-roop).
The author does not claim authorship for this repository. The authors contribution was to provide a convenient API and service around the face swapping.


# Contribute

Any help with maintaining and extending the package is welcome. Feel free to open an issue or a pull request.

ToDo: 
- make inference faster by implementing batching.
- Implement strength factor for applied face
- use a face enhancer gan to improve the quality of the swapped faces.
- create real streaming in the webserver
- improve streaming speed  
- implement video2video with auto face recognition
