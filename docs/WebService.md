
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