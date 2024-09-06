# Web Service for Face2Face

![image of openapi server](example_server.png)

## Starting the Web Service

If not yet installed use `pip install socaity-face2face[full]` to install the module.

Now in your cmd run the server with:
- `python -m face2face.server` 

Note: The first time you start the server, it will download the models. This can take a while.

## How it works:

The Webservice is built with [FastTaskAPI](https://github.com/SocAIty/FastTaskAPI). 
In this regard, for each request it will create a task and return a job id.
You can then check the status of the job and retrieve the result.


## Configuration

You can configure some settings via Enviornment variables:

Model storage folder: 

use the environment variable `MODELS_DIR="path/to/models"` to set the value to the path where the models should be stored.
For example the inswapper and the GPEN models are downloaded and stored in this folder.

Face embedding folder: 

use the environment variable `FACE_EMBEDDINGS_DIR="path/to/face_embeddings"`. The face-emebddings are stored in this folder.
Stored faces can be reused by the face2face model.

# Deployment, Runpod, Docker, File Uploads and more

For more settings and how to deploy the service check-out FastTaskAPI.
For example it allows you to deploy the service with [Runpod](https://runpod.io) out of the box.


## Usage

The webservice provides enpdoints for the swap, add_face and enhance_face functionality.
You can send requests to the endpoints with any http client, e.g. requests, httpx, curl, etc.

Important Note: Read the documentations of [fastSDK](https://github.com/SocAIty/fastSDK), [FastTaskAPI](https://github.com/SocAIty/FastTaskAPI) and 
[media-toolkit](https://github.com/SocAIty/media-toolkit) to get the most out of the service and to familiarize yourself with the concepts.


However, we recommend using the [fastSDK](https://github.com/SocAIty/fastSDK).




### With fastSDK

FastSDK is a python module that provides a convenient way to interact with the webservice.
You can find an implementation of an SDK generated for fastSDK in the [socaity SDK](https://github.com/SocAIty/socaity/tree/main/socaity/api/image/img2img/face2face) documentation.


### With plain web requests
### Send requests

First encode the image as bytes.
```python
# load images from disc
with open("myimage.jpg", "rb") as image:
    my_image = image.read()
```
Then send a post request to the endpoint.
```python
my_job = requests.post("http://localhost:8020/api/add_face", files={"media": my_image, "faces": "biden"})
```

### Parse the results

The response is a json that includes the job id and meta information.
By sending then a request to the job endpoint you can check the status and progress of the job.
If the job is finished, you will get the result, including the swapped image.
```python
import cv2
from io import BytesIO
# check status of job
response = requests.get(f"http://localhost:8020/api/status?job_id={my_job['job_id']}")
# convert result to image file
swapped = cv2.imread(BytesIO(response.json()['result']))
```

