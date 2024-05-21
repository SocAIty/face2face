from setuptools import setup, find_packages

setup(
    name='face2face',
    version='0.0.1',
    description="Swap faces from one image to another. Create face embeddings. Integrate into hosted environments.",
    author='SocAIty',
    packages=find_packages(),
    install_requires=[
        'insightface==0.7.3',
        'onnxruntime-gpu==1.15.1',
        'tqdm',
        'opencv-python',
        'numpy',
        'fastapi',
        'python-multipart',
        'uvicorn',
        'requests'
    ]
)