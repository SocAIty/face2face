from setuptools import setup, find_packages

setup(
    name='socaity-face2face',
    version='0.0.1',
    description="Swap faces from one image to another. Create face embeddings. Integrate into hosted environments.",
    author='SocAIty',
    packages=find_packages(),
    install_requires=[
        'insightface==0.7.3',
        'onnxruntime-gpu==1.15.1',
        'socaity-router'
        'tqdm',
        'opencv-python',
        'numpy',
        'python-multipart',
        'requests'
    ]
)