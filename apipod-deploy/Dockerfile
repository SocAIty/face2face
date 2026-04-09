FROM python:3.8-slim

SHELL ["/bin/bash", "-c"]

# Install system dependencies together with optional extras.
# Always install gcc for reliability; some ML packages still need compilation from source.
# Others we detect automatically based on the deployment flags.
RUN apt-get update && apt-get install -y \
    ffmpeg \
    gcc g++ \
    libcudnn9-cuda-12 \
    libcudnn9-dev-cuda-12 \
    && apt-get clean
# Ensure CUDA and cuDNN libraries are in the library path
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# Set the working directory
WORKDIR /

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install runpod>=1.7.7 && \
    pip install . --no-cache

# Configure APIPod for RunPod Serverless
ENV APIPOD_BACKEND="runpod"
ENV APIPOD_DEPLOYMENT="serverless"

ARG port=8080
ENV APIPOD_PORT=$port
EXPOSE $port
ENV APIPOD_HOST="0.0.0.0"

# Set the entrypoint
CMD ["uvicorn", "face2face\server:app", "--host", "0.0.0.0", "--port", "8000"]