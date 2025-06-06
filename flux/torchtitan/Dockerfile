# Start with CUDA runtime base (no PyTorch preinstalled)
FROM nvidia/cuda:12.6.3-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-distutils \
    python3-pip \
    git \
    unzip \
    wget \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create symbolic links for python
RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/bin/python3 /usr/bin/python

RUN python -m pip install --upgrade pip --no-cache-dir
RUN pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu126 --no-cache-dir

# Set working directory
WORKDIR /workspace/flux

# Copy application code
COPY . .

RUN pip install --no-cache-dir \
    -r requirements.txt \
    -r torchtitan/experiments/flux/requirements-flux.txt \
    -r requirements-mlperf.txt \
    sentencepiece hf_xet

RUN pip install -e .

