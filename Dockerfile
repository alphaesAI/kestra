FROM kestra/kestra:latest-slim

USER root
WORKDIR /app

# System dependencies for OpenCV & C extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 1. Setup Python 3.11 virtual environment (compatible with prebuilt MMCV wheels)
RUN uv python install 3.11 && \
    uv venv /app/.venv --python 3.11

ENV PATH="/app/.venv/bin:$PATH"

# 2. Install prebuilt PyTorch 2.1 CPU & MMCV 2.1.0 precompiled wheel
RUN uv pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cpu && \
    uv pip install mmcv==2.1.0 --find-links https://download.openmmlab.com/mmcv/dist/cpu/torch2.1/index.html

# 3. Copy project files
COPY src   /app/src
COPY flows /app/flows
COPY examples /app/examples
COPY pyproject.toml  /app/
COPY requirements.txt /app/
COPY README.md       /app/

# 4. Install remaining dependencies into the virtual environment
RUN uv pip install -r /app/requirements.txt

USER kestra
ENV PYTHONPATH=/app/src:/app
ENV APP_DIR=/app

EXPOSE 8080
