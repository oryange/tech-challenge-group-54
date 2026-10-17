FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

# ffmpeg/libgl: OpenCV e áudio · libasound2/libssl3/ca-certificates: Azure Speech SDK
RUN apt-get update && apt-get install -y --no-install-recommends \
      ffmpeg libgl1 libglib2.0-0 libsndfile1 libasound2 libssl3 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
# índice CPU do PyTorch: evita baixar ~5 GB de wheels CUDA
RUN pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

RUN useradd -m medwatch && mkdir -p data models outputs && chown -R medwatch /app
USER medwatch

ENTRYPOINT ["medwatch"]
CMD ["healthcheck"]
