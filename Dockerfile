FROM python:3.8

WORKDIR /app
COPY requirements.txt /app

RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

RUN pip3 install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu116

ENTRYPOINT [ "bash" ]