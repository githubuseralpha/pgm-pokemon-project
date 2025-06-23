FROM pytorch/pytorch:2.7.1-cuda11.8-cudnn9-devel

WORKDIR /workspace

COPY requirements.txt .

RUN apt-get update && apt-get install -y git && apt-get clean

RUN pip install --no-cache-dir -r requirements.txt
