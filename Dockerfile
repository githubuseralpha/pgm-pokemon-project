FROM pytorch/pytorch:2.7.1-cuda11.8-cudnn9-runtime

WORKDIR /workspace

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt
