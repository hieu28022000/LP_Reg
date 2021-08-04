FROM nvidia/cuda:10.2-cudnn8-devel-ubuntu18.04

COPY . /license_plates_recognition/
WORKDIR /license_plates_recognition
RUN apt-get update \
    && apt-get install python3.8 -y \
    && apt-get update \
    && apt-get install python3-pip -y \
    && python3 -m pip install --upgrade pip \
    && apt-get update \
    && apt install -y libgl1-mesa-glx \
    && pip3 install -r requirements.txt \
    && cd installation && pip3 install -r craft_requirements.txt \
    && pip3 install -r Deeptext_requirements.txt && pip3 install MORAN_requirements.txt\
