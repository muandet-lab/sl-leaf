#FROM --platform=linux/amd64 python:3.11-bullseye
FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime

# Define build arguments
ARG USER_DIR
ARG PYTHONPATH

RUN mkdir $USER_DIR
WORKDIR $USER_DIR

COPY requirements ./
RUN pip install -r requirements

ENV PYTHONPATH=$PYTHONPATH