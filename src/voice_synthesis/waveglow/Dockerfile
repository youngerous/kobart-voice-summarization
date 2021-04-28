FROM pytorch/pytorch:nightly-devel-cuda10.0-cudnn7
ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}

RUN apt-get update -y
RUN apt-get install git -y
RUN git clone https://github.com/NVIDIA/apex
#ADD apex /apex/

COPY requirements.txt /
RUN pip install -r /requirements.txt

WORKDIR apex
RUN git checkout f3a960f80244cf9e80558ab30f7f7e8cbf03c0a0
RUN pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .
ENV PYTHONUNBUFFERED=0
WORKDIR /waveglow