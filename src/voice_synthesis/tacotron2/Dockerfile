FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-devel
ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}

RUN apt-get update -y
RUN apt-get install git -y
RUN git clone https://github.com/NVIDIA/apex
#RUN pip install numpy scipy matplotlib librosa==0.6.0 tensorboardX inflect==0.2.5 Unidecode==1.0.22 pillow jupyter

COPY requirements.txt /
#ADD apex /apex/
RUN pip install -r /requirements.txt

WORKDIR apex/
RUN git checkout f3a960f80244cf9e80558ab30f7f7e8cbf03c0a0
RUN pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .
ENV PYTHONUNBUFFERED=0
WORKDIR /tacotron2