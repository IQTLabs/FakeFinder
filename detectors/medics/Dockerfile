FROM pytorch/pytorch
RUN apt-get update && apt-get install -y wget libglib2.0-0 libsm6 libxext6 libxrender1 libfontconfig1 build-essential
WORKDIR /workdir
COPY . /workdir
RUN pip3 install -r requirements.txt
CMD ["/bin/bash"]
