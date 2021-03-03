FROM nvidia/cuda:11.0-base
RUN apt-get update && apt-get install -y python3-pip  wget libglib2.0-0 libsm6 libxext6 libxrender1 libfontconfig1
WORKDIR /workdir
COPY . /workdir
RUN pip3 install -r requirements.txt
EXPOSE 5000
CMD ["python3","app.py"]
