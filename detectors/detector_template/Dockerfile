FROM nvidia/cuda:11.0-base
RUN apt-get update && apt-get install -y python3-pip  wget
WORKDIR /
COPY . /
RUN pip3 install -r requirements.txt
EXPOSE 5000
CMD ["python3","app.py"]
