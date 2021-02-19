FROM pytorch/pytorch
RUN apt-get update && apt-get install -y wget
WORKDIR /workdir
COPY . /workdir
RUN pip3 install -r requirements.txt
EXPOSE 5000
CMD ["python3","app.py"]
