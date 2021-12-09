FROM nvidia/cuda:11.0-base
RUN apt-get update && apt-get install -y --no-install-recommends python3-pip wget libglib2.0-0 libsm6 libxext6 libxrender1 libfontconfig1
WORKDIR /app
COPY . /app
RUN python3 -m pip install -U pip && python3 -m pip install -r requirements.txt
ENV PYTHONUNBUFFERED 1
EXPOSE 5000
CMD ["python3","app.py"]
