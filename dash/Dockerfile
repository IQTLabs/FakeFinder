FROM python:3.7

WORKDIR /app

ADD . /app
RUN apt-get update && apt-get install -y python3-pip
RUN python3 -m pip install --trusted-host pypi.python.org -r requirements.txt
RUN python3 -m pip install --trusted-host pypi.python.org gunicorn gevent

ENV PYTHONUNBUFFERED 1
EXPOSE 8050

CMD ["python", "index.py"]
