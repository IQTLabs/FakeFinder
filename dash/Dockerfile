FROM python:3.7

WORKDIR /app

ADD . /app
RUN apt-get update && apt-get install -y python3-pip
RUN python3 -m pip install --trusted-host pypi.python.org -r requirements.txt
RUN python3 -m pip install --trusted-host pypi.python.org gunicorn gevent

EXPOSE 8050

CMD ["python", "index.py"]
#CMD ["gunicorn", "-w 4", "-b 0.0.0.0:80", "app:server"]
