FROM python:3.7

WORKDIR /app

ADD . /app
RUN pip install --trusted-host pypi.python.org -r requirements.txt
RUN pip install --trusted-host pypi.python.org gunicorn gevent

ENV PYTHONUNBUFFERED 1
EXPOSE 8050

CMD ["python", "index.py"]
