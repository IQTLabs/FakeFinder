FROM python:3.7

WORKDIR /app

#ADD . /app
ADD requirements.txt /app
RUN pip install --trusted-host pypi.python.org -r requirements.txt
RUN pip install --trusted-host pypi.python.org gunicorn gevent

EXPOSE 80

CMD ["python", "index.py"]
#CMD ["gunicorn", "-w 4", "-b 0.0.0.0:80", "app:server"]
