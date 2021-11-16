FROM python:3.6

ENV PROJECT_DIR /usr/src/fakefinder-api

WORKDIR ${PROJECT_DIR}

COPY . .

RUN pip3 install -r requirements.txt

ENV PYTHONUNBUFFERED 1
EXPOSE 5000

CMD ["python3", "api.py"]
