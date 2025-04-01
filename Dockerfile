FROM python:3.11-slim AS project_env

RUN apt-get update && apt-get install -y curl

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install --upgrade pip setuptools \
    && pip install -r requirements.txt

FROM project_env

COPY . /app/

CMD ["gunicorn", "--conf", "/app/gunicorn_conf.py", "main:app"]
