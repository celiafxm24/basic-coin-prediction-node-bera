FROM python:3.11-slim AS project_env

RUN apt-get update && apt-get install -y curl

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install --upgrade pip setuptools \
    && pip install -r requirements.txt

FROM project_env

COPY . /app/
COPY updater.sh /app/updater.sh
RUN chmod +x /app/updater.sh  # Make the script executable

CMD ["gunicorn", "--conf", "/app/gunicorn_conf.py", "main:app"]
