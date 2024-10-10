# FROM jupyter/scipy-notebook:2c80cf3537ca
FROM python:3.8.10

COPY . /app/
WORKDIR /app/

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD ["python", "main.py"]