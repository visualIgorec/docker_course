FROM jupyter/scipy-notebook:2c80cf3537ca

RUN pip install scikit-learn
RUN pip install --upgrade pip
RUN pip install psycopg2