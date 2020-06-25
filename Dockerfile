FROM python:3.8-buster

WORKDIR /pyfstat
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
RUN pip install .
