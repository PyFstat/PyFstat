FROM python:3.9-buster

WORKDIR /pyfstat-repo

# Install PyFstat and dependencies
COPY . .
RUN pip install .
