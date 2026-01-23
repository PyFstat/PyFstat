FROM python:3.13-trixie

WORKDIR /pyfstat-repo

# Install PyFstat and dependencies
COPY . .
RUN pip install .
