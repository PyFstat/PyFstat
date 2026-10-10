FROM python:3.14-trixie

WORKDIR /pyfstat-repo

# Install PyFstat and dependencies
COPY . .
RUN pip install .
