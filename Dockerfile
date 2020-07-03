FROM python:3.8-buster

WORKDIR /pyfstat-repo

# Ephemerides arguments
## Put ephemerides in this folder
ARG EPHEMERIDES_FOLDER=/ephemerides/
## URL to wget ephemerides from
ARG EPHEMERIDES_URL=https://git.ligo.org/lscsoft/lalsuite/raw/master/lalpulsar/lib/
## Earth epehemerides file name
ARG EPHEMERIDES_EARTH=earth00-40-DE405.dat.gz
## Sun ephemerides file name
ARG EPHEMERIDES_SUN=sun00-40-DE405.dat.gz
ENV LALPULSAR_DATADIR=${EPHEMERIDES_FOLDER}

# Install PyFstat and dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
RUN pip install .

# Download ephemerides
RUN mkdir -p ${EPHEMERIDES_FOLDER}
RUN wget ${EPHEMERIDES_URL}${EPHEMERIDES_EARTH} -P ${EPHEMERIDES_FOLDER}
RUN wget ${EPHEMERIDES_URL}${EPHEMERIDES_SUN} -P ${EPHEMERIDES_FOLDER}
