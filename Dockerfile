FROM python:3.7-slim

WORKDIR /usr/src/app

COPY requirements.txt .

RUN apt-get update \
    && apt-get install -y cmake gcc g++ mpi-default-bin libsndfile1-dev \
    && pip install -r requirements.txt && pip install horovod~="0.18.2"

ENV HOME /tmp

USER 1000:1000

ENTRYPOINT [ "jupyter", "notebook", "--no-browser", "--ip", "0.0.0.0" ]
