FROM python:3.7-slim

COPY . /usr/src/mltf2/
WORKDIR  /usr/src/mltf2/
RUN mkdir data  && mkdir models && mkdir libs

RUN apt-get update \
    && apt-get install -y cmake gcc g++ mpi-default-bin libsndfile1-dev curl zlib1g-dev zlib1g libssl-dev libffi-dev \
       zip unzip \
    && pip install -r requirements.txt && pip install horovod~="0.18.2"

# bulid custom Python2 for Bregman Toolkit and VGG16.py
WORKDIR /usr/src/python27
RUN curl -O https://www.python.org/ftp/python/2.7.17/Python-2.7.17.tgz
RUN tar xvzf Python-2.7.17.tgz
WORKDIR /usr/src/python27/Python-2.7.17
RUN sh ./configure --enable-shared --prefix=/usr/install/python27 --enable-unicode=ucs4
RUN make && make install

RUN echo "/usr/install/python27/lib/" >> /etc/ld.so.conf
RUN ldconfig

WORKDIR  /usr/install/python27/bin
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN /usr/install/python27/bin/python get-pip.py

RUN  /usr/install/python27/bin/pip install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.14.0-cp27-none-linux_x86_64.whl
RUN /usr/install/python27/bin/pip install -r /usr/src/mltf2/requirements-py2.txt
RUN /usr/install/python27/bin/python -m pip install ipykernel

WORKDIR /usr/src/mltf2
RUN bash ./download-libs.sh
WORKDIR /usr/src/mltf2/libs/BregmanToolkit
RUN /usr/install/python27/bin/python setup.py install

WORKDIR /usr/src/mltf2
RUN bash ./download-data.sh

ENV HOME /tmp

USER 1000:1000

WORKDIR /usr/src/mltf2
ENTRYPOINT ["bash", "/usr/src/mltf2/mltf-entrypoint.sh"]
