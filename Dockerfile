FROM python:3.7-slim

# Internal unpriviled user will have this ID:
ENV CONTAINER_USER_ID="mltf2" \
    CONTAINER_GROUP_ID="mltf2"

COPY . /usr/src/mltf2/
WORKDIR  /usr/src/mltf2/
RUN mkdir data  && mkdir models && mkdir libs

RUN apt-get update \
    && apt-get install -y cmake gcc g++ mpi-default-bin pkg-config libpng-dev libfreetype6-dev libsndfile1-dev curl zlib1g-dev zlib1g libssl-dev libffi-dev \
       zip unzip \
    && pip install -r requirements.txt && pip install horovod~="0.18.2"

# creates a user "mltf2"
RUN useradd -U -d /home/mltf2 -s /bin/sh ${CONTAINER_USER_ID}
RUN mkdir /home/mltf2

# permissions
RUN chown -R mltf2:mltf2 /home/mltf2

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
RUN /usr/install/python27/bin/python2.7 get-pip.py

RUN  /usr/install/python27/bin/pip2.7 install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.14.0-cp27-none-linux_x86_64.whl
RUN /usr/install/python27/bin/pip2.7 install -r /usr/src/mltf2/requirements-py2.txt
RUN /usr/install/python27/bin/python2.7 -m pip install ipykernel

WORKDIR /usr/src/mltf2
RUN bash ./download-libs.sh
WORKDIR /usr/src/mltf2/libs/BregmanToolkit
RUN /usr/install/python27/bin/python setup.py install

WORKDIR /usr/src/mltf2
RUN bash ./download-data.sh

# permissions
RUN chown -R mltf2:mltf2 /usr/src/mltf2

USER mltf2

WORKDIR /usr/src/mltf2
ENTRYPOINT ["bash", "/usr/src/mltf2/mltf-entrypoint.sh"]
