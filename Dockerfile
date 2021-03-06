FROM nvidia/cuda

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        openssh-client \
        openmpi-bin \
        libopenmpi-dev \
        libhdf5-openmpi-dev \
        wget \
        cmake \
        git && \
    rm -rf /var/lib/apt/lists/*

ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/:$LD_LIBRARY_PATH

RUN wget -q --no-check-certificate -P /tmp https://arrayfire.s3.amazonaws.com/3.7.1/ArrayFire-v3.7.1-1_Linux_x86_64.sh && \
    chmod +x /tmp/ArrayFire-v3.7.1-1_Linux_x86_64.sh && \
    ./tmp/ArrayFire-v3.7.1-1_Linux_x86_64.sh --include-subdir --prefix=/opt && \
    rm -rf /tmp/ArrayFire-v3.7.1-1_Linux_x86_64.sh

ENV LD_LIBRARY_PATH=/opt/arrayfire/lib64:$LD_LIBRARY_PATH

RUN git clone --recursive https://github.com/keichi/mpEDM.git && \
    cd mpEDM && \
    mkdir build && \
    cd build && \
    cmake -DHDF5_DIR=/opt/hdf5/share/cmake/hdf5/ -DArrayFire_DIR=/opt/arrayfire/share/ArrayFire/cmake/ -DUSE_BOOST=OFF -DCMAKE_BUILD_TYPE=Release .. && \
    make

ENV PATH=/mpEDM/build:$PATH