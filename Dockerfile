FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV CUDA_PATH=/usr/local/cuda
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Rome

RUN apt-get update && \
    apt-get install -y tzdata && \
    ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone && \
    dpkg-reconfigure -f noninteractive tzdata && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y \
    build-essential \
    cmake \
    libffi-dev \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    python3-pip \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN python -m pip install --no-cache-dir --upgrade pip
RUN python -m pip install --no-cache-dir \
    setuptools \
    wheel \
    pybind11 \
    psutil \
    numpy \
    numba \
    matplotlib \
    joblib \
    gputil

WORKDIR /tmp
RUN curl -L -o genn-5.2.0.tar.gz https://github.com/genn-team/genn/archive/refs/tags/5.2.0.tar.gz
RUN tar -xzf genn-5.2.0.tar.gz
WORKDIR /tmp/genn-5.2.0
RUN python setup.py install

WORKDIR /app
COPY Components/*.py /app/

CMD ["python3.12", "starter.py"]
