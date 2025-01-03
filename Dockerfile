FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel as builder
RUN apt-get update \
    && apt-get install -y --no-install-recommends apt-utils \
    && apt-get install libgomp1 build-essential pandoc -y \
    && apt-get install git -y --no-install-recommends

COPY --from=openjdk:11-jre-slim /usr/local/openjdk-11 /usr/local/openjdk-11
ENV JAVA_HOME /usr/local/openjdk-11
RUN update-alternatives --install /usr/bin/java java /usr/local/openjdk-11/bin/java 1

WORKDIR /home

RUN pip install --no-cache-dir --upgrade pip wheel poetry==1.5.1 poetry-dynamic-versioning \
    && python -m poetry config virtualenvs.create false
COPY . RePlay-Accelerated/
RUN cd RePlay-Accelerated && ./poetry_wrapper.sh install --all-extras

RUN pip install --upgrade torch
RUN pip install rs_datasets
RUN pip install Ninja==1.11.1.1
RUN pip install -U tensorboard

RUN pip3 install triton
RUN pip3 install bitsandbytes
RUN sed -i 's/tl\.libdevice\.llrint/tl\.extra\.cuda\.libdevice\.llrint/g' \
    /opt/conda/lib/python3.11/site-packages/bitsandbytes/triton/quantize_global.py \
    /opt/conda/lib/python3.11/site-packages/bitsandbytes/triton/quantize_rowwise.py \
    /opt/conda/lib/python3.11/site-packages/bitsandbytes/triton/quantize_columnwise_and_transpose.py

CMD ["bash"]
