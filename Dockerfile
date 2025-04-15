FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel as builder
RUN apt-get update \
    && apt-get install -y --no-install-recommends apt-utils \
    && apt-get install libgomp1 build-essential pandoc -y \
    && apt-get install git -y --no-install-recommends

COPY --from=openjdk:11-jre-slim /usr/local/openjdk-11 /usr/local/openjdk-11
ENV JAVA_HOME /usr/local/openjdk-11
RUN update-alternatives --install /usr/bin/java java /usr/local/openjdk-11/bin/java 1

WORKDIR /home

RUN pip install --no-cache-dir --upgrade pip wheel
COPY . project_src/
WORKDIR /home/project_src

RUN pip install numpy==1.24.4 \
    lightning==2.5.1 \
    pandas==1.5.3 \
    polars==1.0.0 \
    optuna==3.2.0 \
    scipy==1.9.3 \
    psutil==6.0.0 \
    scikit-learn==1.3.2 \
    pyarrow==16.0.0

RUN pip install torch==2.5.1
RUN pip install rs_datasets==0.5.1
RUN pip install Ninja==1.11.1.1
RUN pip install -U tensorboard==2.19.0

CMD ["bash"]