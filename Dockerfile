FROM continuumio/conda-ci-linux-64-python3.8
USER root

RUN conda install pytorch torchvision cpuonly -c pytorch
RUN pip install transformers[torch]
WORKDIR ~/
COPY textgen.py .