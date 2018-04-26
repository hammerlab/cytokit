FROM tensorflow/tensorflow:1.7.0-py3

ENV TF_GPU false

RUN apt-get update && apt-get install -y --no-install-recommends git vim wget

RUN mkdir /repos && mkdir /data

RUN cd /repos && \
    git clone https://github.com/hammerlab/flowdec.git && \
    cd flowdec/python && \
    pip install .

RUN cd /repos && \
    git clone -b r0.1 https://github.com/eric-czech/codex.git && \
    cd codex/python/pipeline && \
    pip install .

RUN mkdir -p /data/codex/simulations && \
    cd /data/codex/simulations && \
    wget https://storage.googleapis.com/musc-codex/datasets/simulations/sim-exp-01.zip && \
    unzip sim-exp-01.zip

RUN chmod a+x /repos/codex/python/pipeline/codex/cli/processor.py && \
    ln -s /repos/codex/python/pipeline/codex/cli/processor.py /usr/local/bin/codex-processor