FROM jonathanshiju/runpod-devel:latest

WORKDIR /workspace

RUN apt update && \
    apt-get install --yes --no-install-recommends \
    portaudio19-dev

RUN git clone https://github.com/Jonathan-Shiju/LiveAgent.git

WORKDIR /workspace/LiveAgent

RUN pip install -r requirements.txt

