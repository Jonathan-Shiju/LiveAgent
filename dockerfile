FROM jonathanshiju/runpod-runtime:latest

RUN pip install numpy \
    sounddevice

WORKDIR /workspace

RUN git clone https://github.com/Jonathan-Shiju/LiveAgent.git

WORKDIR /workspace/LiveAgent

RUN pip install -r requirements.txt

