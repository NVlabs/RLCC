FROM nvcr.io/nvidia/pytorch:22.05-py3 as base
RUN apt update
ENV DEBIAN_FRONTEND=noninteractive
RUN apt -yq install pip libxml2-dev
RUN pip install stable_baselines3==1.6.2 gym==0.21 PyYAML==6.0 pytest==7.2.0 wandb==0.13.7
RUN git clone https://github.com/NVlabs/RLCC.git
WORKDIR RLCC/reinforcement_learning
