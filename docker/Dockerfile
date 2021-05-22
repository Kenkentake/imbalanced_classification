FROM pytorch/pytorch

ENV DEBIAN_FRONTEND=noninteractive
RUN addgroup --system app && adduser --system app
ENV APP_DIR=/home/imbalanced_classification
COPY --chown=app:app . $APP_DIR
WORKDIR $APP_DIR

RUN apt-get -y update 
RUN apt-get install -y vim && \
    apt-get install -y libopencv-dev && \
    apt-get install -y lsof && \
    apt-get install -y tree 

RUN pip install matplotlib && \
    pip install tensorboard && \
    pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html && \
    pip install -r $APP_DIR/docker/requirements.txt
