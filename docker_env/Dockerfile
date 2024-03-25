FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel

# Set timezone to avoid Timezone requirement freeze
ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && apt-get install -y --no-install-recommends \
   vim libgl1-mesa-dev libglib2.0-0 \
   git \
   && rm -rf /var/lib/apt/lists/* 

RUN pip install --upgrade pip

RUN pip install jupyterlab sentencepiece transformers accelerate datasets \
    evaluate pycocotools opencv-contrib-python ftfy regex tqdm 
RUN pip install git+https://github.com/openai/CLIP.git

RUN jupyter lab --generate-config
RUN JUPYTER_CONFIG_FILE="/root/.jupyter/jupyter_lab_config.py" && \
echo "c.IPKernelApp.pylab = 'inline' \n\
c.NotebookApp.open_browser = False \n\
c.NotebookApp.ip = '0.0.0.0' \n\
c.NotebookApp.port = 8888 \n\
c.NotebookApp.allow_remote_access = True" \
>> $JUPYTER_CONFIG_FILE

WORKDIR /workspace/myspace
