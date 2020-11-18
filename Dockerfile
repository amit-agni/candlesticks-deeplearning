# Dockerfile for the Candlestick Image Predictions project
# The non-gpu file did not work but haven't tried after realising --gpus all was missing

FROM tensorflow/tensorflow:latest-gpu


EXPOSE 8888
WORKDIR /project-home 
# Optional
VOLUME /project-home

RUN python3 -m pip install --upgrade pip


RUN apt-get update && \
    apt-get install -y git && \
    pip3  --no-cache-dir install numpy matplotlib pandas scikit-learn h5py jupyter 

RUN pip3 --no-cache-dir install mplfinance Pillow tensorflow-addons

RUN pip3 --no-cache-dir install wandb seaborn tensorflow_io alibi-detect

CMD ["jupyter", "notebook", "--ip='*'", "--no-browser", "--allow-root"]


# Next Steps :
# docker build -t candlestick . 

# docker run --gpus all -p 8850:8850 -v "/mnt/c/Amit/OneDrive/Python/Projects/Candlestick Image Prediction:/project-home" -v "/mnt/z:/ramdisk" --name cs candlestick

# Mount the RAM Disk in wsl (RAM disk created in Z: drive)
# sudo mkdir z
# sudo mount -t drvfs Z: /mnt/z
