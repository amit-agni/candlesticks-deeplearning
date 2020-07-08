# Dockerfile for the Candlestick Image Predictions project
# The non-gpu file did not work but haven't tried after realising --gpus all was missing

FROM tensorflow/tensorflow:latest-gpu


EXPOSE 8888
WORKDIR /project-home 
# Optional
VOLUME /project-home


RUN apt-get update && \
    apt-get install -y git && \
    pip3  --no-cache-dir install numpy matplotlib pandas scikit-learn h5py jupyter

RUN pip3 --no-cache-dir install mplfinance Pillow

CMD ["jupyter", "notebook", "--ip='*'", "--no-browser", "--allow-root"]


# Next Steps :
# docker build -t candlestick_gpu . 

# docker run --gpus all -p 8889:8889 -v "/mnt/c/Amit/OneDrive/Python/Projects/Candlestick Image Prediction:/project-home"  --name cs_gpu candlestick_gpu

# docker run --gpus all -p 8889:8889 -v "/mnt/c/Amit/OneDrive/Python/Projects/Candlestick Image Prediction:/project-home" -v "/mnt/z:/ramdisk" --name cs_gpu candlestick_gpu

# Mount the RAM Disk in wsl (RAM disk created in Z: drive)
# sudo mkdir z
# sudo mount -t drvfs Z: /mnt/z
