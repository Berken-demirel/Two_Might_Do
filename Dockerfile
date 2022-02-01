FROM ubuntu:18.04
# The MAINTAINER instruction sets the author field of the generated images.
MAINTAINER adnan.dogan@metu.edu.tr

## DO NOT EDIT the 3 lines.
RUN mkdir /physionet
COPY ./ /physionet
COPY ./keyboard /etc/default/keyboard
WORKDIR /physionet

## Install your dependencies here using apt install, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
       python3-setuptools \
       python3-pip \
       python3-wheel \
       systemd \
       sudo \
       gnupg2 \
       apt-utils \
       curl \
       ca-certificates \
       python3 \
       python3-pip \
    && curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub | apt-key add - \
    && echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda.list \
    && echo "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list \
    && apt-get purge --autoremove -y curl \
    && rm -rf /var/lib/apt/lists/* \
    # && rm -Rf /usr/share/doc && rm -Rf /usr/share/man \
    && apt-get clean

#ENV DEBIAN_FRONTEND noninteractive
# 460.73.01 in server 
# 418.40.04 in physionet
ENV NVIDIA_MAJOR_VERSION 418
ENV NVIDIA_MINOR_VERSION 40
ENV NVIDIA_PATCH_VERSION 04
ENV CUDA_MAJOR-VERSION 10
ENV CUDA_MINOR_VERSION 1
ENV CUDA_PATCH_VERSION 105

ENV NVIDIA_VERSION $NVIDIA_MAJOR_VERSION.$NVIDIA_MINOR_VERSION.$NVIDIA_PATCH_VERSION 

# For libraries in the cuda-compat-* package: https://docs.nvidia.com/cuda/eula/index.html#attachment-a
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    cuda-compat-10-1=418.40.04-1 \
    cuda-drivers-diagnostic=418.40.04-1 \
    cuda-drivers=418.40.04-1 \
    libcuda1-384=418.40.04-0ubuntu1 \
    libnvidia-cfg1-418=418.40.04-0ubuntu1 \
    libnvidia-common-418=418.40.04-0ubuntu1 \
    libnvidia-compute-418=418.40.04-0ubuntu1 \
    libnvidia-decode-418=418.40.04-0ubuntu1 \
    libnvidia-diagnostic-418=418.40.04-0ubuntu1 \
    libnvidia-encode-418=418.40.04-0ubuntu1 \
    libnvidia-fbc1-418=418.40.04-0ubuntu1 \
    libnvidia-gl-418=418.40.04-0ubuntu1 \
    libnvidia-ifr1-418=418.40.04-0ubuntu1 \
    libxnvctrl0=418.40.04-0ubuntu1 \
    libxnvctrl-dev=418.40.04-0ubuntu1 \
    nvidia-384-dev=418.40.04-0ubuntu1 \
    nvidia-384=418.40.04-0ubuntu1 \
    nvidia-compute-utils-418=418.40.04-0ubuntu1 \
    nvidia-dkms-418=418.40.04-0ubuntu1 \
    nvidia-driver-418=418.40.04-0ubuntu1 \
    nvidia-headless-418=418.40.04-0ubuntu1 \
    nvidia-headless-no-dkms-418=418.40.04-0ubuntu1 \
    nvidia-kernel-common-418=418.40.04-0ubuntu1 \
    nvidia-kernel-source-418=418.40.04-0ubuntu1 \
    nvidia-libopencl1-384=418.40.04-0ubuntu1 \
    nvidia-modprobe=418.40.04-0ubuntu1 \
    nvidia-opencl-icd-384=418.40.04-0ubuntu1 \
    nvidia-settings=418.40.04-0ubuntu1 \
    nvidia-utils-418=418.40.04-0ubuntu1 \
    xserver-xorg-video-nvidia-418=418.40.04-0ubuntu1 \
    #cuda-drivers-418=418.40.04-1 \
    #cuda-drivers-fabricmanager-418=418.40.04-1 \
    #cuda-drivers-fabricmanager=418.40.04-1 \
    #cuda-drivers=418.40.04-1 \
    libnvidia-cfg1-418=418.40.04-0ubuntu1 \
    #libnvidia-common-418=418.40.04-0ubuntu1 \
    libnvidia-compute-418=418.40.04-0ubuntu1 \
    libnvidia-decode-418=418.40.04-0ubuntu1 \
    libnvidia-encode-418=418.40.04-0ubuntu1 \
    #libnvidia-extra-418=418.40.04-0ubuntu1 \
    libnvidia-fbc1-418=418.40.04-0ubuntu1 \
    libnvidia-gl-418=418.40.04-0ubuntu1 \
    libnvidia-ifr1-418=418.40.04-0ubuntu1 \
    #libnvidia-nscq-418=418.40.04-1 \
    libxnvctrl0=418.40.04-0ubuntu1 \
    libxnvctrl-dev=418.40.04-0ubuntu1 \
    nvidia-compute-utils-418=418.40.04-0ubuntu1 \
    nvidia-dkms-418=418.40.04-0ubuntu1 \
    nvidia-driver-418=418.40.04-0ubuntu1 \
    #nvidia-fabricmanager-418=418.40.04-1 \
    #nvidia-fabricmanager-dev-418=418.40.04-1 \
    nvidia-headless-418=418.40.04-0ubuntu1 \
    nvidia-headless-no-dkms-418=418.40.04-0ubuntu1 \
    nvidia-kernel-common-418=418.40.04-0ubuntu1 \
    nvidia-kernel-source-418=418.40.04-0ubuntu1 \
    nvidia-modprobe=418.40.04-0ubuntu1 \
    nvidia-settings=418.40.04-0ubuntu1 \
    nvidia-utils-418=418.40.04-0ubuntu1 \
    xserver-xorg-video-nvidia-418=418.40.04-0ubuntu1 \ 
    cuda-toolkit-10-1=10.1.105-1 \
    cuda-runtime-10-1=10.1.105-1 \
    libcudnn7=7.6.5.32-1+cuda10.1 \
    cuda-10-1=10.1.105-1 && \
    rm -rf /var/lib/apt/lists/*

# Required for nvidia-docker v1
RUN echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf && \
    echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf

ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda-10.1/compat/:/usr/local/cuda-10.1/lib64/:/usr/local/cuda/compat/:/usr/local/cuda/lib64/:/usr/local/cuda-10.2/lib64/

RUN ln -s /usr/bin/pip3 /usr/bin/pip
RUN ln -s /usr/bin/python3 /usr/bin/python
#RUN ln -s /usr/local/cuda-10.2/lib64/libcublas.so.10 /usr/local/cuda-10.2/lib64/libcublas.so

## Include the following line if you have a requirements.txt file.
RUN pip install --upgrade pip && \
    python3 -m pip install -r requirements.txt

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV NVIDIA_DRIVER_VERSION "$NVIDIA_VERSION"
ENV NVIDIA_REQUIRE_CUDA "cuda>=10.1 brand=tesla,driver>=396,driver<397 brand=tesla,driver>=410,driver<411 brand=tesla,driver>=418,driver<419"
