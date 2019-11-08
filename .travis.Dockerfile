FROM ubuntu:18.04

RUN apt-get update -qq

RUN apt-get install -y -q gcc-7 g++-7 yasm libxinerama-dev libxcursor-dev \
                          libglu1-mesa-dev libboost-all-dev libx264-dev \
                          libxrandr-dev libxi-dev libglew-dev vpx-tools \
                          binutils libjpeg-turbo8-dev libglfw3-dev git automake

RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 99
RUN update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-7 99

RUN useradd --create-home --shell /bin/bash user
COPY . /home/user/alfalfa/
RUN chown user -R /home/user/alfalfa/

ENV LANG C.UTF-8
ENV LANGUAGE C:en
ENV LC_ALL C.UTF-8

USER user
