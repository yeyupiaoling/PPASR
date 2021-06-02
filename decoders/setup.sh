#!/usr/bin/env bash

sudo apt-get update
if [ $? != 0 ]; then
    echo "Update failed !!!"
    exit 1
fi

# install dependencies
sudo apt-get install -y pkg-config libflac-dev libogg-dev libvorbis-dev libboost-dev swig libsndfile1 git vim gcc
if [ $? != 0 ]; then
    echo "Install dependencies failed !!!"
    exit 1
fi
echo "Success installde pkg-config libflac-dev libogg-dev libvorbis-dev libboost-dev swig libsndfile1 git vim gcc"


if [ ! -d kenlm ]; then
    git clone https://github.com/kpu/kenlm.git
    cd kenlm/
    git checkout df2d717e95183f79a90b2fa6e4307083a351ca6a
    cd ..
    echo -e "\n"
fi

if [ ! -d openfst-1.6.3 ]; then
    echo "Download and extract openfst ..."
    wget http://www.openfst.org/twiki/pub/FST/FstDownload/openfst-1.6.3.tar.gz
    tar -xzvf openfst-1.6.3.tar.gz
    echo -e "\n"
fi

if [ ! -d ThreadPool ]; then
    git clone https://github.com/progschj/ThreadPool.git
    echo -e "\n"
fi


echo "Install decoders ..."
python3 setup.py install --num_processes 4
