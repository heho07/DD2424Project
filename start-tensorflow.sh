#!/bin/sh

docker run --rm -it -p 8888:8888 -p 6006:6006 -v "$PWD":/work tensorflow/tensorflow:latest-py3 bash
