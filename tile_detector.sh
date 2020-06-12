#!/bin/sh

input_path=$1

docker run -v ${input_path}:/var/local --shm-size 2G --rm tile_detector
