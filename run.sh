#!/bin/bash

 docker run --rm -it \
    --platform linux/amd64 \
    --name gfr \
    -v ./workspace:/workspace \
    gfootball:latest
