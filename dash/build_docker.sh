#!/bin/bash

docker build -t dash:beta .
docker run --rm -p 80:8050 -v $PWD:/app dash:beta
