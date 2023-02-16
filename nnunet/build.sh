#!/usr/bin/env bash

docker build \
  --tag doduo1.umcn.nl/grodriguez/nnunet:1.7.0-customized \
  --tag doduo1.umcn.nl/grodriguez/nnunet:latest \
  . && \
docker push doduo1.umcn.nl/grodriguez/nnunet:1.7.0-customized && \
docker push doduo1.umcn.nl/grodriguez/nnunet:latest