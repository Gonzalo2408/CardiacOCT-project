#!/usr/bin/env bash

build_and_push_image () {
  TAG=doduo1.umcn.nl/$1
  BUILD_TARGET=$2
  echo "Building and pushing $BUILD_TARGET image: $TAG"
  export DOCKER_BUILDKIT=1
  set -e
  docker build --tag $TAG --target $BUILD_TARGET ./base
  docker push $TAG
}

CUDA_VERSION=`cat ./base/requirements.txt | grep -Eom1 '+cu[0-9]+' | grep -Eo '[0-9]+'`
TORCH_VERSION=`cat ./base/requirements.txt | grep -Eo 'torch==.*\+' | grep -Eo '[0-9]+.[0-9]+.[0-9]+'`
NNUNET_VERSION=`cat ./base/requirements.txt | grep -Eo 'nnUNet\.git@.*' | grep -Eo '[0-9]+.[0-9]+.[0-9]+-[0-9]+'`

echo "Detected CUDA $CUDA_VERSION"
echo "Detected PyTorch $TORCH_VERSION"
echo "Detected NNUNet $NNUNET_VERSION"

BASEIMAGE=nnunet/processor-base:cu$CUDA_VERSION-pt$TORCH_VERSION-nnunet$NNUNET_VERSION

build_and_push_image $BASEIMAGE "base"
