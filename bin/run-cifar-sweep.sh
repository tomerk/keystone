#!/bin/bash

LOG_DIR=/mnt/log
mkdir -p $LOG_DIR

for LAMBDA in 0.001 0.1 1.0 10.0 50.0 100.0 500.0 1000.0 5000.0
do
  DATE=`date +%Y%m%d.%H%M%S`
  FILTERS=16384
  LOG_FILE=cifar.32.ps.5.$LAMBDA.$FILTERS.$DATE.log
  
  OMP_NUM_THREADS=1 KEYSTONE_MEM=100g ./bin/run-pipeline.sh \
    pipelines.images.cifar.RandomPatchCifarRawAugmentLazy \
    --trainLocation /mnt/cifar_train.bin \
    --testLocation /mnt/cifar_test.bin \
    --numFilters $FILTERS \
    --lambda $LAMBDA \
    --poolSize 10 \
    --poolStride 10 \
    --patchSize 5 > $LOG_DIR/$LOG_FILE
    
  LOG_FILE=cifar.32.ps.7.$LAMBDA.$FILTERS.$DATE.log
  
  OMP_NUM_THREADS=1 KEYSTONE_MEM=100g ./bin/run-pipeline.sh \
    pipelines.images.cifar.RandomPatchCifarRawAugmentLazy \
    --trainLocation /mnt/cifar_train.bin \
    --testLocation /mnt/cifar_test.bin \
    --numFilters $FILTERS \
    --lambda $LAMBDA \
    --poolSize 9 \
    --poolStride 9 \
    --patchSize 7 > $LOG_DIR/$LOG_FILE
done


