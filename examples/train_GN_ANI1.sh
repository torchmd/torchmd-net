#!/bin/bash

dev=1 # list CUDA devices to use
logdir="logs" # log dir for storing checkpoints and metrics

mkdir -p $logdir
CUDA_VISIBLE_DEVICES=$dev python scripts/train.py --conf examples/graph-network.yaml --dataset ANI1 --dataset-root ~/data/ani1 --log-dir $logdir --redirect $logdir/log &
echo "Starting training... (saving logs at $logdir/log)"
