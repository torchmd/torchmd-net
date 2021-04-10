#!/bin/bash

dev=1 # list CUDA devices to use
logdir="logs" # log dir for storing checkpoints and metrics

mkdir -p $logdir
CUDA_VISIBLE_DEVICES=$dev python src/train.py --conf examples/graph-network.yaml --log-dir $logdir > $logdir/log 2>&1 &
echo "Starting training... (saving logs at $logdir/log)"
