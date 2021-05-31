#!/bin/bash

dev=1 # list CUDA devices to use
logdir="logs" # log dir for storing checkpoints and metrics

mkdir -p $logdir
CUDA_VISIBLE_DEVICES=$dev python scripts/train.py --conf examples/graph-network.yaml --dataset MD17 --dataset-root ~/data/md17 --dataset-arg aspirin --energy-weight 0.05 --force-weight 0.95 --standardize true --log-dir $logdir --redirect true &
echo "Starting training... (saving logs at $logdir/log)"
