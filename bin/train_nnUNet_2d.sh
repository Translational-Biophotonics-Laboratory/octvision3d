#!/bin/bash

DATASET_NUMBER=3

# LowRes 3D UNet
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train $DATASET_NUMBER 2d 0 --npz;  # Train first fold individually
sleep 2

CUDA_VISIBLE_DEVICES=0 nnUNetv2_train $DATASET_NUMBER 2d 1 --npz & # Train on GPU 0
sleep 2
CUDA_VISIBLE_DEVICES=1 nnUNetv2_train $DATASET_NUMBER 2d 2 --npz;  # Train on GPU 1 
wait

CUDA_VISIBLE_DEVICES=0 nnUNetv2_train $DATASET_NUMBER 2d 3 --npz & # Train on GPU 0
sleep 2
CUDA_VISIBLE_DEVICES=1 nnUNetv2_train $DATASET_NUMBER 2d 4 --npz;  # Train on GPU 1 
wait




