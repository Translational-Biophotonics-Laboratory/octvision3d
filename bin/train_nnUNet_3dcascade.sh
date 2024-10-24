#!/bin/bash

DATASET_NUMBER=3

# LowRes 3D UNet
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train $DATASET_NUMBER 3d_lowres 0 --npz;  # Train first fold individually

CUDA_VISIBLE_DEVICES=0 nnUNetv2_train $DATASET_NUMBER 3d_lowres 1 --npz & # Train on GPU 0
CUDA_VISIBLE_DEVICES=1 nnUNetv2_train $DATASET_NUMBER 3d_lowres 2 --npz;  # Train on GPU 1 

CUDA_VISIBLE_DEVICES=0 nnUNetv2_train $DATASET_NUMBER 3d_lowres 3 --npz & # Train on GPU 0
CUDA_VISIBLE_DEVICES=1 nnUNetv2_train $DATASET_NUMBER 3d_lowres 4 --npz;  # Train on GPU 1 

# HighRes 3D Cascade UNet
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train $DATASET_NUMBER 3d_cascade_fullres 0 --npz;  # Train first fold individually

CUDA_VISIBLE_DEVICES=0 nnUNetv2_train $DATASET_NUMBER 3d_cascade_fullres 1 --npz & # Train on GPU 0
CUDA_VISIBLE_DEVICES=1 nnUNetv2_train $DATASET_NUMBER 3d_cascade_fullres 2 --npz;  # Train on GPU 1 

CUDA_VISIBLE_DEVICES=0 nnUNetv2_train $DATASET_NUMBER 3d_cascade_fullres 3 --npz & # Train on GPU 0
CUDA_VISIBLE_DEVICES=1 nnUNetv2_train $DATASET_NUMBER 3d_cascade_fullres 4 --npz;  # Train on GPU 1 
