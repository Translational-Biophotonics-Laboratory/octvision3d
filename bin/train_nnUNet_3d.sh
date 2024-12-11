#!/bin/bash

# Check if DATASET_NUMBER is provided as an argument
if [ -z "$1" ]; then
	echo "Error: Please provide DATASET_NUMBER as the first argument."
	echo "Usage: $0 <DATASET_NUMBER>"
	exit 1
fi

# Validate if the provided argument is a number
if ! [[ "$1" =~ ^[0-9]+$ ]]; then
	echo "Error: DATASET_NUMBER must be a positive integer."
	echo "Usage: $0 <DATASET_NUMBER>"
	exit 1
fi

# Assign the provided argument to DATASET_NUMBER
DATASET_NUMBER=$1

# LowRes 3D UNet
CUDA_VISIBLE_DEVICES=1 nnUNetv2_train $DATASET_NUMBER 3d_fullres 0 -tr CustomNNUNetTrainer -p CustomNNUNetPlans --npz;  # Train first fold individually
sleep 2

CUDA_VISIBLE_DEVICES=0 nnUNetv2_train $DATASET_NUMBER 3d_fullres 1 -tr CustomNNUNetTrainer -p CustomNNUNetPlans --npz & # Train on GPU 0
sleep 2
CUDA_VISIBLE_DEVICES=1 nnUNetv2_train $DATASET_NUMBER 3d_fullres 2 -tr CustomNNUNetTrainer -p CustomNNUNetPlans --npz;  # Train on GPU 1 
wait

CUDA_VISIBLE_DEVICES=0 nnUNetv2_train $DATASET_NUMBER 3d_fullres 3 -tr CustomNNUNetTrainer -p CustomNNUNetPlans --npz & # Train on GPU 0
sleep 2
CUDA_VISIBLE_DEVICES=1 nnUNetv2_train $DATASET_NUMBER 3d_fullres 4 -tr CustomNNUNetTrainer -p CustomNNUNetPlans --npz;  # Train on GPU 1 
wait

