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

./train_nnUNet_2d.sh $1 
./train_nnUNet_3d.sh $1
