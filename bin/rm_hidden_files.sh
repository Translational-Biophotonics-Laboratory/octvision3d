#!/bin/bash

# Check if a path is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 [path]"
    exit 1
fi

# Assign the provided path to a variable
DIRECTORY=$1

# Check if the provided path is a valid directory
if [ ! -d "$DIRECTORY" ]; then
    echo "Error: $DIRECTORY is not a valid directory."
    exit 1
fi

# Enable globstar for recursive globbing
shopt -s globstar

# Recursively loop through all hidden files in the specified directory and its subdirectories
for file in "$DIRECTORY"/**/.*; do
    if [ -f "$file" ]; then
        echo "Processing $file"
        # Add your processing logic here
    fi
done

