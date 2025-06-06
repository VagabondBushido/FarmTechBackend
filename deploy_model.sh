#!/bin/bash

# Create data directory if it doesn't exist
mkdir -p /data

# Copy model to data directory
cp -r vgg16_model.keras /data/

echo "Model deployment completed!" 