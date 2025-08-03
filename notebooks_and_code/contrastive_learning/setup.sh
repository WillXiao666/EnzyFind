#!/bin/bash

# Setup script for ConPLex-style contrastive learning environment
# This script installs required dependencies and sets up the environment

echo "Setting up ConPLex-style contrastive learning environment..."

# Check if conda is available
if command -v conda &> /dev/null; then
    echo "Using conda for environment setup..."
    
    # Create or update the existing EnzyFind environment
    conda env update -f ../environment.yml
    conda activate EnzyFind
    
    # Install additional dependencies for contrastive learning
    pip install wandb tensorboard
    
    echo "Conda environment updated successfully!"
    
elif command -v pip &> /dev/null; then
    echo "Using pip for dependency installation..."
    
    # Install requirements
    pip install -r requirements.txt
    
    echo "Dependencies installed successfully!"
    
else
    echo "Error: Neither conda nor pip found. Please install Python package manager."
    exit 1
fi

# Test installation
echo "Testing installation..."
python -c "
import torch
import numpy as np
import pandas as pd
import sklearn
print('✓ Core dependencies installed successfully')
print(f'PyTorch version: {torch.__version__}')
print(f'NumPy version: {np.__version__}')
print(f'Pandas version: {pd.__version__}')
print(f'Scikit-learn version: {sklearn.__version__}')
"

if [ $? -eq 0 ]; then
    echo "✓ Installation test passed!"
    echo ""
    echo "Environment setup complete! You can now run:"
    echo "  python demo.py                    # Run demonstration"
    echo "  python train_conplex.py          # Train model with config.yaml"
    echo ""
else
    echo "✗ Installation test failed. Please check dependencies."
    exit 1
fi