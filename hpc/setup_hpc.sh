#!/bin/bash
#
# Setup script for utterance segmentation pipeline on IU HPC
# Usage: bash setup_hpc.sh
#

set -e  # Exit on error

echo "================================================"
echo "Utterance Segmentation Pipeline - HPC Setup"
echo "================================================"

# Load conda module (adjust for your HPC system)
echo "Loading conda module..."
module load anaconda

# Create conda environment
echo "Creating conda environment from environment.yml..."
conda env create -f environment.yml

# Activate environment
echo "Activating environment..."
source activate utterance-seg

# Install the package in development mode
echo "Installing utterance-segmentation package..."
cd ..
pip install -e .

echo ""
echo "================================================"
echo "Setup complete!"
echo "================================================"
echo ""
echo "To use the pipeline:"
echo "  1. Activate environment: conda activate utterance-seg"
echo "  2. Set API key: export ANTHROPIC_API_KEY=your-key-here"
echo "  3. Submit job: sbatch hpc/run_slurm.sh"
echo ""
echo "For batch processing, edit hpc/run_slurm_batch.sh"
echo "================================================"
