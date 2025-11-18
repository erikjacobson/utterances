#!/bin/bash
#SBATCH --job-name=utt-seg-local
#SBATCH --output=logs/utterance_seg_local_%j.out
#SBATCH --error=logs/utterance_seg_local_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --partition=general
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your-email@iu.edu

# Utterance Segmentation Pipeline - Local LLM (No API Key Required)
#
# This script uses a local GGUF model for semantic segmentation.
# Benefits:
# - No API costs
# - No internet required on compute nodes
# - Works offline
# - Fully reproducible
#
# Prerequisites:
# 1. Download model (from login node with internet):
#    python download_local_model.py --output_dir models
# 2. Install llama-cpp-python:
#    pip install --user llama-cpp-python
#
# Usage: sbatch run_local_llm.sh

set -e  # Exit on error

echo "================================================"
echo "Local LLM Job started: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "================================================"

# Load conda module (adjust module name if needed for your HPC system)
module load anaconda

# Activate conda environment
source activate utterance-seg

# Create logs directory if it doesn't exist
mkdir -p logs

# Input/output paths (MODIFY THESE)
INPUT_ASR="/path/to/your/asr_input.json"
OUTPUT_DIR="output"
OUTPUT_FILE="${OUTPUT_DIR}/utterances_local_${SLURM_JOB_ID}.json"
FEW_SHOT_EXAMPLES="examples/few_shot_boundaries.json"

# Local model path (MODIFY THIS)
# Download model first: python download_local_model.py --output_dir models
LOCAL_MODEL_PATH="models/Phi-3-mini-4k-instruct-Q4_K_M.gguf"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Verify model exists
if [ ! -f "$LOCAL_MODEL_PATH" ]; then
    echo "Error: Local model not found at $LOCAL_MODEL_PATH"
    echo "Download it first using: python download_local_model.py --output_dir models"
    exit 1
fi

echo ""
echo "Processing configuration:"
echo "  Input:         $INPUT_ASR"
echo "  Output:        $OUTPUT_FILE"
echo "  Model:         $LOCAL_MODEL_PATH"
echo "  CPU threads:   $SLURM_CPUS_PER_TASK"
echo ""

# Run the segmentation pipeline with local LLM
python -m utterance_segmentation.cli \
  --input_asr_json "$INPUT_ASR" \
  --few_shot_examples "$FEW_SHOT_EXAMPLES" \
  --output_json "$OUTPUT_FILE" \
  --llm_backend local \
  --local_model_path "$LOCAL_MODEL_PATH" \
  --n_threads $SLURM_CPUS_PER_TASK \
  --min_utterance_length 40 \
  --max_utterance_length 80 \
  --turn_break_threshold 1.5 \
  --max_boundary_candidates 5

echo ""
echo "================================================"
echo "Job completed: $(date)"
echo "Output saved to: $OUTPUT_FILE"
echo "================================================"
