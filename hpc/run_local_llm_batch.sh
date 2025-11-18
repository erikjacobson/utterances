#!/bin/bash
#SBATCH --job-name=utt-seg-local-batch
#SBATCH --output=logs/utterance_seg_local_batch_%A_%a.out
#SBATCH --error=logs/utterance_seg_local_batch_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --partition=general
#SBATCH --array=1-10%5
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your-email@iu.edu

# Utterance Segmentation Pipeline - Local LLM Batch Processing
#
# This script processes multiple ASR files in parallel using local GGUF models.
# The --array=1-10%5 means: process 10 files, max 5 running simultaneously
#
# Benefits over API batch processing:
# - No API costs
# - No rate limiting
# - Works without external network access on compute nodes
# - Fully reproducible
#
# Prerequisites:
# 1. Create file list: ls /path/to/asr/files/*.json > file_list.txt
# 2. Download model: python download_local_model.py --output_dir models
# 3. Install llama-cpp-python: pip install --user llama-cpp-python
# 4. Edit SLURM parameters above (array size, time, etc.)
#
# Usage: sbatch run_local_llm_batch.sh

set -e  # Exit on error

echo "================================================"
echo "Local LLM Batch job started: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "================================================"

# Load conda module
module load anaconda

# Activate conda environment
source activate utterance-seg

# Create necessary directories
mkdir -p logs output

# File list containing paths to ASR JSON files (one per line)
FILE_LIST="file_list.txt"

# Get the input file for this array task
INPUT_ASR=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$FILE_LIST")

if [ -z "$INPUT_ASR" ]; then
    echo "Error: No file found for array task $SLURM_ARRAY_TASK_ID"
    exit 1
fi

# Extract basename for output file
BASENAME=$(basename "$INPUT_ASR" .json)
OUTPUT_FILE="output/${BASENAME}_utterances.json"
FEW_SHOT_EXAMPLES="examples/few_shot_boundaries.json"

# Local model path
LOCAL_MODEL_PATH="models/Phi-3-mini-4k-instruct-Q4_K_M.gguf"

# Verify model exists
if [ ! -f "$LOCAL_MODEL_PATH" ]; then
    echo "Error: Local model not found at $LOCAL_MODEL_PATH"
    echo "Download it first using: python download_local_model.py --output_dir models"
    exit 1
fi

echo ""
echo "Processing file $SLURM_ARRAY_TASK_ID of $(wc -l < $FILE_LIST):"
echo "  Input:       $INPUT_ASR"
echo "  Output:      $OUTPUT_FILE"
echo "  Model:       $LOCAL_MODEL_PATH"
echo "  CPU threads: $SLURM_CPUS_PER_TASK"
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
echo "Task $SLURM_ARRAY_TASK_ID completed: $(date)"
echo "Output saved to: $OUTPUT_FILE"
echo "================================================"
