#!/bin/bash
#SBATCH --job-name=utt-seg-batch
#SBATCH --output=logs/utterance_seg_batch_%A_%a.out
#SBATCH --error=logs/utterance_seg_batch_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=general
#SBATCH --array=1-10%5
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your-email@iu.edu

# Utterance Segmentation Pipeline - Batch Processing
#
# This script processes multiple ASR files in parallel using SLURM job arrays.
# The --array=1-10%5 means: process 10 files, max 5 running simultaneously
#
# Usage:
#   1. Create a file list: ls /path/to/asr/files/*.json > file_list.txt
#   2. Edit SLURM parameters above (array size, time, etc.)
#   3. Submit: sbatch run_slurm_batch.sh

set -e  # Exit on error

echo "================================================"
echo "Batch job started: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURM_NODELIST"
echo "================================================"

# Load conda module
module load anaconda

# Activate conda environment
source activate utterance-seg

# Create necessary directories
mkdir -p logs output

# Set your Anthropic API key
# export ANTHROPIC_API_KEY="your-key-here"

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

echo ""
echo "Processing file $SLURM_ARRAY_TASK_ID of $(wc -l < $FILE_LIST):"
echo "  Input:  $INPUT_ASR"
echo "  Output: $OUTPUT_FILE"
echo ""

# Run the segmentation pipeline
python -m utterance_segmentation.cli \
  --input_asr_json "$INPUT_ASR" \
  --few_shot_examples "$FEW_SHOT_EXAMPLES" \
  --output_json "$OUTPUT_FILE" \
  --min_utterance_length 40 \
  --max_utterance_length 80 \
  --turn_break_threshold 1.5 \
  --max_boundary_candidates 5

echo ""
echo "================================================"
echo "Task $SLURM_ARRAY_TASK_ID completed: $(date)"
echo "Output saved to: $OUTPUT_FILE"
echo "================================================"
