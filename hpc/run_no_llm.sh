#!/bin/bash
#SBATCH --job-name=utt-seg-simple
#SBATCH --output=logs/utterance_seg_simple_%j.out
#SBATCH --error=logs/utterance_seg_simple_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=general

# Utterance Segmentation Pipeline - Simple Mode (No LLM)
#
# This version runs much faster and doesn't require API key.
# Uses span-based segmentation without semantic boundary detection.
#
# Usage: sbatch run_no_llm.sh

set -e  # Exit on error

echo "================================================"
echo "Simple segmentation job started: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "================================================"

# Load conda module
module load anaconda

# Activate conda environment
source activate utterance-seg

# Create logs directory
mkdir -p logs output

# Input/output paths (MODIFY THESE)
INPUT_ASR="/path/to/your/asr_input.json"
OUTPUT_DIR="output"
OUTPUT_FILE="${OUTPUT_DIR}/utterances_simple_${SLURM_JOB_ID}.json"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo ""
echo "Processing (simple mode - no LLM):"
echo "  Input:  $INPUT_ASR"
echo "  Output: $OUTPUT_FILE"
echo ""

# Run the segmentation pipeline WITHOUT LLM
python -m utterance_segmentation.cli \
  --input_asr_json "$INPUT_ASR" \
  --output_json "$OUTPUT_FILE" \
  --no_llm \
  --min_utterance_length 40 \
  --max_utterance_length 80 \
  --turn_break_threshold 1.5

echo ""
echo "================================================"
echo "Job completed: $(date)"
echo "Output saved to: $OUTPUT_FILE"
echo "================================================"
