#!/bin/bash
#SBATCH --job-name=utt-seg
#SBATCH --output=logs/utterance_seg_%j.out
#SBATCH --error=logs/utterance_seg_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=general
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your-email@iu.edu

# Utterance Segmentation Pipeline - Single File Processing
# Usage: sbatch run_slurm.sh

set -e  # Exit on error

echo "================================================"
echo "Job started: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "================================================"

# Load conda module (adjust module name if needed for your HPC system)
module load anaconda

# Activate conda environment
source activate utterance-seg

# Create logs directory if it doesn't exist
mkdir -p logs

# Set your Anthropic API key (or use environment variable)
# export ANTHROPIC_API_KEY="your-key-here"

# Input/output paths (MODIFY THESE)
INPUT_ASR="/path/to/your/asr_input.json"
OUTPUT_DIR="output"
OUTPUT_FILE="${OUTPUT_DIR}/utterances_${SLURM_JOB_ID}.json"
FEW_SHOT_EXAMPLES="examples/few_shot_boundaries.json"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo ""
echo "Processing configuration:"
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
echo "Job completed: $(date)"
echo "Output saved to: $OUTPUT_FILE"
echo "================================================"
