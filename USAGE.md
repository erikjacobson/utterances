# Usage Guide

Comprehensive guide for using the utterance segmentation pipeline.

## Table of Contents

- [Quick Start](#quick-start)
- [Installation Options](#installation-options)
- [LLM Backend Selection](#llm-backend-selection)
- [Command-Line Interface](#command-line-interface)
- [HPC Usage](#hpc-usage)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)

## Quick Start

### 5-Minute Quick Start (No LLM)

```bash
# Install package
pip install -e .

# Run on sample data
python -m utterance_segmentation.cli \
  --input_asr_json examples/sample_asr.json \
  --output_json output.json \
  --no_llm
```

### Standalone Version (No Installation)

```bash
# Just download and run (CPU-only)
python standalone_utterance_seg.py \
  --input_asr_json examples/sample_asr.json \
  --output_json output.json \
  --no_llm

# Install minimal dependencies
pip install pydantic
```

## Installation Options

### Option 1: Full Package (Recommended for Development)

```bash
# Basic installation
pip install -e .

# With local LLM support
pip install -e ".[local]"

# With development tools
pip install -e ".[dev]"

# All features
pip install -e ".[local,dev]"
```

### Option 2: Standalone Script (No Installation)

Perfect for quick deployments, HPC, or portability:

```bash
# Copy the script
cp standalone_utterance_seg.py /your/destination/

# Install minimal dependencies
pip install pydantic anthropic        # For API mode
pip install llama-cpp-python          # For local model mode
```

## LLM Backend Selection

### Comparison Table

| Backend | Quality | Speed | Cost | Internet | Setup |
|---------|---------|-------|------|----------|-------|
| **Anthropic API** | ⭐⭐⭐⭐⭐ | Fast | ~$0.05-0.10/file | Required | API key only |
| **Local GGUF** | ⭐⭐⭐⭐ | Medium | Free | No* | Download model |
| **No LLM** | ⭐⭐ | Very Fast | Free | No | None |

*Internet required only for initial model download

### When to Use Each Backend

**Anthropic API:**
- Final production datasets
- When quality is paramount
- Small to medium datasets (<1000 files)
- Budget available for API costs

**Local GGUF:**
- HPC/cluster environments without external network
- Large datasets (>1000 files)
- Budget constraints
- Offline processing required
- Reproducibility important

**No LLM:**
- Development and testing
- Quick prototyping
- Pipeline debugging
- When semantic quality not critical

## Command-Line Interface

### Basic Usage

**With Anthropic API:**
```bash
export ANTHROPIC_API_KEY=sk-ant-...

python -m utterance_segmentation.cli \
  --input_asr_json data/session.json \
  --few_shot_examples examples/few_shot_boundaries.json \
  --output_json output/utterances.json
```

**With Local Model:**
```bash
# First, download a model
python download_local_model.py

# Then run
python -m utterance_segmentation.cli \
  --input_asr_json data/session.json \
  --few_shot_examples examples/few_shot_boundaries.json \
  --output_json output/utterances.json \
  --llm_backend local \
  --local_model_path models/Phi-3-mini-4k-instruct-Q4_K_M.gguf \
  --n_threads 8
```

**Without LLM:**
```bash
python -m utterance_segmentation.cli \
  --input_asr_json data/session.json \
  --output_json output/utterances.json \
  --no_llm
```

### Complete CLI Options

**Required:**
- `--input_asr_json PATH` - Input ASR JSON file
- `--output_json PATH` - Output utterances JSON file

**LLM Backend:**
- `--llm_backend {anthropic,local}` - LLM backend (default: anthropic)
- `--anthropic_api_key KEY` - Anthropic API key
- `--local_model_path PATH` - Path to GGUF model file
- `--n_ctx SIZE` - Context size for local model (default: 2048)
- `--n_threads NUM` - CPU threads for local model (default: auto)
- `--no_llm` - Skip LLM entirely
- `--verbose` - Enable verbose llama.cpp logs

**Segmentation Parameters:**
- `--few_shot_examples PATH` - Few-shot examples JSON
- `--turn_break_threshold SEC` - Pause threshold (default: 1.5s)
- `--min_utterance_length WORDS` - Min length (default: 40)
- `--max_utterance_length WORDS` - Max length (default: 80)
- `--max_boundary_candidates NUM` - Max candidates/span (default: 5)
- `--num_few_shot_examples NUM` - Examples in prompts (default: 3)

**Output:**
- `--quiet` - Suppress diagnostic output

## HPC Usage

### Standalone Script on HPC

Easiest option - no package installation:

```bash
# Install minimal dependencies
module load python
pip install --user pydantic anthropic

# Run directly
python standalone_utterance_seg.py \
  --input_asr_json /path/to/asr.json \
  --output_json output/utterances.json \
  --no_llm
```

### Full Package on HPC

```bash
# Setup environment
cd hpc
bash setup_hpc.sh

# Activate
conda activate utterance-seg
```

### SLURM Job Submission

**Single File with Anthropic API:**
```bash
# Edit paths in run_slurm.sh
nano run_slurm.sh

# Submit
export ANTHROPIC_API_KEY=sk-ant-...
sbatch run_slurm.sh
```

**Batch Processing with Local Model:**
```bash
# One-time setup
python download_local_model.py --output_dir models
pip install --user llama-cpp-python

# Create file list
ls /path/to/asr/*.json > file_list.txt

# Edit array size in script
nano run_local_llm_batch.sh
# --array=1-N%M where N=total files, M=max simultaneous

# Submit
sbatch run_local_llm_batch.sh
```

See [HPC/README_HPC.md](hpc/README_HPC.md) for complete documentation.

## Examples

### Example 1: Process Single File (Highest Quality)

```bash
export ANTHROPIC_API_KEY=sk-ant-...

python -m utterance_segmentation.cli \
  --input_asr_json data/interview_001.json \
  --few_shot_examples examples/few_shot_boundaries.json \
  --output_json output/interview_001_utterances.json \
  --min_utterance_length 40 \
  --max_utterance_length 80
```

### Example 2: Batch Process with Local Model

```bash
# Download model once
python download_local_model.py --output_dir models

# Process multiple files
for file in data/*.json; do
  basename=$(basename "$file" .json)
  python -m utterance_segmentation.cli \
    --input_asr_json "$file" \
    --output_json "output/${basename}_utterances.json" \
    --llm_backend local \
    --local_model_path models/Phi-3-mini-4k-instruct-Q4_K_M.gguf \
    --n_threads 8
done
```

### Example 3: Quick Test Run

```bash
# No installation, no LLM
python standalone_utterance_seg.py \
  --input_asr_json examples/sample_asr.json \
  --output_json test_output.json \
  --no_llm \
  --quiet
```

### Example 4: Custom Parameters

```bash
python -m utterance_segmentation.cli \
  --input_asr_json data/session.json \
  --output_json output/session_utterances.json \
  --llm_backend local \
  --local_model_path models/Phi-3-mini-4k-instruct-Q4_K_M.gguf \
  --turn_break_threshold 2.0 \
  --min_utterance_length 30 \
  --max_utterance_length 100 \
  --max_boundary_candidates 7 \
  --n_threads 16
```

## Troubleshooting

### Common Issues

#### 1. "API key required"

**Problem:** Missing Anthropic API key

**Solution:**
```bash
# Set environment variable
export ANTHROPIC_API_KEY=sk-ant-your-key-here

# Or pass directly
--anthropic_api_key sk-ant-your-key-here
```

#### 2. "Model file not found"

**Problem:** Local model not downloaded

**Solution:**
```bash
# Download model
python download_local_model.py --output_dir models

# Verify path
ls -lh models/
```

#### 3. "llama-cpp-python not installed"

**Problem:** Missing local LLM dependency

**Solution:**
```bash
pip install llama-cpp-python
# Or with package
pip install -e ".[local]"
```

#### 4. Slow local model performance

**Problem:** Not using enough CPU threads

**Solution:**
```bash
# Use more threads
--n_threads 16

# Or auto-detect
--n_threads $(nproc)
```

#### 5. Out of memory errors (local model)

**Problem:** Not enough RAM for model

**Solution:**
- Use smaller model: `--model phi3-q2` (1.4GB vs 2.4GB)
- Reduce context: `--n_ctx 1024`
- Close other applications

#### 6. JSON parsing errors from local model

**Problem:** Local model not producing valid JSON

**Solution:**
- Use better quality model (Q4 vs Q2)
- Reduce `--num_few_shot_examples 2`
- The pipeline has fallback parsing, so partial failures are handled

### Getting Help

1. Check logs: Look at `--quiet` removed output
2. Verify input format matches examples
3. Test with `--no_llm` first to isolate LLM issues
4. Check [HPC documentation](hpc/README_HPC.md) for cluster-specific issues
5. GitHub Issues: Report bugs with example files

## Performance Tips

### For Speed

- Use `--no_llm` for fastest processing
- Reduce `--max_boundary_candidates` (e.g., 3 instead of 5)
- Use fewer `--num_few_shot_examples` (e.g., 2 instead of 3)
- Use Q2 quantized models instead of Q4

### For Quality

- Use Anthropic API
- Provide good `--few_shot_examples`
- Use Q4 quantized models (vs Q2)
- Increase `--max_boundary_candidates` to 7

### For Cost

- Use local GGUF models (free after download)
- Process in batches to amortize model loading time
- Use `--no_llm` for development/testing

### For HPC

- Use local models to avoid network dependencies
- Set `--n_threads $SLURM_CPUS_PER_TASK`
- Stage model files to local node storage
- Use job arrays for parallel processing
