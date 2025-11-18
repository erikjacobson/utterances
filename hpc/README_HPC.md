# Running Utterance Segmentation on IU HPC

This guide covers setup and usage of the semantic utterance segmentation pipeline on Indiana University's High Performance Computing (HPC) systems (Carbonate, Big Red 200, Quartz, etc.).

## Standalone Option (Easiest for HPC)

For maximum simplicity on HPC, use the **standalone single-file version** (`standalone_utterance_seg.py`) which requires minimal setup:

```bash
# 1. Copy standalone script to your HPC directory
cd /N/u/your-username/Carbonate/projects/utterances/hpc

# 2. Install minimal dependencies
module load python
pip install --user pydantic anthropic

# 3. Run directly (no package installation needed!)
python standalone_utterance_seg.py \
  --input_asr_json /path/to/your/asr.json \
  --output_json output/utterances.json \
  --no_llm
```

**Advantages:**
- ✅ No conda environment setup required
- ✅ No package installation
- ✅ Works with system Python + minimal pip packages
- ✅ 850+ lines in a single file
- ✅ Same functionality as full package

You can also use the standalone script in SLURM jobs - just replace the `python -m utterance_segmentation.cli` commands in the example scripts below with `python standalone_utterance_seg.py`.

## Local LLM Option (No API Key Required)

For HPC environments with network restrictions or to avoid API costs, use a **local GGUF model**:

```bash
# 1. Download a local model (one-time setup)
python download_local_model.py --output_dir models

# 2. Install llama-cpp-python
pip install --user llama-cpp-python

# 3. Run with local model (no API key needed!)
python standalone_utterance_seg.py \
  --input_asr_json /path/to/your/asr.json \
  --output_json output/utterances.json \
  --llm_backend local \
  --local_model_path models/Phi-3-mini-4k-instruct-Q4_K_M.gguf \
  --n_threads 16
```

**Local Model Benefits for HPC:**
- ✅ No internet required after model download
- ✅ No API costs
- ✅ Works on compute nodes without external network access
- ✅ CPU-only, works on any node
- ✅ Adjust `--n_threads` based on your SLURM allocation

**Recommended Models:**
- **Phi-3 Mini Q4_K_M** (~2.4GB): Best quality/speed balance
- **Phi-3 Mini Q2_K** (~1.4GB): Faster, lower quality
- **Llama 3.2 3B Q4_K_M** (~1.9GB): Alternative option

Download script automatically fetches from HuggingFace. Run once and reuse the model file for all jobs.

## Quick Start (Full Package)

```bash
# 1. Clone repository and navigate to project
cd /N/u/your-username/Carbonate/projects
git clone <repository-url>
cd utterances

# 2. Run setup script
cd hpc
bash setup_hpc.sh

# 3. Set API key (if using LLM mode)
export ANTHROPIC_API_KEY="sk-ant-your-key-here"

# 4. Edit SLURM script with your file paths
nano run_slurm.sh  # Edit INPUT_ASR path

# 5. Submit job
sbatch run_slurm.sh
```

## Detailed Setup

### 1. Initial Environment Setup

The `setup_hpc.sh` script automates environment creation:

```bash
cd hpc
bash setup_hpc.sh
```

This will:
- Load the anaconda module
- Create a conda environment from `environment.yml`
- Install the utterance-segmentation package
- Display usage instructions

### Manual Setup (Alternative)

If you prefer manual setup:

```bash
# Load conda
module load anaconda

# Create environment
conda env create -f hpc/environment.yml

# Activate environment
conda activate utterance-seg

# Install package
pip install -e .
```

### 2. Configure LLM Access

You have three options for LLM-based semantic segmentation:

**Option A: Anthropic API (Requires API Key)**

Add to your `~/.bashrc`:
```bash
export ANTHROPIC_API_KEY="sk-ant-your-key-here"
```

Or pass directly:
```bash
python -m utterance_segmentation.cli --anthropic_api_key "your-key-here" ...
```

**Option B: Local GGUF Model (No API Key)**

Download model once (do this from a login node with internet):
```bash
python download_local_model.py --output_dir models
pip install --user llama-cpp-python
```

Then use in your scripts:
```bash
python -m utterance_segmentation.cli \
  --llm_backend local \
  --local_model_path models/Phi-3-mini-4k-instruct-Q4_K_M.gguf \
  --n_threads 16 \
  ...
```

**Option C: No LLM (Fastest)**

Use `--no_llm` flag for simple rule-based segmentation.

## Usage Modes

### Mode 1: Single File Processing (with LLM)

Best for: High-quality semantic segmentation of individual files

```bash
# Edit run_slurm.sh with your paths
nano run_slurm.sh

# Submit job
sbatch run_slurm.sh

# Monitor job
squeue -u $USER
```

**Resource Requirements:**
- Time: 1-2 hours per file (depends on length and LLM API speed)
- Memory: 16GB (conservative estimate)
- CPUs: 4 (for I/O and processing)

### Mode 2: Batch Processing (with LLM)

Best for: Processing multiple files in parallel

```bash
# Create file list
ls /path/to/asr/files/*.json > file_list.txt

# Edit array size in run_slurm_batch.sh
# --array=1-N%M where N=total files, M=max simultaneous jobs
nano run_slurm_batch.sh

# Submit batch job
sbatch run_slurm_batch.sh

# Monitor progress
squeue -u $USER
watch -n 10 'squeue -u $USER'
```

**Example for 50 files, 10 at a time:**
```bash
#SBATCH --array=1-50%10
```

**Note on Local LLM for Batch Processing:**

To use local models in batch jobs, modify your SLURM scripts to include:
```bash
python -m utterance_segmentation.cli \
  --llm_backend local \
  --local_model_path /path/to/models/Phi-3-mini-4k-instruct-Q4_K_M.gguf \
  --n_threads $SLURM_CPUS_PER_TASK \
  ...
```

**Resource Requirements for Local LLM:**
- Time: 2-4 hours per file (slower than API but no network required)
- Memory: 8-16GB (model size + working memory)
- CPUs: 8-16 recommended (adjust with --n_threads)

### Mode 3: Local LLM Single File

Best for: High-quality semantic segmentation without API costs or network access

**Prerequisites:**
```bash
# One-time setup (do this from login node with internet)
python download_local_model.py --output_dir models
pip install --user llama-cpp-python
```

**Usage:**
```bash
# Edit run_local_llm.sh with your paths
nano run_local_llm.sh

# Submit job
sbatch run_local_llm.sh

# Monitor job
squeue -u $USER
```

**Resource Requirements:**
- Time: 2-4 hours per file
- Memory: 16GB (model + processing)
- CPUs: 16 (adjustable with --n_threads)

### Mode 4: Local LLM Batch Processing

Best for: Processing many files with no API costs, no network required

**Prerequisites:**
```bash
# One-time setup (same as Mode 3)
python download_local_model.py --output_dir models
pip install --user llama-cpp-python

# Create file list
ls /path/to/asr/files/*.json > file_list.txt
```

**Usage:**
```bash
# Edit array size in run_local_llm_batch.sh
# --array=1-N%M where N=total files, M=max simultaneous jobs
nano run_local_llm_batch.sh

# Submit batch job
sbatch run_local_llm_batch.sh

# Monitor progress
squeue -u $USER
watch -n 10 'squeue -u $USER'
```

**Example for 100 files, 10 at a time:**
```bash
#SBATCH --array=1-100%10
```

**Resource Requirements:**
- Time: 2-4 hours per file
- Memory: 16GB per job
- CPUs: 16 per job
- No API rate limits!

### Mode 5: Simple Mode (no LLM)

Best for: Fast processing without semantic analysis, no API key needed

```bash
# Edit run_no_llm.sh with your paths
nano run_no_llm.sh

# Submit job
sbatch run_no_llm.sh
```

**Resource Requirements:**
- Time: 5-15 minutes per file
- Memory: 8GB
- CPUs: 2

## Directory Structure on HPC

Recommended project organization:

```
/N/u/your-username/Carbonate/projects/utterances/
├── hpc/                           # HPC scripts (this directory)
│   ├── environment.yml
│   ├── setup_hpc.sh
│   ├── run_slurm.sh              # Single file with Anthropic API
│   ├── run_slurm_batch.sh        # Batch with Anthropic API
│   ├── run_local_llm.sh          # Single file with local model
│   ├── run_local_llm_batch.sh    # Batch with local model
│   ├── run_no_llm.sh             # Simple mode (no LLM)
│   ├── standalone_utterance_seg.py
│   └── README_HPC.md
├── download_local_model.py        # Model downloader script
├── models/                        # Local GGUF models (after download)
│   └── Phi-3-mini-4k-instruct-Q4_K_M.gguf
├── examples/                      # Sample data
│   ├── sample_asr.json
│   └── few_shot_boundaries.json
├── data/                          # Your ASR input files
│   └── *.json
├── output/                        # Generated utterance files
│   └── *.json
├── logs/                          # SLURM job logs
│   ├── utterance_seg_*.out
│   ├── utterance_seg_*.err
│   ├── utterance_seg_local_*.out
│   └── utterance_seg_local_*.err
└── file_list.txt                  # File list for batch processing
```

## Job Monitoring

### Check Job Status

```bash
# View your jobs
squeue -u $USER

# Detailed job info
scontrol show job <job-id>

# View job array details
squeue -u $USER --array
```

### Check Job Output

```bash
# View output log (updates in real-time)
tail -f logs/utterance_seg_<job-id>.out

# View error log
tail -f logs/utterance_seg_<job-id>.err

# For batch jobs
tail -f logs/utterance_seg_batch_<job-id>_<task-id>.out
```

### Cancel Jobs

```bash
# Cancel specific job
scancel <job-id>

# Cancel all your jobs
scancel -u $USER

# Cancel specific job array tasks
scancel <job-id>_[1-5]  # Cancel tasks 1-5
```

## Resource Guidelines

### Choosing Partition

IU HPC partitions (adjust based on your system):

- `general`: Standard jobs (default, max 7 days)
- `debug`: Quick testing (max 1 hour, high priority)
- `gpu`: GPU jobs (not needed for this pipeline)

```bash
#SBATCH --partition=general
```

### Memory Allocation

Estimate based on ASR file size:

| ASR File Words | Recommended Memory |
|----------------|-------------------|
| < 10,000       | 8GB              |
| 10,000-50,000  | 16GB             |
| 50,000-100,000 | 32GB             |
| > 100,000      | 64GB             |

### Time Limits

**With LLM (--array jobs):**
- LLM API latency varies (typically 0.5-2s per boundary query)
- Estimate: ~1-2 hours per 20-minute conversation
- Set generous time limits: `#SBATCH --time=04:00:00`

**Without LLM (--no_llm):**
- Much faster, deterministic
- Estimate: ~5-15 minutes per file
- Time limit: `#SBATCH --time=00:30:00`

## Troubleshooting

### Common Issues

**1. Module not found**

```bash
# List available modules
module avail

# Try different anaconda versions
module load anaconda/2023.09
```

**2. Conda environment not found**

```bash
# List environments
conda env list

# Recreate environment
conda env remove -n utterance-seg
bash setup_hpc.sh
```

**3. Out of memory errors**

Increase memory in SLURM script:
```bash
#SBATCH --mem=32G  # or higher
```

**4. API rate limiting**

Reduce concurrent jobs in batch processing:
```bash
#SBATCH --array=1-50%5  # Max 5 simultaneous instead of 10
```

**5. Job timeout**

Increase time limit:
```bash
#SBATCH --time=08:00:00  # 8 hours
```

### Testing Before Batch Submission

Test with a single file first:

```bash
# Interactive session
srun -p general --pty bash

# Load environment
module load anaconda
conda activate utterance-seg

# Test command
python -m utterance_segmentation.cli \
  --input_asr_json examples/sample_asr.json \
  --output_json test_output.json \
  --no_llm
```

## Performance Optimization

### 1. Optimal Batch Size

For job arrays, balance between:
- Total throughput (more parallel jobs = faster overall)
- API rate limits (fewer concurrent jobs = more reliable)
- Cluster load (be a good citizen)

**Recommended:**
```bash
#SBATCH --array=1-100%10  # 100 files, max 10 simultaneous
```

### 2. File Staging

For large datasets, stage to local node storage:

```bash
# In your SLURM script
LOCAL_DIR="/tmp/$SLURM_JOB_ID"
mkdir -p "$LOCAL_DIR"

# Copy input
cp "$INPUT_ASR" "$LOCAL_DIR/"

# Process from local storage
python -m utterance_segmentation.cli \
  --input_asr_json "$LOCAL_DIR/$(basename $INPUT_ASR)" \
  --output_json "$LOCAL_DIR/output.json"

# Copy output back
cp "$LOCAL_DIR/output.json" "$OUTPUT_FILE"

# Cleanup
rm -rf "$LOCAL_DIR"
```

### 3. Reduce LLM Calls

Adjust parameters to reduce API calls:

```bash
python -m utterance_segmentation.cli \
  --max_boundary_candidates 3 \     # Default is 5
  --num_few_shot_examples 2 \       # Default is 3
  ...
```

## Cost Estimation

### LLM API Costs (Anthropic Claude)

Approximate costs per file (20-minute conversation, ~3000 words):
- Boundary queries: ~10-20 queries
- Tokens per query: ~200-400 input, ~50 output
- Cost: ~$0.05-0.10 per file

For batch processing:
- 100 files: ~$5-10
- 1000 files: ~$50-100

### HPC Compute Units

Check your allocation:
```bash
mybalance
```

Typical usage:
- 1 file (2 hours, 4 CPUs): ~8 CPU-hours
- 100 files in parallel (4 hours, 400 CPUs): ~1600 CPU-hours

## Additional Resources

- **IU HPC Documentation**: https://kb.iu.edu/d/alde
- **SLURM Documentation**: https://slurm.schedmd.com/
- **Carbonate User Guide**: https://kb.iu.edu/d/aolp
- **Project GitHub**: [Add your repository URL]

## Support

For IU HPC-specific issues:
- Email: hpc@iu.edu
- Knowledge Base: https://kb.iu.edu

For pipeline issues:
- GitHub Issues: [Add your repository URL]
- Check logs in `logs/` directory
