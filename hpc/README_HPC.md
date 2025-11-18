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
- ✅ Works with system Python + 2 pip packages
- ✅ 730 lines in a single file
- ✅ Same functionality as full package

You can also use the standalone script in SLURM jobs - just replace the `python -m utterance_segmentation.cli` commands in the example scripts below with `python standalone_utterance_seg.py`.

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

### 2. Configure API Access (LLM Mode Only)

For LLM-based semantic segmentation, you need an Anthropic API key.

**Option 1: Environment Variable (Recommended)**

Add to your `~/.bashrc`:
```bash
export ANTHROPIC_API_KEY="sk-ant-your-key-here"
```

**Option 2: SLURM Script**

Uncomment and edit the line in the SLURM script:
```bash
export ANTHROPIC_API_KEY="your-key-here"
```

**Option 3: Command-line Argument**

Pass directly to the CLI:
```bash
python -m utterance_segmentation.cli \
  --anthropic_api_key "your-key-here" \
  ...
```

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

### Mode 3: Simple Mode (no LLM)

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
│   ├── run_slurm.sh
│   ├── run_slurm_batch.sh
│   ├── run_no_llm.sh
│   └── README_HPC.md
├── examples/                      # Sample data
│   ├── sample_asr.json
│   └── few_shot_boundaries.json
├── data/                          # Your ASR input files
│   └── *.json
├── output/                        # Generated utterance files
│   └── *.json
├── logs/                          # SLURM job logs
│   ├── utterance_seg_*.out
│   └── utterance_seg_*.err
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
