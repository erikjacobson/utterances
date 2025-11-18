# Utterance Segmentation Pipeline

Semantic utterance segmentation for ASR output with speaker diarization.

## Overview

This pipeline processes word-level ASR + diarization JSON and produces semantically coherent utterances using:
- Deterministic turn segmentation based on speaker changes and pauses
- LLM-guided boundary detection for semantic coherence
- Target utterance length of 40-80 words with flexibility for complete ideas

## Installation

```bash
pip install -e .
```

For development:
```bash
pip install -e ".[dev]"
```

## Usage

### Basic Usage (with LLM)

```bash
export ANTHROPIC_API_KEY=sk-ant-...

python -m utterance_segmentation.cli \
  --input_asr_json path/to/asr.json \
  --few_shot_examples path/to/examples.json \
  --output_json path/to/output.json
```

### Simple Mode (without LLM)

```bash
python -m utterance_segmentation.cli \
  --input_asr_json path/to/asr.json \
  --output_json path/to/output.json \
  --no_llm
```

### Try it with sample data

```bash
python -m utterance_segmentation.cli \
  --input_asr_json examples/sample_asr.json \
  --output_json output.json \
  --no_llm
```

### CLI Options

- `--input_asr_json`: Path to input ASR JSON (required)
- `--output_json`: Path to output utterances JSON (required)
- `--few_shot_examples`: Path to few-shot examples JSON (optional)
- `--anthropic_api_key`: Anthropic API key (or use ANTHROPIC_API_KEY env var)
- `--no_llm`: Skip LLM and use simple span-based segmentation
- `--turn_break_threshold`: Pause threshold for turn breaks in seconds (default: 1.5)
- `--min_utterance_length`: Minimum target length in words (default: 40)
- `--max_utterance_length`: Maximum target length in words (default: 80)
- `--max_boundary_candidates`: Max candidates per span (default: 5)
- `--num_few_shot_examples`: Number of examples in prompts (default: 3)
- `--quiet`: Suppress diagnostic output

## Input Format

ASR JSON must follow WhisperX-style format:

```json
{
  "audio": {
    "path": "session.wav",
    "duration": 3600.0
  },
  "words": [
    {
      "index": 0,
      "text": "hello",
      "start": 1.0,
      "end": 1.5,
      "speaker": "S1",
      "confidence": 0.95
    }
  ]
}
```

## Output Format

```json
{
  "audio": {...},
  "words": [...],
  "utterances": [
    {
      "id": "utt_0000",
      "speaker": "S1",
      "start": 1.0,
      "end": 10.5,
      "word_indices": [0, 1, 2, ...],
      "text": "hello world this is a test",
      "num_words": 6,
      "source_turn_id": "turn_0000"
    }
  ]
}
```

## Running on HPC (SLURM)

For batch processing on HPC systems (Indiana University's Carbonate, Big Red 200, etc.), see **[HPC Documentation](hpc/README_HPC.md)**.

Quick start:
```bash
# Setup conda environment
cd hpc
bash setup_hpc.sh

# Submit single job
sbatch run_slurm.sh

# Submit batch processing (multiple files)
sbatch run_slurm_batch.sh

# Fast processing without LLM
sbatch run_no_llm.sh
```

The `hpc/` directory includes:
- `environment.yml` - Conda environment specification
- `setup_hpc.sh` - Automated setup script
- `run_slurm.sh` - Single file SLURM job
- `run_slurm_batch.sh` - Batch processing with job arrays
- `run_no_llm.sh` - Fast processing without LLM
- `README_HPC.md` - Comprehensive HPC usage guide

## Testing

Run all tests:
```bash
pytest tests/ -v
```

All 96 tests passing!

## Project Structure

```
src/utterance_segmentation/
  ├── models.py          # Core data structures (Word, Turn, Utterance, etc.)
  ├── io.py              # I/O helpers for ASR JSON
  ├── turns.py           # Turn construction from speaker/pauses
  ├── spans.py           # Candidate span generation (40-80 words)
  ├── llm_boundaries.py  # LLM boundary classification (Claude API)
  ├── utterances.py      # Final utterance construction
  └── cli.py             # Command-line interface

tests/                   # Comprehensive unit tests (96 tests)
examples/                # Sample data files
hpc/                     # HPC/SLURM batch processing scripts
  ├── environment.yml    # Conda environment
  ├── setup_hpc.sh       # Setup script
  ├── run_slurm.sh       # Single file job
  ├── run_slurm_batch.sh # Batch processing
  ├── run_no_llm.sh      # Fast mode
  └── README_HPC.md      # HPC usage guide
```

## Development Phases

- [x] Phase 1: Project scaffolding and core data structures
- [x] Phase 2: Turn construction and candidate span generation
- [x] Phase 3: LLM boundary classifier interface
- [x] Phase 4: Span-level boundary selection
- [x] Phase 5: End-to-end CLI + diagnostics

**All phases complete!** The pipeline is production-ready.
