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

```bash
python -m utterance_segmentation.cli \
  --input_asr_json path/to/asr.json \
  --few_shot_examples path/to/examples.json \
  --output_json path/to/output.json
```

## Project Structure

```
src/utterance_segmentation/
  ├── models.py          # Core data structures
  ├── io.py              # I/O helpers for ASR JSON
  ├── turns.py           # Turn construction
  ├── spans.py           # Candidate span generation
  ├── llm_boundaries.py  # LLM boundary classification
  ├── utterances.py      # Final utterance construction
  └── cli.py             # Command-line interface

tests/                   # Unit tests
```

## Development Phases

- [x] Phase 1: Project scaffolding and core data structures
- [ ] Phase 2: Turn construction and candidate span generation
- [ ] Phase 3: LLM boundary classifier interface
- [ ] Phase 4: Span-level boundary selection
- [ ] Phase 5: End-to-end CLI + diagnostics
