# CLAUDE.md - AI Assistant Guide for Utterance Segmentation Pipeline

**Last Updated:** 2025-11-18
**Version:** 1.0.0

This document provides comprehensive guidance for AI assistants (like Claude) working on the utterance segmentation pipeline codebase.

---

## Table of Contents

- [Overview](#overview)
- [Codebase Structure](#codebase-structure)
- [Architecture & Design Principles](#architecture--design-principles)
- [Development Workflow](#development-workflow)
- [Key Conventions](#key-conventions)
- [Testing Strategy](#testing-strategy)
- [LLM Backend System](#llm-backend-system)
- [HPC Deployment](#hpc-deployment)
- [Common Tasks](#common-tasks)
- [Troubleshooting](#troubleshooting)

---

## Overview

**Purpose:** Semantic utterance segmentation for ASR (Automatic Speech Recognition) output with speaker diarization.

**Core Functionality:**
- Processes word-level ASR + diarization JSON
- Groups words into speaker turns based on speaker changes and pauses
- Creates candidate spans (40-80 words)
- Uses LLM-guided boundary detection for semantic coherence
- Outputs semantically coherent utterances

**Key Features:**
- Three LLM backends: Anthropic API, Local GGUF models, No LLM
- Standalone single-file version (850+ lines, CPU-only)
- HPC/SLURM batch processing support
- Comprehensive test coverage (96 tests)

---

## Codebase Structure

### Directory Layout

```
utterances/
├── src/utterance_segmentation/     # Main package
│   ├── models.py                   # Pydantic data models (Word, Turn, Utterance, etc.)
│   ├── io.py                       # I/O helpers for JSON loading/saving
│   ├── turns.py                    # Turn construction from speaker changes/pauses
│   ├── spans.py                    # Candidate span generation (40-80 words)
│   ├── llm_boundaries.py           # LLM boundary classification (3 backends)
│   ├── utterances.py               # Final utterance construction
│   ├── cli.py                      # Command-line interface
│   └── __init__.py                 # Package initialization
│
├── tests/                          # Unit tests (mirrors src/ structure)
│   ├── test_models.py              # Test Pydantic models & validation
│   ├── test_io.py                  # Test JSON I/O
│   ├── test_turns.py               # Test turn construction
│   ├── test_spans.py               # Test span generation
│   ├── test_llm_boundaries.py      # Test LLM integration
│   ├── test_utterances.py          # Test utterance construction
│   └── __init__.py
│
├── examples/                       # Sample data
│   ├── sample_asr.json             # 36-word sample conversation
│   └── few_shot_boundaries.json    # 5 boundary decision examples
│
├── hpc/                            # HPC/SLURM deployment
│   ├── environment.yml             # Conda environment spec
│   ├── setup_hpc.sh                # Automated setup script
│   ├── run_slurm.sh                # Single file with Anthropic API
│   ├── run_slurm_batch.sh          # Batch with Anthropic API
│   ├── run_local_llm.sh            # Single file with local model
│   ├── run_local_llm_batch.sh      # Batch with local model
│   ├── run_no_llm.sh               # Simple mode (no LLM)
│   ├── standalone_utterance_seg.py # Standalone script copy
│   └── README_HPC.md               # HPC usage guide
│
├── standalone_utterance_seg.py     # Single-file version (850+ lines)
├── download_local_model.py         # GGUF model downloader
├── pyproject.toml                  # Package configuration & dependencies
├── README.md                       # User-facing documentation
├── USAGE.md                        # Comprehensive usage guide
└── CLAUDE.md                       # This file (AI assistant guide)
```

### Module Responsibilities

| Module | Purpose | Key Classes/Functions | Dependencies |
|--------|---------|----------------------|--------------|
| **models.py** | Data structures | `Word`, `Turn`, `CandidateSpan`, `Utterance`, `ASRDocument`, `UtteranceDocument` | `pydantic` |
| **io.py** | File operations | `load_asr_json()`, `save_utterance_json()`, `load_few_shot_examples()` | `models.py` |
| **turns.py** | Turn segmentation | `build_turns(words, threshold=1.5)` | `models.py` |
| **spans.py** | Span generation | `build_candidate_spans()`, `build_candidate_spans_for_all_turns()` | `models.py` |
| **llm_boundaries.py** | LLM integration | `BoundaryClassifierBase`, `LLMBoundaryClassifier`, `LocalLLMBoundaryClassifier`, `create_classifier()` | `anthropic`, `llama-cpp-python` (optional) |
| **utterances.py** | Utterance construction | `build_utterances()`, `build_utterances_simple()`, `evaluate_span_boundaries()` | All above |
| **cli.py** | CLI interface | `main()`, `print_diagnostics()` | All above |

---

## Architecture & Design Principles

### 1. **Pipeline Architecture**

The system follows a linear pipeline with clearly separated stages:

```
ASR JSON Input
    ↓
[1. Load & Validate]  (io.py, models.py)
    ↓
[2. Build Turns]      (turns.py)
    ↓
[3. Generate Spans]   (spans.py)
    ↓
[4. LLM Boundaries]   (llm_boundaries.py) - OPTIONAL
    ↓
[5. Build Utterances] (utterances.py)
    ↓
[6. Save Output]      (io.py)
    ↓
Utterance JSON Output
```

**Key Principle:** Each stage is independent and testable. Outputs of one stage are inputs to the next.

### 2. **Design Patterns Used**

#### Strategy Pattern (LLM Backends)
```python
# Abstract base class
class BoundaryClassifierBase(ABC):
    @abstractmethod
    def query_boundary(self, prompt: str, max_tokens: int) -> Dict[str, Any]:
        pass

# Concrete implementations
class LLMBoundaryClassifier(BoundaryClassifierBase):         # Anthropic API
class LocalLLMBoundaryClassifier(BoundaryClassifierBase):    # Local GGUF
```

#### Factory Pattern (Classifier Creation)
```python
def create_classifier(backend: str = "anthropic", ...) -> BoundaryClassifierBase:
    if backend == "anthropic":
        return LLMBoundaryClassifier(...)
    elif backend == "local":
        return LocalLLMBoundaryClassifier(...)
```

#### Builder Pattern (Utterance Construction)
Incremental building of complex utterances from simple components (words → turns → spans → utterances).

### 3. **Data Validation Strategy**

**All data structures use Pydantic models** for automatic validation:
- Type checking at runtime
- Field validation (e.g., `start < end`, `times >= 0`)
- Immutable data structures (prevents accidental mutation)

**Example:**
```python
class Word(BaseModel):
    index: int
    text: str
    start: float
    end: float
    speaker: str
    confidence: Optional[float] = None

    @field_validator('start', 'end')
    @classmethod
    def validate_times(cls, v: float) -> float:
        if v < 0:
            raise ValueError(f"Time must be non-negative, got {v}")
        return v
```

### 4. **Error Handling Philosophy**

- **Fail early:** Validate inputs at entry points
- **Explicit exceptions:** Use specific exception types (`ValueError`, `FileNotFoundError`, etc.)
- **Graceful degradation:** LLM failures should not crash the pipeline
- **User-friendly messages:** CLI provides clear error messages with solutions

---

## Development Workflow

### Git Branching Strategy

**Current branch:** `claude/semantic-utterance-segmentation-01CRtK8swJ4fQcjoTHNmndEL`

**Naming convention:**
- Feature branches: `claude/feature-name-{session-id}`
- Always push to the specified branch
- Never push to main/master without explicit permission

### Commit Guidelines

**Format:**
```
Title: Brief description (50 chars or less)

Detailed explanation of changes:
- What was changed
- Why it was changed
- How it affects other components

Key highlights:
- Feature 1
- Feature 2

Files modified:
- path/to/file1.py: Description
- path/to/file2.py: Description
```

**Example:**
```
Add local GGUF model support for CPU-only inference

Implements Phase 1 of local LLM support as alternative to Anthropic API.
This enables fully offline operation on HPC clusters and removes API costs.

Features:
- LocalLLMBoundaryClassifier class using llama-cpp-python
- create_classifier() factory function
- CLI arguments: --llm_backend, --local_model_path

Changes:
- src/utterance_segmentation/llm_boundaries.py: Add LocalLLMBoundaryClassifier
- src/utterance_segmentation/cli.py: Add local LLM arguments
```

### Testing Workflow

**Always run tests before committing:**
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_llm_boundaries.py

# Run with coverage
pytest --cov=src/utterance_segmentation
```

**Test requirements:**
- All new functions must have unit tests
- Maintain >90% code coverage
- Tests must be independent (no shared state)
- Use descriptive test names: `test_build_turns_with_speaker_change()`

### Code Review Checklist

Before committing, verify:
- [ ] All tests pass (`pytest`)
- [ ] New code has unit tests
- [ ] Docstrings added for public functions
- [ ] Type hints used consistently
- [ ] Error handling implemented
- [ ] No hardcoded paths or secrets
- [ ] Documentation updated (README, USAGE)
- [ ] Standalone script updated if package changed

---

## Key Conventions

### 1. **Code Style**

**Python Style:**
- Follow PEP 8
- Line length: 100 characters (flexible to 120 for readability)
- Use type hints for all function signatures
- Docstrings: Google style

**Example:**
```python
def build_turns(
    words: List[Word],
    turn_break_threshold: float = 1.5
) -> List[Turn]:
    """
    Group words into speaker turns based on speaker changes and pauses.

    Args:
        words: List of Word objects with timing and speaker info
        turn_break_threshold: Pause duration (seconds) to trigger turn break

    Returns:
        List of Turn objects, each representing a contiguous speaker segment

    Raises:
        ValueError: If words list is empty or contains invalid data
    """
    pass
```

### 2. **Naming Conventions**

| Type | Convention | Example |
|------|------------|---------|
| **Modules** | `lowercase_with_underscores` | `llm_boundaries.py` |
| **Classes** | `PascalCase` | `BoundaryClassifierBase` |
| **Functions** | `lowercase_with_underscores` | `build_candidate_spans()` |
| **Constants** | `UPPERCASE_WITH_UNDERSCORES` | `DEFAULT_THRESHOLD = 1.5` |
| **Private** | `_leading_underscore` | `_parse_response()` |
| **Variables** | `lowercase_with_underscores` | `span_word_indices` |

### 3. **Import Organization**

**Standard order:**
```python
# 1. Standard library
import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

# 2. Third-party packages
from pydantic import BaseModel, Field
from anthropic import Anthropic

# 3. Local imports (relative)
from .models import Word, Turn, Utterance
from .io import load_asr_json
```

### 4. **Configuration Management**

**Default values:**
- Defined as constants at module level
- CLI arguments override defaults
- Never hardcode magic numbers in logic

**Example:**
```python
# At module level
DEFAULT_TURN_BREAK_THRESHOLD = 1.5
DEFAULT_MIN_UTTERANCE_LENGTH = 40
DEFAULT_MAX_UTTERANCE_LENGTH = 80

# In function
def build_turns(words: List[Word], turn_break_threshold: float = DEFAULT_TURN_BREAK_THRESHOLD):
    pass
```

### 5. **Documentation Standards**

**Every public function needs:**
- One-line summary
- Args description with types
- Returns description with type
- Raises (if applicable)
- Example usage (for complex functions)

**Module-level docstrings:**
```python
"""Turn construction from ASR words.

This module handles speaker turn segmentation based on:
- Speaker changes
- Pause duration thresholds
- Temporal ordering

Typical usage:
    words = load_asr_json("input.json").words
    turns = build_turns(words, turn_break_threshold=1.5)
"""
```

---

## Testing Strategy

### Test Organization

**Structure mirrors source:**
```
src/utterance_segmentation/models.py  →  tests/test_models.py
src/utterance_segmentation/turns.py   →  tests/test_turns.py
```

### Test Coverage Requirements

**Current coverage: 96 tests, >90% coverage**

**Required tests for each module:**

1. **Happy path tests** - Normal, expected inputs
2. **Edge case tests** - Boundary conditions (empty lists, single item, max values)
3. **Error handling tests** - Invalid inputs, exceptions
4. **Integration tests** - Module interactions

### Test Naming Convention

```python
def test_<function_name>_<scenario>_<expected_result>():
    """Test that <function> <does what> when <scenario>."""
    pass
```

**Examples:**
```python
def test_build_turns_with_speaker_change():
    """Test that build_turns creates separate turns on speaker change."""

def test_build_turns_with_pause_threshold():
    """Test that build_turns creates turn break on long pause."""

def test_build_turns_raises_error_on_empty_words():
    """Test that build_turns raises ValueError for empty word list."""
```

### Test Structure (AAA Pattern)

```python
def test_example():
    # Arrange - Setup test data
    words = [
        Word(index=0, text="hello", start=0.0, end=0.5, speaker="S1"),
        Word(index=1, text="world", start=0.5, end=1.0, speaker="S1"),
    ]

    # Act - Execute the function
    result = build_turns(words, turn_break_threshold=1.5)

    # Assert - Verify expected outcomes
    assert len(result) == 1
    assert result[0].speaker == "S1"
    assert result[0].num_words == 2
```

### Mocking LLM Calls

**Always mock external API calls in tests:**

```python
from unittest.mock import Mock, patch

def test_llm_boundary_classifier():
    # Arrange
    mock_response = {
        "break_here": True,
        "left_is_complete": True,
        "right_is_new_or_complete": True,
        "reason": "Clear topic change"
    }

    classifier = LLMBoundaryClassifier(api_key="test-key")

    # Mock the API call
    with patch.object(classifier.client.messages, 'create') as mock_create:
        mock_create.return_value.content = [Mock(text=json.dumps(mock_response))]

        # Act
        result = classifier.query_boundary("test prompt")

        # Assert
        assert result["break_here"] is True
```

---

## LLM Backend System

### Three-Backend Architecture

| Backend | Class | Use Case | Dependencies |
|---------|-------|----------|--------------|
| **Anthropic API** | `LLMBoundaryClassifier` | Production, highest quality | `anthropic` |
| **Local GGUF** | `LocalLLMBoundaryClassifier` | HPC, offline, no API cost | `llama-cpp-python` |
| **No LLM** | None (simple logic) | Testing, development | None |

### Adding New LLM Backend

**To add a new backend (e.g., OpenAI):**

1. **Create new class implementing `BoundaryClassifierBase`:**

```python
class OpenAIBoundaryClassifier(BoundaryClassifierBase):
    def __init__(self, api_key: Optional[str] = None):
        import openai
        self.client = openai.OpenAI(api_key=api_key)

    def query_boundary(self, prompt: str, max_tokens: int = 200) -> Dict[str, Any]:
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens
        )
        return json.loads(response.choices[0].message.content)
```

2. **Update `create_classifier()` factory:**

```python
def create_classifier(
    backend: str = "anthropic",
    anthropic_api_key: Optional[str] = None,
    openai_api_key: Optional[str] = None,  # NEW
    ...
) -> BoundaryClassifierBase:
    if backend == "anthropic":
        return LLMBoundaryClassifier(api_key=anthropic_api_key)
    elif backend == "openai":  # NEW
        return OpenAIBoundaryClassifier(api_key=openai_api_key)
    elif backend == "local":
        return LocalLLMBoundaryClassifier(...)
```

3. **Update CLI arguments:**

```python
parser.add_argument(
    "--llm_backend",
    choices=["anthropic", "local", "openai"],  # Add new backend
    default="anthropic",
    help="LLM backend to use"
)
parser.add_argument("--openai_api_key", help="OpenAI API key")  # NEW
```

4. **Add tests:**

```python
def test_openai_boundary_classifier():
    classifier = OpenAIBoundaryClassifier(api_key="test-key")
    # Test implementation
```

5. **Update documentation:**
   - README.md - Add usage example
   - USAGE.md - Add to comparison table
   - pyproject.toml - Add optional dependency

### LLM Response Format

**All LLM backends must return this structure:**

```json
{
  "break_here": true,           // bool - Should we place a boundary?
  "left_is_complete": true,     // bool - Is left text complete?
  "right_is_new_or_complete": true,  // bool - Is right text new topic?
  "reason": "explanation"       // str - Why this decision?
}
```

**Error handling:**
- If JSON parsing fails, provide fallback logic
- Never crash on bad LLM responses
- Log warnings for debugging

---

## HPC Deployment

### SLURM Script Structure

**All SLURM scripts follow this pattern:**

```bash
#!/bin/bash
#SBATCH --job-name=...
#SBATCH --output=logs/...
#SBATCH --time=...
#SBATCH --cpus-per-task=...
#SBATCH --mem=...

# 1. Error handling
set -e

# 2. Environment setup
module load anaconda
source activate utterance-seg

# 3. Path configuration
INPUT_ASR="/path/to/input.json"
OUTPUT_JSON="output/result.json"

# 4. Validation
if [ ! -f "$INPUT_ASR" ]; then
    echo "Error: Input file not found"
    exit 1
fi

# 5. Execution
python -m utterance_segmentation.cli \
  --input_asr_json "$INPUT_ASR" \
  --output_json "$OUTPUT_JSON" \
  ...

# 6. Completion message
echo "Job completed: $(date)"
```

### Resource Allocation Guidelines

| Mode | Time | Memory | CPUs | Best For |
|------|------|--------|------|----------|
| **Anthropic API** | 1-2h | 16GB | 4 | Small batches, highest quality |
| **Local GGUF** | 2-4h | 16GB | 16 | Large batches, no API cost |
| **No LLM** | 5-15min | 8GB | 2 | Fast processing, testing |

### HPC Best Practices

1. **Always verify input files exist** before processing
2. **Use `$SLURM_CPUS_PER_TASK`** for `--n_threads`
3. **Create logs directory** before job submission
4. **Use job arrays** for batch processing
5. **Stage large model files** to local node storage

---

## Common Tasks

### Task 1: Adding New CLI Argument

**Steps:**

1. Add argument to `cli.py`:
```python
parser.add_argument(
    "--new_option",
    type=str,
    default="default_value",
    help="Description of the option"
)
```

2. Pass to relevant function:
```python
result = some_function(arg1, arg2, new_option=args.new_option)
```

3. Update help examples in epilog

4. Add to `USAGE.md` documentation

5. Test with `python -m utterance_segmentation.cli --help`

### Task 2: Modifying Pydantic Models

**Always consider backward compatibility!**

1. Make changes in `models.py`:
```python
class Word(BaseModel):
    # Existing fields
    index: int
    text: str

    # New optional field (backward compatible)
    new_field: Optional[str] = None
```

2. Update validation if needed:
```python
@field_validator('new_field')
@classmethod
def validate_new_field(cls, v):
    # Validation logic
    return v
```

3. Add tests in `test_models.py`

4. Update example JSON files in `examples/`

5. Update I/O functions if serialization changes

### Task 3: Adding New Feature

**Follow phased approach:**

1. **Phase 1: Design & Plan**
   - Write design doc or update CLAUDE.md
   - Identify affected modules
   - Plan tests

2. **Phase 2: Core Implementation**
   - Implement in appropriate module(s)
   - Add unit tests
   - Verify all tests pass

3. **Phase 3: Integration**
   - Update CLI if needed
   - Update standalone script
   - Test end-to-end

4. **Phase 4: Documentation**
   - Update README.md
   - Update USAGE.md
   - Update HPC scripts if relevant
   - Update this file (CLAUDE.md)

5. **Phase 5: Commit & Push**
   - Write comprehensive commit message
   - Push to correct branch

---

## Troubleshooting

### Common Issues for AI Assistants

#### Issue 1: Tests Failing After Code Changes

**Symptoms:** `pytest` shows failures in unrelated tests

**Causes:**
- Shared mutable state between tests
- Implicit dependencies on execution order
- Changed function signatures

**Solutions:**
1. Check if fixtures are properly isolated
2. Verify no global state modification
3. Update all call sites when changing signatures
4. Run tests individually to isolate issue

#### Issue 2: Import Errors

**Symptoms:** `ModuleNotFoundError` or `ImportError`

**Causes:**
- Package not installed in editable mode
- Circular imports
- Missing `__init__.py`

**Solutions:**
1. Install package: `pip install -e .`
2. Check import graph for cycles
3. Use relative imports within package: `from .models import Word`

#### Issue 3: Standalone Script Out of Sync

**Symptoms:** Feature works in package but not standalone

**Causes:**
- Forgot to update standalone_utterance_seg.py
- Changes in package structure

**Solutions:**
1. Always update both versions when modifying core logic
2. Keep standalone script in `hpc/` directory synced
3. Test standalone version separately

#### Issue 4: LLM Response Parsing Errors

**Symptoms:** JSON parsing failures, missing fields

**Causes:**
- Local model produces non-JSON text
- API response format changed
- Network issues

**Solutions:**
1. Implement robust JSON extraction with regex fallback
2. Provide default values for missing fields
3. Add try-except with graceful degradation
4. Log actual response for debugging

---

## Quick Reference Commands

### Development Commands

```bash
# Install package (editable mode)
pip install -e .

# Install with all dependencies
pip install -e ".[local,dev]"

# Run all tests
pytest

# Run tests with coverage
pytest --cov=src/utterance_segmentation

# Run specific test file
pytest tests/test_llm_boundaries.py

# Run specific test
pytest tests/test_llm_boundaries.py::test_create_classifier

# Format code (if using black)
black src/ tests/

# Type checking (if using mypy)
mypy src/utterance_segmentation
```

### Git Commands

```bash
# Check current branch
git branch --show-current

# Stage all changes
git add -A

# Commit with message
git commit -m "Your message"

# Push to feature branch
git push -u origin claude/feature-name-{session-id}

# Check status
git status

# View changes
git diff

# View log
git log --oneline -10
```

### HPC Commands

```bash
# Submit job
sbatch run_slurm.sh

# Check job status
squeue -u $USER

# Cancel job
scancel <job_id>

# View job output
cat logs/utterance_seg_<job_id>.out

# Monitor job in real-time
tail -f logs/utterance_seg_<job_id>.out
```

---

## Changelog

### Version 1.0.0 (2025-11-18)

**Initial Creation:**
- Complete codebase documentation
- Architecture overview
- Development workflows
- Testing strategy
- LLM backend system
- HPC deployment guide
- Common tasks reference
- Troubleshooting section

**Key Features Documented:**
- Three LLM backends (Anthropic, Local GGUF, No LLM)
- Standalone single-file version
- HPC/SLURM batch processing (5 modes)
- Comprehensive test coverage (96 tests)
- Pydantic-based data validation
- Pipeline architecture

---

## Questions & Support

**For AI Assistants:**
- Always check this file before making changes
- Follow established patterns and conventions
- Update documentation when adding features
- Run tests before committing
- Ask user for clarification if requirements unclear

**For Users:**
- See README.md for usage documentation
- See USAGE.md for comprehensive examples
- See hpc/README_HPC.md for HPC-specific guidance
- Report issues on GitHub with example files

---

**End of CLAUDE.md**
