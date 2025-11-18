"""I/O helpers for loading and saving ASR JSON files."""

import json
from pathlib import Path
from typing import Union

from .models import ASRDocument, UtteranceDocument


def load_asr_json(path: Union[str, Path]) -> ASRDocument:
    """
    Load ASR JSON file and validate structure.

    Args:
        path: Path to ASR JSON file

    Returns:
        Validated ASRDocument

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If JSON structure is invalid
        ValidationError: If data doesn't match schema
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"ASR file not found: {path}")

    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Validate and parse using Pydantic
    return ASRDocument.model_validate(data)


def save_utterance_json(doc: UtteranceDocument, path: Union[str, Path]) -> None:
    """
    Save utterance document to JSON file.

    Args:
        doc: UtteranceDocument to save
        path: Output path for JSON file
    """
    path = Path(path)

    # Create parent directory if needed
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to dict and save
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(
            doc.model_dump(mode='python'),
            f,
            indent=2,
            ensure_ascii=False
        )


def load_few_shot_examples(path: Union[str, Path]) -> dict:
    """
    Load few-shot examples JSON for LLM prompts.

    Args:
        path: Path to examples JSON file

    Returns:
        Dict with 'examples' list

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If JSON structure is invalid
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Examples file not found: {path}")

    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Basic validation
    if 'examples' not in data:
        raise ValueError("Examples JSON must contain 'examples' list")

    if not isinstance(data['examples'], list):
        raise ValueError("'examples' must be a list")

    # Validate each example has required fields
    required_fields = {'left_text', 'right_text', 'break_here', 'explanation'}
    for i, example in enumerate(data['examples']):
        if not isinstance(example, dict):
            raise ValueError(f"Example {i} must be a dict")

        missing = required_fields - set(example.keys())
        if missing:
            raise ValueError(
                f"Example {i} missing required fields: {missing}"
            )

    return data
