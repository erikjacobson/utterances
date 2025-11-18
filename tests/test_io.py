"""Tests for I/O helpers."""

import json
import pytest
from pathlib import Path
from pydantic import ValidationError

from utterance_segmentation.io import (
    load_asr_json,
    save_utterance_json,
    load_few_shot_examples,
)
from utterance_segmentation.models import (
    ASRDocument,
    UtteranceDocument,
    AudioInfo,
    Word,
    Utterance,
)


@pytest.fixture
def tmp_dir(tmp_path):
    """Provide a temporary directory for test files."""
    return tmp_path


@pytest.fixture
def sample_asr_data():
    """Sample ASR data for testing."""
    return {
        "audio": {
            "path": "session_001.wav",
            "duration": 3600.0
        },
        "words": [
            {
                "index": 0,
                "text": "so",
                "start": 12.30,
                "end": 12.56,
                "speaker": "S1",
                "confidence": 0.93
            },
            {
                "index": 1,
                "text": "I",
                "start": 12.56,
                "end": 12.70,
                "speaker": "S1",
                "confidence": 0.95
            },
            {
                "index": 2,
                "text": "think",
                "start": 12.71,
                "end": 13.00,
                "speaker": "S1",
                "confidence": 0.98
            }
        ]
    }


@pytest.fixture
def sample_few_shot_data():
    """Sample few-shot examples for testing."""
    return {
        "examples": [
            {
                "left_text": "I think we should start with some easier problems",
                "right_text": "then we can move on to harder ones",
                "break_here": False,
                "explanation": "Natural continuation of the same idea"
            },
            {
                "left_text": "So what did you notice about the pattern",
                "right_text": "take a moment and look at the differences",
                "break_here": True,
                "explanation": "Question followed by new directive"
            }
        ]
    }


class TestLoadASRJson:
    """Tests for load_asr_json function."""

    def test_load_valid_asr_json(self, tmp_dir, sample_asr_data):
        """Test loading a valid ASR JSON file."""
        # Create test file
        asr_file = tmp_dir / "test_asr.json"
        with open(asr_file, 'w') as f:
            json.dump(sample_asr_data, f)

        # Load and validate
        doc = load_asr_json(asr_file)
        assert isinstance(doc, ASRDocument)
        assert doc.audio.path == "session_001.wav"
        assert len(doc.words) == 3
        assert doc.words[0].text == "so"
        assert doc.words[0].speaker == "S1"

    def test_load_nonexistent_file(self, tmp_dir):
        """Test that loading nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_asr_json(tmp_dir / "nonexistent.json")

    def test_load_invalid_json(self, tmp_dir):
        """Test that invalid JSON raises an error."""
        # Create invalid JSON file
        bad_file = tmp_dir / "bad.json"
        with open(bad_file, 'w') as f:
            f.write("{ this is not valid json }")

        with pytest.raises(json.JSONDecodeError):
            load_asr_json(bad_file)

    def test_load_missing_required_fields(self, tmp_dir):
        """Test that missing required fields raises ValidationError."""
        # Create JSON missing required field
        bad_data = {
            "audio": {"path": "test.wav", "duration": 100.0},
            "words": [
                {
                    "index": 0,
                    "text": "hello",
                    # Missing start, end, speaker
                }
            ]
        }
        bad_file = tmp_dir / "bad_schema.json"
        with open(bad_file, 'w') as f:
            json.dump(bad_data, f)

        with pytest.raises(ValidationError):
            load_asr_json(bad_file)

    def test_load_unsorted_words(self, tmp_dir):
        """Test that unsorted words raise ValidationError."""
        bad_data = {
            "audio": {"path": "test.wav", "duration": 100.0},
            "words": [
                {
                    "index": 0,
                    "text": "world",
                    "start": 2.0,
                    "end": 2.5,
                    "speaker": "S1"
                },
                {
                    "index": 1,
                    "text": "hello",
                    "start": 1.0,
                    "end": 1.5,
                    "speaker": "S1"
                }
            ]
        }
        bad_file = tmp_dir / "unsorted.json"
        with open(bad_file, 'w') as f:
            json.dump(bad_data, f)

        with pytest.raises(ValidationError):
            load_asr_json(bad_file)


class TestSaveUtteranceJson:
    """Tests for save_utterance_json function."""

    def test_save_utterance_json(self, tmp_dir):
        """Test saving an utterance document to JSON."""
        # Create test document
        doc = UtteranceDocument(
            audio=AudioInfo(path="test.wav", duration=100.0),
            words=[
                Word(index=0, text="hello", start=1.0, end=1.5, speaker="S1"),
                Word(index=1, text="world", start=1.6, end=2.0, speaker="S1"),
            ],
            utterances=[
                Utterance(
                    id="utt_0001",
                    speaker="S1",
                    start=1.0,
                    end=2.0,
                    word_indices=[0, 1],
                    text="hello world",
                    num_words=2,
                    source_turn_id="turn_0001"
                )
            ]
        )

        # Save to file
        output_file = tmp_dir / "output.json"
        save_utterance_json(doc, output_file)

        # Verify file exists and can be loaded
        assert output_file.exists()
        with open(output_file) as f:
            loaded = json.load(f)

        assert loaded["audio"]["path"] == "test.wav"
        assert len(loaded["words"]) == 2
        assert len(loaded["utterances"]) == 1
        assert loaded["utterances"][0]["text"] == "hello world"

    def test_save_creates_parent_directory(self, tmp_dir):
        """Test that save creates parent directories if needed."""
        doc = UtteranceDocument(
            audio=AudioInfo(path="test.wav", duration=100.0),
            words=[],
            utterances=[]
        )

        # Save to nested path that doesn't exist
        output_file = tmp_dir / "subdir" / "nested" / "output.json"
        save_utterance_json(doc, output_file)

        assert output_file.exists()


class TestLoadFewShotExamples:
    """Tests for load_few_shot_examples function."""

    def test_load_valid_examples(self, tmp_dir, sample_few_shot_data):
        """Test loading valid few-shot examples."""
        examples_file = tmp_dir / "examples.json"
        with open(examples_file, 'w') as f:
            json.dump(sample_few_shot_data, f)

        data = load_few_shot_examples(examples_file)
        assert "examples" in data
        assert len(data["examples"]) == 2
        assert data["examples"][0]["left_text"] == "I think we should start with some easier problems"
        assert data["examples"][0]["break_here"] is False

    def test_load_nonexistent_file(self, tmp_dir):
        """Test that loading nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_few_shot_examples(tmp_dir / "nonexistent.json")

    def test_missing_examples_key(self, tmp_dir):
        """Test that missing 'examples' key raises ValueError."""
        bad_data = {"wrong_key": []}
        bad_file = tmp_dir / "bad.json"
        with open(bad_file, 'w') as f:
            json.dump(bad_data, f)

        with pytest.raises(ValueError, match="must contain 'examples'"):
            load_few_shot_examples(bad_file)

    def test_examples_not_list(self, tmp_dir):
        """Test that non-list 'examples' raises ValueError."""
        bad_data = {"examples": "not a list"}
        bad_file = tmp_dir / "bad.json"
        with open(bad_file, 'w') as f:
            json.dump(bad_data, f)

        with pytest.raises(ValueError, match="must be a list"):
            load_few_shot_examples(bad_file)

    def test_example_missing_required_field(self, tmp_dir):
        """Test that examples missing required fields raise ValueError."""
        bad_data = {
            "examples": [
                {
                    "left_text": "some text",
                    # Missing right_text, break_here, explanation
                }
            ]
        }
        bad_file = tmp_dir / "bad.json"
        with open(bad_file, 'w') as f:
            json.dump(bad_data, f)

        with pytest.raises(ValueError, match="missing required fields"):
            load_few_shot_examples(bad_file)

    def test_empty_examples_list(self, tmp_dir):
        """Test that empty examples list is allowed."""
        data = {"examples": []}
        examples_file = tmp_dir / "empty.json"
        with open(examples_file, 'w') as f:
            json.dump(data, f)

        result = load_few_shot_examples(examples_file)
        assert result["examples"] == []
