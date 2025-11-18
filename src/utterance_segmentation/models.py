"""Core data structures for utterance segmentation pipeline."""

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


class Word(BaseModel):
    """A single word from ASR output with timing and speaker info."""

    index: int = Field(..., description="Position in the original word sequence")
    text: str = Field(..., description="The word text")
    start: float = Field(..., description="Start time in seconds")
    end: float = Field(..., description="End time in seconds")
    speaker: str = Field(..., description="Speaker label (e.g., 'S1', 'S2')")
    confidence: Optional[float] = Field(None, description="ASR confidence score")

    @field_validator('start', 'end')
    @classmethod
    def validate_times(cls, v: float) -> float:
        """Ensure times are non-negative."""
        if v < 0:
            raise ValueError(f"Time must be non-negative, got {v}")
        return v

    @field_validator('end')
    @classmethod
    def validate_end_after_start(cls, v: float, info) -> float:
        """Ensure end time is after start time."""
        if 'start' in info.data and v < info.data['start']:
            raise ValueError(f"end ({v}) must be >= start ({info.data['start']})")
        return v


class Turn(BaseModel):
    """A contiguous sequence of words from a single speaker."""

    turn_id: str = Field(..., description="Unique turn identifier")
    speaker: str = Field(..., description="Speaker label")
    start: float = Field(..., description="Turn start time (first word)")
    end: float = Field(..., description="Turn end time (last word)")
    word_indices: List[int] = Field(..., description="Indices into the original words list")

    @property
    def num_words(self) -> int:
        """Number of words in this turn."""
        return len(self.word_indices)


class CandidateSpan(BaseModel):
    """A 40-80 word span within a turn for LLM boundary evaluation."""

    turn_id: str = Field(..., description="Parent turn identifier")
    start_word_index: int = Field(..., description="Index of first word in span")
    end_word_index: int = Field(..., description="Index of last word in span (inclusive)")

    @property
    def num_words(self) -> int:
        """Number of words in this span."""
        return self.end_word_index - self.start_word_index + 1


class Utterance(BaseModel):
    """A semantically coherent utterance with timing and speaker info."""

    id: str = Field(..., description="Unique utterance identifier")
    speaker: str = Field(..., description="Speaker label")
    start: float = Field(..., description="Utterance start time")
    end: float = Field(..., description="Utterance end time")
    word_indices: List[int] = Field(..., description="Indices into original words list")
    text: str = Field(..., description="Concatenated word text")
    num_words: int = Field(..., description="Number of words")
    source_turn_id: str = Field(..., description="Originating turn identifier")


class AudioInfo(BaseModel):
    """Metadata about the source audio file."""

    path: str = Field(..., description="Path to audio file")
    duration: float = Field(..., description="Duration in seconds")


class ASRDocument(BaseModel):
    """Complete ASR output with audio metadata and words."""

    audio: AudioInfo
    words: List[Word]

    @field_validator('words')
    @classmethod
    def validate_words_sorted(cls, v: List[Word]) -> List[Word]:
        """Ensure words are sorted by start time."""
        if len(v) > 1:
            for i in range(len(v) - 1):
                if v[i].start > v[i + 1].start:
                    raise ValueError(
                        f"Words must be sorted by start time. "
                        f"Word {i} starts at {v[i].start}, "
                        f"word {i+1} starts at {v[i+1].start}"
                    )
        return v


class UtteranceDocument(BaseModel):
    """Complete output with original words and segmented utterances."""

    audio: AudioInfo
    words: List[Word]
    utterances: List[Utterance]
