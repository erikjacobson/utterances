"""Tests for core data structures."""

import pytest
from pydantic import ValidationError

from utterance_segmentation.models import (
    Word,
    Turn,
    CandidateSpan,
    Utterance,
    AudioInfo,
    ASRDocument,
    UtteranceDocument,
)


class TestWord:
    """Tests for Word model."""

    def test_valid_word(self):
        """Test creating a valid word."""
        word = Word(
            index=0,
            text="hello",
            start=1.0,
            end=1.5,
            speaker="S1",
            confidence=0.95
        )
        assert word.index == 0
        assert word.text == "hello"
        assert word.start == 1.0
        assert word.end == 1.5
        assert word.speaker == "S1"
        assert word.confidence == 0.95

    def test_word_without_confidence(self):
        """Test word with no confidence score."""
        word = Word(
            index=0,
            text="hello",
            start=1.0,
            end=1.5,
            speaker="S1"
        )
        assert word.confidence is None

    def test_negative_start_time(self):
        """Test that negative start time is rejected."""
        with pytest.raises(ValidationError):
            Word(
                index=0,
                text="hello",
                start=-1.0,
                end=1.5,
                speaker="S1"
            )

    def test_end_before_start(self):
        """Test that end < start is rejected."""
        with pytest.raises(ValidationError):
            Word(
                index=0,
                text="hello",
                start=2.0,
                end=1.0,
                speaker="S1"
            )

    def test_zero_duration_word(self):
        """Test that zero-duration words are allowed (start == end)."""
        word = Word(
            index=0,
            text="hello",
            start=1.0,
            end=1.0,
            speaker="S1"
        )
        assert word.start == word.end


class TestTurn:
    """Tests for Turn model."""

    def test_valid_turn(self):
        """Test creating a valid turn."""
        turn = Turn(
            turn_id="turn_0001",
            speaker="S1",
            start=1.0,
            end=10.5,
            word_indices=[0, 1, 2, 3, 4]
        )
        assert turn.turn_id == "turn_0001"
        assert turn.speaker == "S1"
        assert turn.num_words == 5

    def test_empty_turn(self):
        """Test turn with no words."""
        turn = Turn(
            turn_id="turn_0001",
            speaker="S1",
            start=1.0,
            end=1.0,
            word_indices=[]
        )
        assert turn.num_words == 0


class TestCandidateSpan:
    """Tests for CandidateSpan model."""

    def test_valid_span(self):
        """Test creating a valid candidate span."""
        span = CandidateSpan(
            turn_id="turn_0001",
            start_word_index=0,
            end_word_index=49
        )
        assert span.turn_id == "turn_0001"
        assert span.num_words == 50

    def test_single_word_span(self):
        """Test span with a single word."""
        span = CandidateSpan(
            turn_id="turn_0001",
            start_word_index=5,
            end_word_index=5
        )
        assert span.num_words == 1


class TestUtterance:
    """Tests for Utterance model."""

    def test_valid_utterance(self):
        """Test creating a valid utterance."""
        utterance = Utterance(
            id="utt_0001",
            speaker="S1",
            start=1.0,
            end=10.5,
            word_indices=[0, 1, 2, 3, 4],
            text="This is a test utterance",
            num_words=5,
            source_turn_id="turn_0001"
        )
        assert utterance.id == "utt_0001"
        assert utterance.speaker == "S1"
        assert utterance.num_words == 5
        assert utterance.text == "This is a test utterance"


class TestASRDocument:
    """Tests for ASRDocument model."""

    def test_valid_asr_document(self):
        """Test creating a valid ASR document."""
        doc = ASRDocument(
            audio=AudioInfo(path="test.wav", duration=100.0),
            words=[
                Word(index=0, text="hello", start=1.0, end=1.5, speaker="S1"),
                Word(index=1, text="world", start=1.6, end=2.0, speaker="S1"),
            ]
        )
        assert doc.audio.path == "test.wav"
        assert len(doc.words) == 2

    def test_empty_words_list(self):
        """Test ASR document with no words."""
        doc = ASRDocument(
            audio=AudioInfo(path="test.wav", duration=100.0),
            words=[]
        )
        assert len(doc.words) == 0

    def test_unsorted_words_rejected(self):
        """Test that unsorted words are rejected."""
        with pytest.raises(ValidationError):
            ASRDocument(
                audio=AudioInfo(path="test.wav", duration=100.0),
                words=[
                    Word(index=0, text="hello", start=2.0, end=2.5, speaker="S1"),
                    Word(index=1, text="world", start=1.0, end=1.5, speaker="S1"),
                ]
            )

    def test_sorted_words_accepted(self):
        """Test that properly sorted words are accepted."""
        doc = ASRDocument(
            audio=AudioInfo(path="test.wav", duration=100.0),
            words=[
                Word(index=0, text="hello", start=1.0, end=1.5, speaker="S1"),
                Word(index=1, text="world", start=1.5, end=2.0, speaker="S1"),
                Word(index=2, text="test", start=2.1, end=2.5, speaker="S2"),
            ]
        )
        assert len(doc.words) == 3


class TestUtteranceDocument:
    """Tests for UtteranceDocument model."""

    def test_valid_utterance_document(self):
        """Test creating a valid utterance document."""
        words = [
            Word(index=0, text="hello", start=1.0, end=1.5, speaker="S1"),
            Word(index=1, text="world", start=1.6, end=2.0, speaker="S1"),
        ]
        doc = UtteranceDocument(
            audio=AudioInfo(path="test.wav", duration=100.0),
            words=words,
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
        assert len(doc.utterances) == 1
        assert doc.utterances[0].text == "hello world"
