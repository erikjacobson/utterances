"""Tests for turn construction."""

import pytest
from utterance_segmentation.models import Word, Turn
from utterance_segmentation.turns import build_turns, _create_turn


class TestBuildTurns:
    """Tests for build_turns function."""

    def test_empty_words_list(self):
        """Test that empty word list returns empty turn list."""
        turns = build_turns([])
        assert turns == []

    def test_single_word(self):
        """Test single word creates single turn."""
        words = [
            Word(index=0, text="hello", start=1.0, end=1.5, speaker="S1")
        ]
        turns = build_turns(words)

        assert len(turns) == 1
        assert turns[0].speaker == "S1"
        assert turns[0].num_words == 1
        assert turns[0].word_indices == [0]

    def test_single_speaker_no_pauses(self):
        """Test words from single speaker with no pauses create one turn."""
        words = [
            Word(index=0, text="hello", start=1.0, end=1.5, speaker="S1"),
            Word(index=1, text="world", start=1.6, end=2.0, speaker="S1"),
            Word(index=2, text="test", start=2.1, end=2.5, speaker="S1"),
        ]
        turns = build_turns(words, turn_break_threshold=1.5)

        assert len(turns) == 1
        assert turns[0].speaker == "S1"
        assert turns[0].num_words == 3
        assert turns[0].word_indices == [0, 1, 2]
        assert turns[0].start == 1.0
        assert turns[0].end == 2.5

    def test_speaker_change_creates_new_turn(self):
        """Test that speaker changes create new turns."""
        words = [
            Word(index=0, text="hello", start=1.0, end=1.5, speaker="S1"),
            Word(index=1, text="hi", start=1.6, end=2.0, speaker="S2"),
            Word(index=2, text="world", start=2.1, end=2.5, speaker="S1"),
        ]
        turns = build_turns(words, turn_break_threshold=1.5)

        assert len(turns) == 3
        assert turns[0].speaker == "S1"
        assert turns[0].word_indices == [0]
        assert turns[1].speaker == "S2"
        assert turns[1].word_indices == [1]
        assert turns[2].speaker == "S1"
        assert turns[2].word_indices == [2]

    def test_large_pause_creates_new_turn(self):
        """Test that large pause creates new turn even with same speaker."""
        words = [
            Word(index=0, text="hello", start=1.0, end=1.5, speaker="S1"),
            Word(index=1, text="world", start=1.6, end=2.0, speaker="S1"),
            # Large pause here (2.5 seconds)
            Word(index=2, text="later", start=4.5, end=5.0, speaker="S1"),
        ]
        turns = build_turns(words, turn_break_threshold=1.5)

        assert len(turns) == 2
        assert turns[0].speaker == "S1"
        assert turns[0].word_indices == [0, 1]
        assert turns[0].end == 2.0
        assert turns[1].speaker == "S1"
        assert turns[1].word_indices == [2]
        assert turns[1].start == 4.5

    def test_pause_exactly_at_threshold(self):
        """Test that pause exactly at threshold does not break turn."""
        words = [
            Word(index=0, text="hello", start=1.0, end=1.5, speaker="S1"),
            Word(index=1, text="world", start=3.0, end=3.5, speaker="S1"),  # Gap = 1.5s
        ]
        turns = build_turns(words, turn_break_threshold=1.5)

        # Should NOT break (gap is not > threshold)
        assert len(turns) == 1
        assert turns[0].word_indices == [0, 1]

    def test_pause_slightly_over_threshold(self):
        """Test that pause slightly over threshold breaks turn."""
        words = [
            Word(index=0, text="hello", start=1.0, end=1.5, speaker="S1"),
            Word(index=1, text="world", start=3.01, end=3.5, speaker="S1"),  # Gap = 1.51s
        ]
        turns = build_turns(words, turn_break_threshold=1.5)

        # Should break (gap > threshold)
        assert len(turns) == 2

    def test_multiple_speakers_and_pauses(self):
        """Test complex scenario with multiple speakers and pauses."""
        words = [
            Word(index=0, text="hello", start=1.0, end=1.5, speaker="S1"),
            Word(index=1, text="there", start=1.6, end=2.0, speaker="S1"),
            # Speaker change
            Word(index=2, text="hi", start=2.1, end=2.5, speaker="S2"),
            Word(index=3, text="back", start=2.6, end=3.0, speaker="S2"),
            # Large pause, same speaker
            Word(index=4, text="later", start=5.0, end=5.5, speaker="S2"),
            # Speaker change
            Word(index=5, text="bye", start=5.6, end=6.0, speaker="S1"),
        ]
        turns = build_turns(words, turn_break_threshold=1.5)

        assert len(turns) == 4
        # Turn 0: S1 [0, 1]
        assert turns[0].speaker == "S1"
        assert turns[0].word_indices == [0, 1]
        # Turn 1: S2 [2, 3]
        assert turns[1].speaker == "S2"
        assert turns[1].word_indices == [2, 3]
        # Turn 2: S2 [4] (after pause)
        assert turns[2].speaker == "S2"
        assert turns[2].word_indices == [4]
        # Turn 3: S1 [5]
        assert turns[3].speaker == "S1"
        assert turns[3].word_indices == [5]

    def test_turn_ids_sequential(self):
        """Test that turn IDs are sequential and properly formatted."""
        words = [
            Word(index=0, text="a", start=1.0, end=1.5, speaker="S1"),
            Word(index=1, text="b", start=2.0, end=2.5, speaker="S2"),
            Word(index=2, text="c", start=3.0, end=3.5, speaker="S1"),
        ]
        turns = build_turns(words, turn_break_threshold=1.5)

        assert turns[0].turn_id == "turn_0000"
        assert turns[1].turn_id == "turn_0001"
        assert turns[2].turn_id == "turn_0002"

    def test_overlapping_words(self):
        """Test words that overlap in time (gap is negative, clamped to 0)."""
        words = [
            Word(index=0, text="hello", start=1.0, end=2.0, speaker="S1"),
            Word(index=1, text="world", start=1.5, end=2.5, speaker="S1"),  # Overlaps
        ]
        turns = build_turns(words, turn_break_threshold=1.5)

        # Gap is negative, but clamped to 0, so should stay in same turn
        assert len(turns) == 1
        assert turns[0].word_indices == [0, 1]

    def test_custom_threshold(self):
        """Test using custom turn break threshold."""
        words = [
            Word(index=0, text="hello", start=1.0, end=1.5, speaker="S1"),
            Word(index=1, text="world", start=2.0, end=2.5, speaker="S1"),  # Gap = 0.5s
        ]

        # With threshold 0.3, gap of 0.5s should break
        turns = build_turns(words, turn_break_threshold=0.3)
        assert len(turns) == 2

        # With threshold 1.0, gap of 0.5s should not break
        turns = build_turns(words, turn_break_threshold=1.0)
        assert len(turns) == 1


class TestCreateTurn:
    """Tests for _create_turn helper function."""

    def test_create_turn_basic(self):
        """Test creating a turn from word indices."""
        words = [
            Word(index=0, text="hello", start=1.0, end=1.5, speaker="S1"),
            Word(index=1, text="world", start=1.6, end=2.0, speaker="S1"),
            Word(index=2, text="test", start=2.1, end=2.5, speaker="S1"),
        ]

        turn = _create_turn(
            turn_id="turn_0000",
            speaker="S1",
            word_indices=[0, 1, 2],
            words=words
        )

        assert turn.turn_id == "turn_0000"
        assert turn.speaker == "S1"
        assert turn.start == 1.0
        assert turn.end == 2.5
        assert turn.word_indices == [0, 1, 2]
        assert turn.num_words == 3

    def test_create_turn_single_word(self):
        """Test creating a turn with a single word."""
        words = [
            Word(index=5, text="hello", start=10.0, end=10.5, speaker="S2"),
        ]

        turn = _create_turn(
            turn_id="turn_0042",
            speaker="S2",
            word_indices=[0],  # Position 0 in the words array
            words=words
        )

        assert turn.start == 10.0
        assert turn.end == 10.5
        assert turn.num_words == 1

    def test_create_turn_copies_indices(self):
        """Test that word_indices are copied to avoid mutation."""
        words = [
            Word(index=0, text="hello", start=1.0, end=1.5, speaker="S1"),
            Word(index=1, text="world", start=1.6, end=2.0, speaker="S1"),
        ]

        original_indices = [0, 1]
        turn = _create_turn(
            turn_id="turn_0000",
            speaker="S1",
            word_indices=original_indices,
            words=words
        )

        # Mutate original
        original_indices.append(999)

        # Turn's indices should be unaffected
        assert turn.word_indices == [0, 1]
