"""Tests for candidate span generation."""

import pytest
from utterance_segmentation.models import Turn, CandidateSpan
from utterance_segmentation.spans import (
    build_candidate_spans,
    build_candidate_spans_for_all_turns
)


class TestBuildCandidateSpans:
    """Tests for build_candidate_spans function."""

    def test_empty_turn(self):
        """Test that empty turn returns empty span list."""
        turn = Turn(
            turn_id="turn_0000",
            speaker="S1",
            start=1.0,
            end=1.0,
            word_indices=[]
        )
        spans = build_candidate_spans(turn)
        assert spans == []

    def test_very_short_turn(self):
        """Test turn shorter than min_len creates single span."""
        # Turn with only 5 words (< 40 min_len)
        turn = Turn(
            turn_id="turn_0000",
            speaker="S1",
            start=1.0,
            end=5.0,
            word_indices=[0, 1, 2, 3, 4]
        )
        spans = build_candidate_spans(turn, min_len=40, max_len=80)

        assert len(spans) == 1
        assert spans[0].turn_id == "turn_0000"
        assert spans[0].start_word_index == 0
        assert spans[0].end_word_index == 4
        assert spans[0].num_words == 5

    def test_single_word_turn(self):
        """Test turn with single word."""
        turn = Turn(
            turn_id="turn_0000",
            speaker="S1",
            start=1.0,
            end=1.5,
            word_indices=[42]
        )
        spans = build_candidate_spans(turn, min_len=40, max_len=80)

        assert len(spans) == 1
        assert spans[0].start_word_index == 42
        assert spans[0].end_word_index == 42
        assert spans[0].num_words == 1

    def test_exactly_min_len_turn(self):
        """Test turn with exactly min_len words."""
        # Turn with exactly 40 words
        word_indices = list(range(0, 40))
        turn = Turn(
            turn_id="turn_0000",
            speaker="S1",
            start=1.0,
            end=50.0,
            word_indices=word_indices
        )
        spans = build_candidate_spans(turn, min_len=40, max_len=80)

        assert len(spans) == 1
        assert spans[0].start_word_index == 0
        assert spans[0].end_word_index == 39
        assert spans[0].num_words == 40

    def test_exactly_max_len_turn(self):
        """Test turn with exactly max_len words."""
        # Turn with exactly 80 words
        word_indices = list(range(0, 80))
        turn = Turn(
            turn_id="turn_0000",
            speaker="S1",
            start=1.0,
            end=100.0,
            word_indices=word_indices
        )
        spans = build_candidate_spans(turn, min_len=40, max_len=80)

        assert len(spans) == 1
        assert spans[0].start_word_index == 0
        assert spans[0].end_word_index == 79
        assert spans[0].num_words == 80

    def test_turn_slightly_over_max_len(self):
        """Test turn with 81 words (just over max_len)."""
        # Turn with 81 words should create 2 spans: 80 + 1
        word_indices = list(range(100, 181))  # 81 words
        turn = Turn(
            turn_id="turn_0000",
            speaker="S1",
            start=1.0,
            end=100.0,
            word_indices=word_indices
        )
        spans = build_candidate_spans(turn, min_len=40, max_len=80)

        assert len(spans) == 2
        # First span: 80 words
        assert spans[0].start_word_index == 100
        assert spans[0].end_word_index == 179
        assert spans[0].num_words == 80
        # Second span: 1 word (remainder)
        assert spans[1].start_word_index == 180
        assert spans[1].end_word_index == 180
        assert spans[1].num_words == 1

    def test_turn_with_two_max_len_spans(self):
        """Test turn with exactly 160 words (2 * max_len)."""
        word_indices = list(range(0, 160))
        turn = Turn(
            turn_id="turn_0000",
            speaker="S1",
            start=1.0,
            end=200.0,
            word_indices=word_indices
        )
        spans = build_candidate_spans(turn, min_len=40, max_len=80)

        assert len(spans) == 2
        # First span: 0-79 (80 words)
        assert spans[0].start_word_index == 0
        assert spans[0].end_word_index == 79
        assert spans[0].num_words == 80
        # Second span: 80-159 (80 words)
        assert spans[1].start_word_index == 80
        assert spans[1].end_word_index == 159
        assert spans[1].num_words == 80

    def test_turn_with_small_remainder(self):
        """Test turn where remainder is less than min_len."""
        # 100 words: should create 80 + 20
        word_indices = list(range(0, 100))
        turn = Turn(
            turn_id="turn_0000",
            speaker="S1",
            start=1.0,
            end=200.0,
            word_indices=word_indices
        )
        spans = build_candidate_spans(turn, min_len=40, max_len=80)

        assert len(spans) == 2
        # First span: 80 words
        assert spans[0].num_words == 80
        # Second span: 20 words (< min_len, but that's allowed for remainder)
        assert spans[1].num_words == 20
        assert spans[1].start_word_index == 80
        assert spans[1].end_word_index == 99

    def test_turn_with_large_remainder(self):
        """Test turn where remainder is between min_len and max_len."""
        # 130 words: should create 80 + 50
        word_indices = list(range(0, 130))
        turn = Turn(
            turn_id="turn_0000",
            speaker="S1",
            start=1.0,
            end=200.0,
            word_indices=word_indices
        )
        spans = build_candidate_spans(turn, min_len=40, max_len=80)

        assert len(spans) == 2
        # First span: 80 words
        assert spans[0].num_words == 80
        # Second span: 50 words
        assert spans[1].num_words == 50

    def test_long_turn_multiple_spans(self):
        """Test long turn creates multiple spans."""
        # 250 words: should create 80 + 80 + 80 + 10
        word_indices = list(range(500, 750))  # 250 words
        turn = Turn(
            turn_id="turn_0000",
            speaker="S1",
            start=1.0,
            end=300.0,
            word_indices=word_indices
        )
        spans = build_candidate_spans(turn, min_len=40, max_len=80)

        assert len(spans) == 4
        assert spans[0].num_words == 80
        assert spans[0].start_word_index == 500
        assert spans[0].end_word_index == 579
        assert spans[1].num_words == 80
        assert spans[1].start_word_index == 580
        assert spans[1].end_word_index == 659
        assert spans[2].num_words == 80
        assert spans[2].start_word_index == 660
        assert spans[2].end_word_index == 739
        assert spans[3].num_words == 10
        assert spans[3].start_word_index == 740
        assert spans[3].end_word_index == 749

    def test_custom_min_max_lengths(self):
        """Test using custom min and max lengths."""
        # 25 words with min_len=10, max_len=15
        word_indices = list(range(0, 25))
        turn = Turn(
            turn_id="turn_0000",
            speaker="S1",
            start=1.0,
            end=30.0,
            word_indices=word_indices
        )
        spans = build_candidate_spans(turn, min_len=10, max_len=15)

        # Should create: 15 + 10
        assert len(spans) == 2
        assert spans[0].num_words == 15
        assert spans[1].num_words == 10

    def test_all_spans_have_correct_turn_id(self):
        """Test that all spans reference their parent turn."""
        word_indices = list(range(0, 200))
        turn = Turn(
            turn_id="turn_0042",
            speaker="S1",
            start=1.0,
            end=300.0,
            word_indices=word_indices
        )
        spans = build_candidate_spans(turn, min_len=40, max_len=80)

        for span in spans:
            assert span.turn_id == "turn_0042"

    def test_spans_are_contiguous(self):
        """Test that spans cover all words without gaps or overlaps."""
        word_indices = list(range(0, 150))
        turn = Turn(
            turn_id="turn_0000",
            speaker="S1",
            start=1.0,
            end=200.0,
            word_indices=word_indices
        )
        spans = build_candidate_spans(turn, min_len=40, max_len=80)

        # Verify no gaps or overlaps
        for i in range(len(spans) - 1):
            # Next span should start exactly after current span ends
            assert spans[i + 1].start_word_index == spans[i].end_word_index + 1

        # Verify coverage
        assert spans[0].start_word_index == word_indices[0]
        assert spans[-1].end_word_index == word_indices[-1]

    def test_word_indices_from_middle_of_document(self):
        """Test turn where word indices don't start at 0."""
        # Turn starting from word 50 in the original document
        word_indices = list(range(50, 180))  # 130 words: indices 50-179
        turn = Turn(
            turn_id="turn_0000",
            speaker="S1",
            start=1.0,
            end=200.0,
            word_indices=word_indices
        )
        spans = build_candidate_spans(turn, min_len=40, max_len=80)

        # Should create 2 spans: 80 + 50
        assert len(spans) == 2
        assert spans[0].start_word_index == 50
        assert spans[0].end_word_index == 129
        assert spans[0].num_words == 80
        assert spans[1].start_word_index == 130
        assert spans[1].end_word_index == 179
        assert spans[1].num_words == 50


class TestBuildCandidateSpansForAllTurns:
    """Tests for build_candidate_spans_for_all_turns function."""

    def test_empty_turns_list(self):
        """Test that empty turns list returns empty spans list."""
        spans = build_candidate_spans_for_all_turns([])
        assert spans == []

    def test_single_turn(self):
        """Test processing a single turn."""
        turn = Turn(
            turn_id="turn_0000",
            speaker="S1",
            start=1.0,
            end=50.0,
            word_indices=list(range(0, 100))  # 100 words
        )
        spans = build_candidate_spans_for_all_turns([turn], min_len=40, max_len=80)

        # Should create 80 + 20
        assert len(spans) == 2
        assert all(s.turn_id == "turn_0000" for s in spans)

    def test_multiple_turns(self):
        """Test processing multiple turns."""
        turns = [
            Turn(
                turn_id="turn_0000",
                speaker="S1",
                start=1.0,
                end=50.0,
                word_indices=list(range(0, 50))  # 50 words
            ),
            Turn(
                turn_id="turn_0001",
                speaker="S2",
                start=51.0,
                end=100.0,
                word_indices=list(range(50, 100))  # 50 words
            ),
            Turn(
                turn_id="turn_0002",
                speaker="S1",
                start=101.0,
                end=200.0,
                word_indices=list(range(100, 200))  # 100 words
            ),
        ]
        spans = build_candidate_spans_for_all_turns(turns, min_len=40, max_len=80)

        # Turn 0: 50 words -> 1 span
        # Turn 1: 50 words -> 1 span
        # Turn 2: 100 words -> 2 spans (80 + 20)
        assert len(spans) == 4

        # Verify turn IDs
        assert spans[0].turn_id == "turn_0000"
        assert spans[1].turn_id == "turn_0001"
        assert spans[2].turn_id == "turn_0002"
        assert spans[3].turn_id == "turn_0002"

    def test_turns_with_various_lengths(self):
        """Test turns of various lengths."""
        turns = [
            # Very short turn
            Turn(
                turn_id="turn_0000",
                speaker="S1",
                start=1.0,
                end=2.0,
                word_indices=[0, 1, 2]  # 3 words
            ),
            # Medium turn
            Turn(
                turn_id="turn_0001",
                speaker="S2",
                start=3.0,
                end=50.0,
                word_indices=list(range(3, 53))  # 50 words
            ),
            # Long turn
            Turn(
                turn_id="turn_0002",
                speaker="S1",
                start=51.0,
                end=200.0,
                word_indices=list(range(53, 253))  # 200 words
            ),
        ]
        spans = build_candidate_spans_for_all_turns(turns, min_len=40, max_len=80)

        # Turn 0: 3 words -> 1 span
        # Turn 1: 50 words -> 1 span
        # Turn 2: 200 words -> 3 spans (80 + 80 + 40)
        assert len(spans) == 5

    def test_custom_parameters_applied_to_all(self):
        """Test that custom min/max are applied to all turns."""
        turns = [
            Turn(
                turn_id="turn_0000",
                speaker="S1",
                start=1.0,
                end=20.0,
                word_indices=list(range(0, 30))
            ),
            Turn(
                turn_id="turn_0001",
                speaker="S2",
                start=21.0,
                end=40.0,
                word_indices=list(range(30, 60))
            ),
        ]
        spans = build_candidate_spans_for_all_turns(turns, min_len=10, max_len=20)

        # Each turn has 30 words, should create: 20 + 10
        assert len(spans) == 4
        assert spans[0].num_words == 20
        assert spans[1].num_words == 10
        assert spans[2].num_words == 20
        assert spans[3].num_words == 10
