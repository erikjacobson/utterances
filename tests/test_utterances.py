"""Tests for utterance construction."""

import pytest
from unittest.mock import Mock, patch

from utterance_segmentation.models import Word, Turn, CandidateSpan, Utterance
from utterance_segmentation.utterances import (
    evaluate_span_boundaries,
    build_utterances,
    build_utterances_simple,
)
from utterance_segmentation.llm_boundaries import LLMBoundaryClassifier


@pytest.fixture
def sample_words():
    """Create sample word list for testing."""
    words = []
    start_time = 0.0
    for i in range(100):
        word = Word(
            index=i,
            text=f"word{i}",
            start=start_time,
            end=start_time + 0.3,
            speaker="S1" if i < 50 else "S2",
            confidence=0.95
        )
        words.append(word)
        start_time += 0.4  # 0.1s gap between words
    return words


@pytest.fixture
def sample_turn():
    """Create sample turn for testing."""
    return Turn(
        turn_id="turn_0000",
        speaker="S1",
        start=0.0,
        end=20.0,
        word_indices=list(range(0, 50))
    )


@pytest.fixture
def sample_span():
    """Create sample candidate span for testing."""
    return CandidateSpan(
        turn_id="turn_0000",
        start_word_index=0,
        end_word_index=49  # 50 words
    )


@pytest.fixture
def mock_llm_client():
    """Create mock LLM client."""
    client = Mock(spec=LLMBoundaryClassifier)
    return client


class TestEvaluateSpanBoundaries:
    """Tests for evaluate_span_boundaries function."""

    def test_short_span_no_boundaries(self, sample_words, mock_llm_client):
        """Test that short spans (< min_len) return no boundaries."""
        # Create a span with only 20 words
        span = CandidateSpan(
            turn_id="turn_0000",
            start_word_index=0,
            end_word_index=19
        )

        config = {'min_utterance_length': 40}
        boundaries = evaluate_span_boundaries(
            span, sample_words, mock_llm_client, [], config
        )

        assert boundaries == []
        # LLM should not be called for short spans
        mock_llm_client.query_boundary.assert_not_called()

    def test_llm_approves_boundary(self, sample_span, sample_words, mock_llm_client):
        """Test boundary approved by LLM is included."""
        # Mock LLM to approve all boundaries
        mock_llm_client.query_boundary.return_value = {
            'break_here': True,
            'left_is_complete': True,
            'right_is_new_or_complete': True,
            'reason': 'Test'
        }

        config = {'min_utterance_length': 40, 'max_boundary_candidates': 2}
        boundaries = evaluate_span_boundaries(
            sample_span, sample_words, mock_llm_client, [], config
        )

        # Should have some approved boundaries
        assert len(boundaries) > 0
        # All boundaries should be within span
        for b in boundaries:
            assert sample_span.start_word_index < b <= sample_span.end_word_index

    def test_llm_rejects_boundary(self, sample_span, sample_words, mock_llm_client):
        """Test boundary rejected by LLM is not included."""
        # Mock LLM to reject all boundaries
        mock_llm_client.query_boundary.return_value = {
            'break_here': False,
            'left_is_complete': False,
            'right_is_new_or_complete': False,
            'reason': 'Test'
        }

        config = {'min_utterance_length': 40}
        boundaries = evaluate_span_boundaries(
            sample_span, sample_words, mock_llm_client, [], config
        )

        assert boundaries == []

    def test_max_candidates_limit(self, sample_span, sample_words, mock_llm_client):
        """Test that max_boundary_candidates limits LLM calls."""
        mock_llm_client.query_boundary.return_value = {
            'break_here': True,
            'left_is_complete': True,
            'right_is_new_or_complete': True,
            'reason': 'Test'
        }

        config = {'min_utterance_length': 40, 'max_boundary_candidates': 3}
        boundaries = evaluate_span_boundaries(
            sample_span, sample_words, mock_llm_client, [], config
        )

        # Should make at most 3 LLM calls
        assert mock_llm_client.query_boundary.call_count <= 3

    def test_llm_error_handling(self, sample_span, sample_words, mock_llm_client):
        """Test that LLM errors are handled gracefully."""
        # Mock LLM to raise error
        mock_llm_client.query_boundary.side_effect = Exception("API Error")

        config = {'min_utterance_length': 40, 'max_boundary_candidates': 2}
        # Should not raise, just return empty list
        boundaries = evaluate_span_boundaries(
            sample_span, sample_words, mock_llm_client, [], config
        )

        assert boundaries == []


class TestBuildUtterancesSimple:
    """Tests for build_utterances_simple function."""

    def test_single_span(self, sample_words):
        """Test building utterances from single span."""
        turn = Turn(
            turn_id="turn_0000",
            speaker="S1",
            start=0.0,
            end=20.0,
            word_indices=list(range(0, 50))
        )
        span = CandidateSpan(
            turn_id="turn_0000",
            start_word_index=0,
            end_word_index=49
        )

        utterances = build_utterances_simple([turn], [span], sample_words)

        assert len(utterances) == 1
        assert utterances[0].speaker == "S1"
        assert utterances[0].num_words == 50
        assert utterances[0].source_turn_id == "turn_0000"
        assert utterances[0].id == "utt_0000"

    def test_multiple_spans(self, sample_words):
        """Test building utterances from multiple spans."""
        turn = Turn(
            turn_id="turn_0000",
            speaker="S1",
            start=0.0,
            end=40.0,
            word_indices=list(range(0, 100))
        )
        spans = [
            CandidateSpan(turn_id="turn_0000", start_word_index=0, end_word_index=39),
            CandidateSpan(turn_id="turn_0000", start_word_index=40, end_word_index=79),
            CandidateSpan(turn_id="turn_0000", start_word_index=80, end_word_index=99),
        ]

        utterances = build_utterances_simple([turn], spans, sample_words)

        assert len(utterances) == 3
        assert utterances[0].num_words == 40
        assert utterances[1].num_words == 40
        assert utterances[2].num_words == 20
        # Verify IDs are sequential
        assert utterances[0].id == "utt_0000"
        assert utterances[1].id == "utt_0001"
        assert utterances[2].id == "utt_0002"

    def test_utterance_text_concatenation(self, sample_words):
        """Test that utterance text is correctly concatenated."""
        turn = Turn(
            turn_id="turn_0000",
            speaker="S1",
            start=0.0,
            end=2.0,
            word_indices=[0, 1, 2]
        )
        span = CandidateSpan(
            turn_id="turn_0000",
            start_word_index=0,
            end_word_index=2
        )

        utterances = build_utterances_simple([turn], [span], sample_words)

        assert utterances[0].text == "word0 word1 word2"

    def test_utterance_timing(self, sample_words):
        """Test that utterance timing matches first/last words."""
        turn = Turn(
            turn_id="turn_0000",
            speaker="S1",
            start=0.0,
            end=10.0,
            word_indices=list(range(10, 20))
        )
        span = CandidateSpan(
            turn_id="turn_0000",
            start_word_index=10,
            end_word_index=19
        )

        utterances = build_utterances_simple([turn], [span], sample_words)

        assert utterances[0].start == sample_words[10].start
        assert utterances[0].end == sample_words[19].end

    def test_empty_inputs(self):
        """Test handling of empty inputs."""
        utterances = build_utterances_simple([], [], [])
        assert utterances == []

    def test_span_without_matching_turn(self, sample_words):
        """Test that spans without matching turns are skipped."""
        turn = Turn(
            turn_id="turn_0000",
            speaker="S1",
            start=0.0,
            end=10.0,
            word_indices=list(range(0, 10))
        )
        span = CandidateSpan(
            turn_id="turn_9999",  # Non-existent turn
            start_word_index=0,
            end_word_index=9
        )

        utterances = build_utterances_simple([turn], [span], sample_words)

        assert len(utterances) == 0


class TestBuildUtterances:
    """Tests for build_utterances function (with LLM)."""

    def test_no_llm_boundaries(self, sample_words, mock_llm_client):
        """Test when LLM rejects all boundaries (span becomes single utterance)."""
        turn = Turn(
            turn_id="turn_0000",
            speaker="S1",
            start=0.0,
            end=20.0,
            word_indices=list(range(0, 50))
        )
        span = CandidateSpan(
            turn_id="turn_0000",
            start_word_index=0,
            end_word_index=49
        )

        # Mock LLM to reject all boundaries
        mock_llm_client.query_boundary.return_value = {
            'break_here': False,
            'left_is_complete': False,
            'right_is_new_or_complete': False,
            'reason': 'No break'
        }

        config = {'min_utterance_length': 40}
        utterances = build_utterances(
            [turn], [span], sample_words, mock_llm_client, [], config
        )

        # Should create one utterance for entire span
        assert len(utterances) == 1
        assert utterances[0].num_words == 50

    def test_with_llm_boundaries(self, sample_words, mock_llm_client):
        """Test when LLM approves boundaries."""
        turn = Turn(
            turn_id="turn_0000",
            speaker="S1",
            start=0.0,
            end=30.0,
            word_indices=list(range(0, 80))
        )
        span = CandidateSpan(
            turn_id="turn_0000",
            start_word_index=0,
            end_word_index=79
        )

        # Mock LLM to approve first boundary checked
        call_count = [0]

        def mock_query(prompt):
            call_count[0] += 1
            # Approve first boundary, reject rest
            return {
                'break_here': call_count[0] == 1,
                'left_is_complete': True,
                'right_is_new_or_complete': True,
                'reason': 'Test'
            }

        mock_llm_client.query_boundary.side_effect = mock_query

        config = {'min_utterance_length': 40, 'max_boundary_candidates': 2}
        utterances = build_utterances(
            [turn], [span], sample_words, mock_llm_client, [], config
        )

        # Should create multiple utterances
        assert len(utterances) >= 2

    def test_multiple_turns(self, sample_words, mock_llm_client):
        """Test building utterances across multiple turns."""
        turns = [
            Turn(
                turn_id="turn_0000",
                speaker="S1",
                start=0.0,
                end=20.0,
                word_indices=list(range(0, 50))
            ),
            Turn(
                turn_id="turn_0001",
                speaker="S2",
                start=20.0,
                end=40.0,
                word_indices=list(range(50, 100))
            ),
        ]
        spans = [
            CandidateSpan(turn_id="turn_0000", start_word_index=0, end_word_index=49),
            CandidateSpan(turn_id="turn_0001", start_word_index=50, end_word_index=99),
        ]

        mock_llm_client.query_boundary.return_value = {'break_here': False, 'left_is_complete': False, 'right_is_new_or_complete': False, 'reason': 'Test'}

        config = {'min_utterance_length': 40}
        utterances = build_utterances(
            turns, spans, sample_words, mock_llm_client, [], config
        )

        # Should have utterances from both turns
        assert len(utterances) >= 2
        # Verify speakers are correct
        speakers = {utt.speaker for utt in utterances}
        assert "S1" in speakers
        assert "S2" in speakers

    def test_utterance_ids_sequential(self, sample_words, mock_llm_client):
        """Test that utterance IDs are sequential across turns."""
        turns = [
            Turn(turn_id="turn_0000", speaker="S1", start=0.0, end=20.0, word_indices=list(range(0, 40))),
            Turn(turn_id="turn_0001", speaker="S2", start=20.0, end=40.0, word_indices=list(range(40, 80))),
        ]
        spans = [
            CandidateSpan(turn_id="turn_0000", start_word_index=0, end_word_index=39),
            CandidateSpan(turn_id="turn_0001", start_word_index=40, end_word_index=79),
        ]

        mock_llm_client.query_boundary.return_value = {'break_here': False, 'left_is_complete': False, 'right_is_new_or_complete': False, 'reason': 'Test'}

        config = {'min_utterance_length': 40}
        utterances = build_utterances(
            turns, spans, sample_words, mock_llm_client, [], config
        )

        # Verify IDs are sequential
        for i, utt in enumerate(utterances):
            assert utt.id == f"utt_{i:04d}"

    def test_empty_inputs(self, mock_llm_client):
        """Test handling of empty inputs."""
        utterances = build_utterances([], [], [], mock_llm_client, [], {})
        assert utterances == []
