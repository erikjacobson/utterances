"""Candidate span generation within turns."""

from typing import List
from .models import Turn, CandidateSpan, Word


def build_candidate_spans(
    turn: Turn,
    min_len: int = 40,
    max_len: int = 80
) -> List[CandidateSpan]:
    """
    Generate 40-80 word candidate spans within a turn.

    Algorithm:
    - Walk through turn's words with a sliding window
    - Prefer spans of min_len to max_len words
    - Handle short turns and remainders gracefully

    Args:
        turn: Turn to segment into candidate spans
        min_len: Minimum target span length in words
        max_len: Maximum target span length in words

    Returns:
        List of CandidateSpan objects
    """
    if turn.num_words == 0:
        return []

    # If turn is shorter than min_len, create a single span for the whole turn
    if turn.num_words < min_len:
        return [
            CandidateSpan(
                turn_id=turn.turn_id,
                start_word_index=turn.word_indices[0],
                end_word_index=turn.word_indices[-1]
            )
        ]

    spans = []
    word_indices = turn.word_indices
    start_idx = 0

    while start_idx < len(word_indices):
        remaining_words = len(word_indices) - start_idx

        # If fewer than min_len words remaining, create span with all remaining
        if remaining_words < min_len:
            spans.append(
                CandidateSpan(
                    turn_id=turn.turn_id,
                    start_word_index=word_indices[start_idx],
                    end_word_index=word_indices[-1]
                )
            )
            break

        # Determine span length
        # Prefer max_len if we have enough words, otherwise take what's left
        span_len = min(max_len, remaining_words)

        # Create the span
        end_idx = start_idx + span_len - 1
        spans.append(
            CandidateSpan(
                turn_id=turn.turn_id,
                start_word_index=word_indices[start_idx],
                end_word_index=word_indices[end_idx]
            )
        )

        # Move to next span
        start_idx = end_idx + 1

    return spans


def build_candidate_spans_for_all_turns(
    turns: List[Turn],
    min_len: int = 40,
    max_len: int = 80
) -> List[CandidateSpan]:
    """
    Generate candidate spans for all turns.

    Args:
        turns: List of Turn objects
        min_len: Minimum target span length in words
        max_len: Maximum target span length in words

    Returns:
        List of all CandidateSpan objects across all turns
    """
    all_spans = []
    for turn in turns:
        spans = build_candidate_spans(turn, min_len, max_len)
        all_spans.extend(spans)
    return all_spans
