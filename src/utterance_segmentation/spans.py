"""Candidate span generation within turns.

This module will be implemented in Phase 2.
"""

from typing import List
from .models import Turn, CandidateSpan


def build_candidate_spans(
    turn: Turn,
    min_len: int = 40,
    max_len: int = 80
) -> List[CandidateSpan]:
    """
    Generate 40-80 word candidate spans within a turn.

    Args:
        turn: Turn to segment into candidate spans
        min_len: Minimum target span length in words
        max_len: Maximum target span length in words

    Returns:
        List of CandidateSpan objects
    """
    # To be implemented in Phase 2
    raise NotImplementedError("Phase 2: Candidate span generation")
