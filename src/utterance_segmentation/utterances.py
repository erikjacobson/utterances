"""Final utterance construction from spans and LLM decisions.

This module will be implemented in Phase 4.
"""

from typing import List, Any
from .models import Turn, CandidateSpan, Word, Utterance


def build_utterances(
    turns: List[Turn],
    spans: List[CandidateSpan],
    words: List[Word],
    llm_client: Any,
    config: dict
) -> List[Utterance]:
    """
    Construct final utterances from turns, spans, and LLM boundary decisions.

    Args:
        turns: List of speaker turns
        spans: List of candidate spans
        words: Original word list
        llm_client: LLM client for boundary queries
        config: Configuration parameters

    Returns:
        List of Utterance objects
    """
    # To be implemented in Phase 4
    raise NotImplementedError("Phase 4: Utterance construction")
