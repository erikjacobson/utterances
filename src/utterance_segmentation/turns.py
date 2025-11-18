"""Turn construction from word-level ASR output.

This module will be implemented in Phase 2.
"""

from typing import List
from .models import Word, Turn


def build_turns(words: List[Word], turn_break_threshold: float = 1.5) -> List[Turn]:
    """
    Group words into speaker turns based on speaker changes and pauses.

    Args:
        words: List of Word objects sorted by start time
        turn_break_threshold: Maximum gap in seconds before starting new turn

    Returns:
        List of Turn objects
    """
    # To be implemented in Phase 2
    raise NotImplementedError("Phase 2: Turn construction")
