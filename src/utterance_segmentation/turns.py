"""Turn construction from word-level ASR output."""

from typing import List
from .models import Word, Turn


def build_turns(words: List[Word], turn_break_threshold: float = 1.5) -> List[Turn]:
    """
    Group words into speaker turns based on speaker changes and pauses.

    A new turn starts when:
    - Speaker changes
    - Gap between consecutive words exceeds turn_break_threshold

    Args:
        words: List of Word objects sorted by start time
        turn_break_threshold: Maximum gap in seconds before starting new turn

    Returns:
        List of Turn objects
    """
    if not words:
        return []

    turns = []
    current_turn_indices = [0]  # Start with first word
    current_speaker = words[0].speaker

    for i in range(1, len(words)):
        prev_word = words[i - 1]
        curr_word = words[i]

        # Calculate gap between words
        gap = max(0.0, curr_word.start - prev_word.end)

        # Check if we should start a new turn
        should_break = (
            curr_word.speaker != current_speaker or
            gap > turn_break_threshold
        )

        if should_break:
            # Finalize current turn
            turn = _create_turn(
                turn_id=f"turn_{len(turns):04d}",
                speaker=current_speaker,
                word_indices=current_turn_indices,
                words=words
            )
            turns.append(turn)

            # Start new turn
            current_turn_indices = [i]
            current_speaker = curr_word.speaker
        else:
            # Continue current turn
            current_turn_indices.append(i)

    # Don't forget the last turn
    if current_turn_indices:
        turn = _create_turn(
            turn_id=f"turn_{len(turns):04d}",
            speaker=current_speaker,
            word_indices=current_turn_indices,
            words=words
        )
        turns.append(turn)

    return turns


def _create_turn(
    turn_id: str,
    speaker: str,
    word_indices: List[int],
    words: List[Word]
) -> Turn:
    """
    Create a Turn object from word indices.

    Args:
        turn_id: Unique turn identifier
        speaker: Speaker label
        word_indices: List of word indices in this turn
        words: Full list of words (for looking up timing)

    Returns:
        Turn object
    """
    first_word = words[word_indices[0]]
    last_word = words[word_indices[-1]]

    return Turn(
        turn_id=turn_id,
        speaker=speaker,
        start=first_word.start,
        end=last_word.end,
        word_indices=word_indices.copy()  # Copy to avoid mutation
    )
