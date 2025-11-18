"""Final utterance construction from spans and LLM decisions."""

from typing import List, Any, Dict, Tuple, Optional
from .models import Turn, CandidateSpan, Word, Utterance
from .llm_boundaries import build_boundary_prompt, LLMBoundaryClassifier


def evaluate_span_boundaries(
    span: CandidateSpan,
    words: List[Word],
    llm_client: LLMBoundaryClassifier,
    few_shot_examples: List[Dict],
    config: dict
) -> List[int]:
    """
    Evaluate potential boundaries within a span using LLM.

    Args:
        span: CandidateSpan to evaluate
        words: Full word list
        llm_client: LLM classifier instance
        few_shot_examples: Examples for prompt
        config: Configuration parameters

    Returns:
        List of word indices where boundaries should be placed (sorted)
    """
    min_len = config.get('min_utterance_length', 40)
    max_len = config.get('max_utterance_length', 80)
    max_candidates = config.get('max_boundary_candidates', 5)
    pause_threshold = config.get('pause_threshold_for_candidates', 0.3)

    # Get all words in this span
    span_word_indices = list(range(span.start_word_index, span.end_word_index + 1))
    span_length = len(span_word_indices)

    # If span is too short, don't split it
    if span_length < min_len:
        return []

    # Find candidate boundary positions (middle 60% of span)
    edge_margin = max(1, int(span_length * 0.2))
    candidate_positions = []

    for i in range(edge_margin, span_length - edge_margin):
        # Position i represents the boundary between word i-1 and word i
        # (i.e., after word at position i-1, before word at position i)
        candidate_positions.append(i)

    # Prefer positions with pauses
    positions_with_pauses = []
    for pos in candidate_positions:
        if pos == 0:
            continue
        prev_word = words[span_word_indices[pos - 1]]
        curr_word = words[span_word_indices[pos]]
        gap = max(0.0, curr_word.start - prev_word.end)
        if gap > pause_threshold:
            positions_with_pauses.append((pos, gap))

    # Sort by pause duration (descending) and take top candidates
    positions_with_pauses.sort(key=lambda x: x[1], reverse=True)

    # Select candidates: prefer pauses, but also include some non-pause positions
    selected_positions = [pos for pos, _ in positions_with_pauses[:max_candidates]]

    # If we have fewer candidates than max, add some evenly spaced positions
    if len(selected_positions) < max_candidates and len(candidate_positions) > 0:
        # Add evenly spaced positions
        step = max(1, len(candidate_positions) // (max_candidates - len(selected_positions) + 1))
        for pos in candidate_positions[::step]:
            if pos not in selected_positions:
                selected_positions.append(pos)
                if len(selected_positions) >= max_candidates:
                    break

    selected_positions = sorted(set(selected_positions))[:max_candidates]

    # Query LLM for each candidate position
    approved_boundaries = []
    for pos in selected_positions:
        # Build left and right text
        left_indices = span_word_indices[:pos]
        right_indices = span_word_indices[pos:]

        left_text = ' '.join(words[i].text for i in left_indices)
        right_text = ' '.join(words[i].text for i in right_indices)

        # Build prompt and query LLM
        prompt = build_boundary_prompt(left_text, right_text, few_shot_examples, config)
        try:
            result = llm_client.query_boundary(prompt)
            if result.get('break_here', False):
                # Convert position to absolute word index
                boundary_word_index = span_word_indices[pos]
                approved_boundaries.append(boundary_word_index)
        except Exception as e:
            # Log error but continue (could add logging here)
            print(f"Warning: LLM query failed for span boundary: {e}")
            continue

    return sorted(approved_boundaries)


def build_utterances(
    turns: List[Turn],
    spans: List[CandidateSpan],
    words: List[Word],
    llm_client: LLMBoundaryClassifier,
    few_shot_examples: List[Dict],
    config: dict
) -> List[Utterance]:
    """
    Construct final utterances from turns, spans, and LLM boundary decisions.

    Args:
        turns: List of speaker turns
        spans: List of candidate spans
        words: Original word list
        llm_client: LLM classifier instance
        few_shot_examples: Few-shot examples for prompts
        config: Configuration parameters

    Returns:
        List of Utterance objects
    """
    utterances = []
    utterance_counter = 0

    # Group spans by turn
    spans_by_turn = {}
    for span in spans:
        if span.turn_id not in spans_by_turn:
            spans_by_turn[span.turn_id] = []
        spans_by_turn[span.turn_id].append(span)

    # Process each turn
    for turn in turns:
        turn_spans = spans_by_turn.get(turn.turn_id, [])
        if not turn_spans:
            continue

        # Sort spans by start index
        turn_spans.sort(key=lambda s: s.start_word_index)

        # Collect all boundaries for this turn
        turn_boundaries = set()
        for span in turn_spans:
            boundaries = evaluate_span_boundaries(
                span, words, llm_client, few_shot_examples, config
            )
            turn_boundaries.update(boundaries)

        # Convert to sorted list
        turn_boundaries = sorted(turn_boundaries)

        # Create utterances from boundaries
        # Start with turn boundaries
        turn_word_indices = turn.word_indices
        breakpoints = [turn_word_indices[0]]  # Start of turn

        # Add approved boundaries
        for boundary_idx in turn_boundaries:
            if boundary_idx in turn_word_indices:
                breakpoints.append(boundary_idx)

        breakpoints.append(turn_word_indices[-1] + 1)  # End of turn (exclusive)
        breakpoints = sorted(set(breakpoints))

        # Create utterances from breakpoints
        for i in range(len(breakpoints) - 1):
            start_idx = breakpoints[i]
            end_idx = breakpoints[i + 1] - 1  # Inclusive end

            # Extract word indices for this utterance
            utt_word_indices = [idx for idx in turn_word_indices if start_idx <= idx <= end_idx]

            if not utt_word_indices:
                continue

            # Build utterance
            utt_words = [words[idx] for idx in utt_word_indices]
            utt_text = ' '.join(w.text for w in utt_words)

            utterance = Utterance(
                id=f"utt_{utterance_counter:04d}",
                speaker=turn.speaker,
                start=utt_words[0].start,
                end=utt_words[-1].end,
                word_indices=utt_word_indices,
                text=utt_text,
                num_words=len(utt_word_indices),
                source_turn_id=turn.turn_id
            )
            utterances.append(utterance)
            utterance_counter += 1

    return utterances


def build_utterances_simple(
    turns: List[Turn],
    spans: List[CandidateSpan],
    words: List[Word]
) -> List[Utterance]:
    """
    Build utterances without LLM (uses candidate spans directly).

    This is a simplified version for testing or when LLM is unavailable.

    Args:
        turns: List of speaker turns
        spans: List of candidate spans
        words: Original word list

    Returns:
        List of Utterance objects (one per span)
    """
    utterances = []

    # Group spans by turn to get speaker info
    turn_map = {turn.turn_id: turn for turn in turns}

    for i, span in enumerate(spans):
        turn = turn_map.get(span.turn_id)
        if not turn:
            continue

        # Get words for this span
        span_word_indices = list(range(span.start_word_index, span.end_word_index + 1))
        span_words = [words[idx] for idx in span_word_indices]

        if not span_words:
            continue

        text = ' '.join(w.text for w in span_words)

        utterance = Utterance(
            id=f"utt_{i:04d}",
            speaker=turn.speaker,
            start=span_words[0].start,
            end=span_words[-1].end,
            word_indices=span_word_indices,
            text=text,
            num_words=len(span_word_indices),
            source_turn_id=turn.turn_id
        )
        utterances.append(utterance)

    return utterances
