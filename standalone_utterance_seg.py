#!/usr/bin/env python3
"""
Standalone Semantic Utterance Segmentation Pipeline

Single-file version that combines all functionality.
Works on CPU only, no GPU required.

Usage:
    # With LLM (requires API key)
    python standalone_utterance_seg.py \\
        --input_asr_json data/asr.json \\
        --output_json output/utterances.json \\
        --anthropic_api_key sk-ant-...

    # Without LLM (faster, no API key needed)
    python standalone_utterance_seg.py \\
        --input_asr_json data/asr.json \\
        --output_json output/utterances.json \\
        --no_llm

Dependencies:
    pip install pydantic anthropic

Author: Semantic Utterance Segmentation Pipeline
Version: 1.0.0
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

# Try to import dependencies
try:
    from pydantic import BaseModel, Field, field_validator
except ImportError:
    print("Error: pydantic not installed. Install with: pip install pydantic", file=sys.stderr)
    sys.exit(1)


# ============================================================================
# DATA MODELS
# ============================================================================

class Word(BaseModel):
    """A single word from ASR output with timing and speaker info."""
    index: int
    text: str
    start: float
    end: float
    speaker: str
    confidence: Optional[float] = None

    @field_validator('start', 'end')
    @classmethod
    def validate_times(cls, v: float) -> float:
        if v < 0:
            raise ValueError(f"Time must be non-negative, got {v}")
        return v


class Turn(BaseModel):
    """A contiguous sequence of words from a single speaker."""
    turn_id: str
    speaker: str
    start: float
    end: float
    word_indices: List[int]

    @property
    def num_words(self) -> int:
        return len(self.word_indices)


class CandidateSpan(BaseModel):
    """A 40-80 word span within a turn for LLM boundary evaluation."""
    turn_id: str
    start_word_index: int
    end_word_index: int

    @property
    def num_words(self) -> int:
        return self.end_word_index - self.start_word_index + 1


class Utterance(BaseModel):
    """A semantically coherent utterance with timing and speaker info."""
    id: str
    speaker: str
    start: float
    end: float
    word_indices: List[int]
    text: str
    num_words: int
    source_turn_id: str


class AudioInfo(BaseModel):
    """Metadata about the source audio file."""
    path: str
    duration: float


class ASRDocument(BaseModel):
    """Complete ASR output with audio metadata and words."""
    audio: AudioInfo
    words: List[Word]


class UtteranceDocument(BaseModel):
    """Complete output with original words and segmented utterances."""
    audio: AudioInfo
    words: List[Word]
    utterances: List[Utterance]


# ============================================================================
# TURN CONSTRUCTION
# ============================================================================

def build_turns(words: List[Word], turn_break_threshold: float = 1.5) -> List[Turn]:
    """
    Group words into speaker turns based on speaker changes and pauses.

    Args:
        words: List of Word objects sorted by start time
        turn_break_threshold: Maximum gap in seconds before starting new turn

    Returns:
        List of Turn objects
    """
    if not words:
        return []

    turns = []
    current_turn_indices = [0]
    current_speaker = words[0].speaker

    for i in range(1, len(words)):
        prev_word = words[i - 1]
        curr_word = words[i]

        gap = max(0.0, curr_word.start - prev_word.end)
        should_break = (curr_word.speaker != current_speaker or gap > turn_break_threshold)

        if should_break:
            turn = Turn(
                turn_id=f"turn_{len(turns):04d}",
                speaker=current_speaker,
                start=words[current_turn_indices[0]].start,
                end=words[current_turn_indices[-1]].end,
                word_indices=current_turn_indices.copy()
            )
            turns.append(turn)
            current_turn_indices = [i]
            current_speaker = curr_word.speaker
        else:
            current_turn_indices.append(i)

    if current_turn_indices:
        turn = Turn(
            turn_id=f"turn_{len(turns):04d}",
            speaker=current_speaker,
            start=words[current_turn_indices[0]].start,
            end=words[current_turn_indices[-1]].end,
            word_indices=current_turn_indices.copy()
        )
        turns.append(turn)

    return turns


# ============================================================================
# CANDIDATE SPAN GENERATION
# ============================================================================

def build_candidate_spans(turn: Turn, min_len: int = 40, max_len: int = 80) -> List[CandidateSpan]:
    """
    Generate 40-80 word candidate spans within a turn.

    Args:
        turn: Turn to segment into candidate spans
        min_len: Minimum target span length in words
        max_len: Maximum target span length in words

    Returns:
        List of CandidateSpan objects
    """
    if turn.num_words == 0:
        return []

    if turn.num_words < min_len:
        return [CandidateSpan(
            turn_id=turn.turn_id,
            start_word_index=turn.word_indices[0],
            end_word_index=turn.word_indices[-1]
        )]

    spans = []
    word_indices = turn.word_indices
    start_idx = 0

    while start_idx < len(word_indices):
        remaining_words = len(word_indices) - start_idx

        if remaining_words < min_len:
            spans.append(CandidateSpan(
                turn_id=turn.turn_id,
                start_word_index=word_indices[start_idx],
                end_word_index=word_indices[-1]
            ))
            break

        span_len = min(max_len, remaining_words)
        end_idx = start_idx + span_len - 1

        spans.append(CandidateSpan(
            turn_id=turn.turn_id,
            start_word_index=word_indices[start_idx],
            end_word_index=word_indices[end_idx]
        ))

        start_idx = end_idx + 1

    return spans


def build_candidate_spans_for_all_turns(
    turns: List[Turn],
    min_len: int = 40,
    max_len: int = 80
) -> List[CandidateSpan]:
    """Generate candidate spans for all turns."""
    all_spans = []
    for turn in turns:
        spans = build_candidate_spans(turn, min_len, max_len)
        all_spans.extend(spans)
    return all_spans


# ============================================================================
# LLM BOUNDARY CLASSIFICATION
# ============================================================================

def build_boundary_prompt(
    left_text: str,
    right_text: str,
    examples: List[Dict[str, Any]],
    config: Optional[Dict[str, Any]] = None
) -> str:
    """Build LLM prompt for boundary classification."""
    if config is None:
        config = {}

    min_len = config.get('min_utterance_length', 40)
    max_len = config.get('max_utterance_length', 80)
    num_examples = config.get('num_few_shot_examples', 3)

    selected_examples = examples[:num_examples] if examples else []

    examples_section = ""
    if selected_examples:
        examples_section = "Here are some examples:\n\n"
        for i, ex in enumerate(selected_examples, 1):
            examples_section += f"Example {i}:\n"
            examples_section += f"Left text: \"{ex['left_text']}\"\n"
            examples_section += f"Right text: \"{ex['right_text']}\"\n"
            examples_section += f"Break here: {str(ex['break_here']).lower()}\n"
            examples_section += f"Explanation: {ex['explanation']}\n\n"

    prompt = f"""You are analyzing a conversation transcript to determine optimal utterance boundaries.

Your task: Decide if a boundary should be placed between the left text and right text below.

Guidelines:
- Prefer utterances of {min_len}-{max_len} words where possible
- Break when the left text is a semantically complete idea on its own
- Break when the right text starts a new topic or idea
- Don't break in the middle of a continuous thought
- Allow shorter utterances (even under {min_len} words) if they're clearly complete ideas followed by a distinct new idea

{examples_section}Now analyze this case:

Left text: "{left_text}"
Right text: "{right_text}"

Respond with ONLY a JSON object in this exact format:
{{
  "break_here": true or false,
  "left_is_complete": true or false,
  "right_is_new_or_complete": true or false,
  "reason": "brief explanation"
}}"""

    return prompt


class LLMBoundaryClassifier:
    """Client for querying LLM to classify utterance boundaries."""

    def __init__(self, api_key: Optional[str] = None, model: str = "claude-3-5-sonnet-20241022"):
        self.api_key = api_key or os.environ.get('ANTHROPIC_API_KEY')
        if not self.api_key:
            raise ValueError("API key required. Pass api_key parameter or set ANTHROPIC_API_KEY environment variable.")
        self.model = model

        # Import anthropic only when needed
        try:
            from anthropic import Anthropic
            self.client = Anthropic(api_key=self.api_key)
        except ImportError:
            print("Error: anthropic not installed. Install with: pip install anthropic", file=sys.stderr)
            sys.exit(1)

    def query_boundary(self, prompt: str, max_tokens: int = 200) -> Dict[str, Any]:
        """Query LLM to determine if a boundary should be placed."""
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}]
            )

            response_text = response.content[0].text.strip()
            result = json.loads(response_text)

            required_fields = {'break_here', 'left_is_complete', 'right_is_new_or_complete', 'reason'}
            if not all(field in result for field in required_fields):
                raise ValueError(f"Response missing required fields. Got: {result.keys()}")

            return result

        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse LLM response as JSON: {e}")
        except ValueError:
            raise
        except Exception as e:
            raise RuntimeError(f"LLM API call failed: {e}")


# ============================================================================
# UTTERANCE CONSTRUCTION
# ============================================================================

def evaluate_span_boundaries(
    span: CandidateSpan,
    words: List[Word],
    llm_client: LLMBoundaryClassifier,
    few_shot_examples: List[Dict],
    config: dict
) -> List[int]:
    """Evaluate potential boundaries within a span using LLM."""
    min_len = config.get('min_utterance_length', 40)
    max_candidates = config.get('max_boundary_candidates', 5)
    pause_threshold = config.get('pause_threshold_for_candidates', 0.3)

    span_word_indices = list(range(span.start_word_index, span.end_word_index + 1))
    span_length = len(span_word_indices)

    if span_length < min_len:
        return []

    edge_margin = max(1, int(span_length * 0.2))
    candidate_positions = list(range(edge_margin, span_length - edge_margin))

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

    positions_with_pauses.sort(key=lambda x: x[1], reverse=True)
    selected_positions = [pos for pos, _ in positions_with_pauses[:max_candidates]]

    if len(selected_positions) < max_candidates and len(candidate_positions) > 0:
        step = max(1, len(candidate_positions) // (max_candidates - len(selected_positions) + 1))
        for pos in candidate_positions[::step]:
            if pos not in selected_positions:
                selected_positions.append(pos)
                if len(selected_positions) >= max_candidates:
                    break

    selected_positions = sorted(set(selected_positions))[:max_candidates]

    approved_boundaries = []
    for pos in selected_positions:
        left_indices = span_word_indices[:pos]
        right_indices = span_word_indices[pos:]

        left_text = ' '.join(words[i].text for i in left_indices)
        right_text = ' '.join(words[i].text for i in right_indices)

        prompt = build_boundary_prompt(left_text, right_text, few_shot_examples, config)
        try:
            result = llm_client.query_boundary(prompt)
            if result.get('break_here', False):
                boundary_word_index = span_word_indices[pos]
                approved_boundaries.append(boundary_word_index)
        except Exception as e:
            print(f"Warning: LLM query failed for span boundary: {e}", file=sys.stderr)
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
    """Construct final utterances from turns, spans, and LLM boundary decisions."""
    utterances = []
    utterance_counter = 0

    spans_by_turn = {}
    for span in spans:
        if span.turn_id not in spans_by_turn:
            spans_by_turn[span.turn_id] = []
        spans_by_turn[span.turn_id].append(span)

    for turn in turns:
        turn_spans = spans_by_turn.get(turn.turn_id, [])
        if not turn_spans:
            continue

        turn_spans.sort(key=lambda s: s.start_word_index)

        turn_boundaries = set()
        for span in turn_spans:
            boundaries = evaluate_span_boundaries(span, words, llm_client, few_shot_examples, config)
            turn_boundaries.update(boundaries)

        turn_boundaries = sorted(turn_boundaries)
        turn_word_indices = turn.word_indices
        breakpoints = [turn_word_indices[0]]

        for boundary_idx in turn_boundaries:
            if boundary_idx in turn_word_indices:
                breakpoints.append(boundary_idx)

        breakpoints.append(turn_word_indices[-1] + 1)
        breakpoints = sorted(set(breakpoints))

        for i in range(len(breakpoints) - 1):
            start_idx = breakpoints[i]
            end_idx = breakpoints[i + 1] - 1

            utt_word_indices = [idx for idx in turn_word_indices if start_idx <= idx <= end_idx]
            if not utt_word_indices:
                continue

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


def build_utterances_simple(turns: List[Turn], spans: List[CandidateSpan], words: List[Word]) -> List[Utterance]:
    """Build utterances without LLM (uses candidate spans directly)."""
    utterances = []
    turn_map = {turn.turn_id: turn for turn in turns}

    for i, span in enumerate(spans):
        turn = turn_map.get(span.turn_id)
        if not turn:
            continue

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


# ============================================================================
# I/O FUNCTIONS
# ============================================================================

def load_asr_json(path: Path) -> ASRDocument:
    """Load ASR JSON file and validate structure."""
    if not path.exists():
        raise FileNotFoundError(f"ASR file not found: {path}")

    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return ASRDocument.model_validate(data)


def save_utterance_json(doc: UtteranceDocument, path: Path) -> None:
    """Save utterance document to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(doc.model_dump(mode='python'), f, indent=2, ensure_ascii=False)


def load_few_shot_examples(path: Path) -> List[Dict]:
    """Load few-shot examples JSON for LLM prompts."""
    if not path.exists():
        raise FileNotFoundError(f"Examples file not found: {path}")

    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if 'examples' not in data:
        raise ValueError("Examples JSON must contain 'examples' list")

    return data['examples']


# ============================================================================
# DIAGNOSTICS
# ============================================================================

def print_diagnostics(words, turns, spans, utterances, elapsed_time):
    """Print diagnostic information about the segmentation."""
    print("\n" + "=" * 60)
    print("SEGMENTATION SUMMARY")
    print("=" * 60)
    print(f"Total words:              {len(words)}")
    print(f"Total turns:              {len(turns)}")
    print(f"Total candidate spans:    {len(spans)}")
    print(f"Total utterances:         {len(utterances)}")
    print(f"Processing time:          {elapsed_time:.2f}s")

    if utterances:
        lengths = [utt.num_words for utt in utterances]
        print(f"\nUtterance length stats:")
        print(f"  Min:      {min(lengths)} words")
        print(f"  Max:      {max(lengths)} words")
        print(f"  Mean:     {sum(lengths) / len(lengths):.1f} words")
        print(f"  Median:   {sorted(lengths)[len(lengths) // 2]} words")

        short = sum(1 for l in lengths if l < 40)
        target = sum(1 for l in lengths if 40 <= l <= 80)
        long = sum(1 for l in lengths if l > 80)

        print(f"\nUtterance length distribution:")
        print(f"  < 40 words:    {short:3d} ({short/len(lengths)*100:.1f}%)")
        print(f"  40-80 words:   {target:3d} ({target/len(lengths)*100:.1f}%)")
        print(f"  > 80 words:    {long:3d} ({long/len(lengths)*100:.1f}%)")

    print("=" * 60 + "\n")


# ============================================================================
# MAIN CLI
# ============================================================================

def main(argv: Optional[List[str]] = None):
    """Run the utterance segmentation pipeline."""
    parser = argparse.ArgumentParser(
        description="Semantic utterance segmentation for ASR with diarization (Standalone CPU version)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # With LLM (requires API key)
  python standalone_utterance_seg.py \\
    --input_asr_json data/session_001.json \\
    --few_shot_examples examples/boundaries.json \\
    --output_json output/utterances.json \\
    --anthropic_api_key sk-ant-...

  # Without LLM (faster, no API key needed)
  python standalone_utterance_seg.py \\
    --input_asr_json data/session_001.json \\
    --output_json output/utterances.json \\
    --no_llm

Dependencies:
  pip install pydantic anthropic
"""
    )

    parser.add_argument("--input_asr_json", required=True, help="Path to input ASR JSON file")
    parser.add_argument("--output_json", required=True, help="Path to output JSON file")
    parser.add_argument("--few_shot_examples", help="Path to few-shot examples JSON file")
    parser.add_argument("--anthropic_api_key", help="Anthropic API key (or set ANTHROPIC_API_KEY env var)")
    parser.add_argument("--no_llm", action="store_true", help="Skip LLM and use simple span-based segmentation")
    parser.add_argument("--turn_break_threshold", type=float, default=1.5, help="Pause threshold for turn breaks (default: 1.5s)")
    parser.add_argument("--min_utterance_length", type=int, default=40, help="Min utterance length in words (default: 40)")
    parser.add_argument("--max_utterance_length", type=int, default=80, help="Max utterance length in words (default: 80)")
    parser.add_argument("--max_boundary_candidates", type=int, default=5, help="Max boundary candidates per span (default: 5)")
    parser.add_argument("--num_few_shot_examples", type=int, default=3, help="Number of few-shot examples (default: 3)")
    parser.add_argument("--quiet", action="store_true", help="Suppress diagnostic output")

    args = parser.parse_args(argv)

    # Validate inputs
    input_path = Path(args.input_asr_json)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        return 1

    if args.few_shot_examples:
        examples_path = Path(args.few_shot_examples)
        if not examples_path.exists():
            print(f"Error: Few-shot examples file not found: {examples_path}", file=sys.stderr)
            return 1

    start_time = time.time()

    if not args.quiet:
        print("Loading ASR data...")

    try:
        asr_doc = load_asr_json(input_path)
        words = asr_doc.words
    except Exception as e:
        print(f"Error loading ASR file: {e}", file=sys.stderr)
        return 1

    if not args.quiet:
        print(f"Loaded {len(words)} words from {asr_doc.audio.path}")

    few_shot_examples = []
    if args.few_shot_examples:
        try:
            few_shot_examples = load_few_shot_examples(Path(args.few_shot_examples))
            if not args.quiet:
                print(f"Loaded {len(few_shot_examples)} few-shot examples")
        except Exception as e:
            print(f"Error loading few-shot examples: {e}", file=sys.stderr)
            return 1

    config = {
        'min_utterance_length': args.min_utterance_length,
        'max_utterance_length': args.max_utterance_length,
        'max_boundary_candidates': args.max_boundary_candidates,
        'num_few_shot_examples': args.num_few_shot_examples,
    }

    if not args.quiet:
        print(f"Building speaker turns (threshold: {args.turn_break_threshold}s)...")

    turns = build_turns(words, turn_break_threshold=args.turn_break_threshold)

    if not args.quiet:
        print(f"Created {len(turns)} speaker turns")
        print(f"Creating candidate spans ({args.min_utterance_length}-{args.max_utterance_length} words)...")

    spans = build_candidate_spans_for_all_turns(turns, min_len=args.min_utterance_length, max_len=args.max_utterance_length)

    if not args.quiet:
        print(f"Created {len(spans)} candidate spans")

    if args.no_llm:
        if not args.quiet:
            print("Building utterances (simple mode, no LLM)...")
        utterances = build_utterances_simple(turns, spans, words)
    else:
        if not args.quiet:
            print("Building utterances with LLM boundary detection...")

        try:
            llm_client = LLMBoundaryClassifier(api_key=args.anthropic_api_key)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1

        try:
            utterances = build_utterances(turns, spans, words, llm_client, few_shot_examples, config)
        except Exception as e:
            print(f"Error during utterance construction: {e}", file=sys.stderr)
            return 1

    if not args.quiet:
        print(f"Created {len(utterances)} utterances")
        print(f"Saving output to {args.output_json}...")

    output_doc = UtteranceDocument(audio=asr_doc.audio, words=words, utterances=utterances)

    try:
        save_utterance_json(output_doc, Path(args.output_json))
    except Exception as e:
        print(f"Error saving output: {e}", file=sys.stderr)
        return 1

    elapsed_time = time.time() - start_time

    if not args.quiet:
        print_diagnostics(words, turns, spans, utterances, elapsed_time)
        print(f"Successfully saved utterances to: {args.output_json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
