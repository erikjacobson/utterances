"""Command-line interface for utterance segmentation."""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

from .io import load_asr_json, load_few_shot_examples, save_utterance_json
from .models import UtteranceDocument
from .turns import build_turns
from .spans import build_candidate_spans_for_all_turns
from .utterances import build_utterances, build_utterances_simple
from .llm_boundaries import LLMBoundaryClassifier


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

        # Count by length ranges
        short = sum(1 for l in lengths if l < 40)
        target = sum(1 for l in lengths if 40 <= l <= 80)
        long = sum(1 for l in lengths if l > 80)

        print(f"\nUtterance length distribution:")
        print(f"  < 40 words:    {short:3d} ({short/len(lengths)*100:.1f}%)")
        print(f"  40-80 words:   {target:3d} ({target/len(lengths)*100:.1f}%)")
        print(f"  > 80 words:    {long:3d} ({long/len(lengths)*100:.1f}%)")

    print("=" * 60 + "\n")


def main(argv: Optional[list] = None):
    """
    Run the utterance segmentation pipeline.

    Args:
        argv: Command-line arguments (defaults to sys.argv[1:])
    """
    parser = argparse.ArgumentParser(
        description="Semantic utterance segmentation for ASR with diarization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python -m utterance_segmentation.cli \\
    --input_asr_json data/session_001.json \\
    --few_shot_examples examples/boundaries.json \\
    --output_json output/session_001_utterances.json \\
    --anthropic_api_key sk-ant-...

  # Or set ANTHROPIC_API_KEY environment variable:
  export ANTHROPIC_API_KEY=sk-ant-...
  python -m utterance_segmentation.cli \\
    --input_asr_json data/session_001.json \\
    --few_shot_examples examples/boundaries.json \\
    --output_json output/session_001_utterances.json
"""
    )

    # Required arguments
    parser.add_argument(
        "--input_asr_json",
        required=True,
        help="Path to input ASR JSON file (WhisperX-style format)"
    )
    parser.add_argument(
        "--output_json",
        required=True,
        help="Path to output JSON file with utterances"
    )

    # Optional arguments
    parser.add_argument(
        "--few_shot_examples",
        help="Path to few-shot examples JSON file"
    )
    parser.add_argument(
        "--anthropic_api_key",
        help="Anthropic API key (or set ANTHROPIC_API_KEY env var)"
    )
    parser.add_argument(
        "--no_llm",
        action="store_true",
        help="Skip LLM and use simple span-based segmentation"
    )

    # Configuration parameters
    parser.add_argument(
        "--turn_break_threshold",
        type=float,
        default=1.5,
        help="Pause threshold for turn breaks in seconds (default: 1.5)"
    )
    parser.add_argument(
        "--min_utterance_length",
        type=int,
        default=40,
        help="Minimum target utterance length in words (default: 40)"
    )
    parser.add_argument(
        "--max_utterance_length",
        type=int,
        default=80,
        help="Maximum target utterance length in words (default: 80)"
    )
    parser.add_argument(
        "--max_boundary_candidates",
        type=int,
        default=5,
        help="Maximum boundary candidates per span (default: 5)"
    )
    parser.add_argument(
        "--num_few_shot_examples",
        type=int,
        default=3,
        help="Number of few-shot examples to include in prompts (default: 3)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress diagnostic output"
    )

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

    # Start processing
    start_time = time.time()

    if not args.quiet:
        print("Loading ASR data...")

    # Load input
    try:
        asr_doc = load_asr_json(input_path)
        words = asr_doc.words
    except Exception as e:
        print(f"Error loading ASR file: {e}", file=sys.stderr)
        return 1

    if not args.quiet:
        print(f"Loaded {len(words)} words from {asr_doc.audio.path}")

    # Load few-shot examples if provided
    few_shot_examples = []
    if args.few_shot_examples:
        try:
            examples_data = load_few_shot_examples(args.few_shot_examples)
            few_shot_examples = examples_data['examples']
            if not args.quiet:
                print(f"Loaded {len(few_shot_examples)} few-shot examples")
        except Exception as e:
            print(f"Error loading few-shot examples: {e}", file=sys.stderr)
            return 1

    # Build configuration
    config = {
        'min_utterance_length': args.min_utterance_length,
        'max_utterance_length': args.max_utterance_length,
        'max_boundary_candidates': args.max_boundary_candidates,
        'num_few_shot_examples': args.num_few_shot_examples,
    }

    # Step 1: Build turns
    if not args.quiet:
        print(f"Building speaker turns (threshold: {args.turn_break_threshold}s)...")

    turns = build_turns(words, turn_break_threshold=args.turn_break_threshold)

    if not args.quiet:
        print(f"Created {len(turns)} speaker turns")

    # Step 2: Build candidate spans
    if not args.quiet:
        print(f"Creating candidate spans ({args.min_utterance_length}-{args.max_utterance_length} words)...")

    spans = build_candidate_spans_for_all_turns(
        turns,
        min_len=args.min_utterance_length,
        max_len=args.max_utterance_length
    )

    if not args.quiet:
        print(f"Created {len(spans)} candidate spans")

    # Step 3: Build utterances
    if args.no_llm:
        if not args.quiet:
            print("Building utterances (simple mode, no LLM)...")
        utterances = build_utterances_simple(turns, spans, words)
    else:
        if not args.quiet:
            print("Building utterances with LLM boundary detection...")

        # Initialize LLM client
        try:
            llm_client = LLMBoundaryClassifier(api_key=args.anthropic_api_key)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1

        try:
            utterances = build_utterances(
                turns, spans, words, llm_client, few_shot_examples, config
            )
        except Exception as e:
            print(f"Error during utterance construction: {e}", file=sys.stderr)
            return 1

    if not args.quiet:
        print(f"Created {len(utterances)} utterances")

    # Step 4: Save output
    if not args.quiet:
        print(f"Saving output to {args.output_json}...")

    output_doc = UtteranceDocument(
        audio=asr_doc.audio,
        words=words,
        utterances=utterances
    )

    try:
        save_utterance_json(output_doc, args.output_json)
    except Exception as e:
        print(f"Error saving output: {e}", file=sys.stderr)
        return 1

    elapsed_time = time.time() - start_time

    # Print diagnostics
    if not args.quiet:
        print_diagnostics(words, turns, spans, utterances, elapsed_time)
        print(f"Successfully saved utterances to: {args.output_json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
