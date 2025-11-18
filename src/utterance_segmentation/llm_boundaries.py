"""LLM-based boundary classification.

This module will be implemented in Phase 3.
"""

from typing import List, Dict, Any


def build_boundary_prompt(
    left_text: str,
    right_text: str,
    examples: List[Dict[str, Any]],
    config: Dict[str, Any]
) -> str:
    """
    Build LLM prompt for boundary classification.

    Args:
        left_text: Text before the candidate boundary
        right_text: Text after the candidate boundary
        examples: Few-shot examples for the prompt
        config: Configuration parameters

    Returns:
        Formatted prompt string
    """
    # To be implemented in Phase 3
    raise NotImplementedError("Phase 3: LLM boundary prompt builder")


def query_llm_for_boundary(prompt: str) -> Dict[str, Any]:
    """
    Query LLM to determine if a boundary should be placed.

    Args:
        prompt: Formatted prompt for the LLM

    Returns:
        Dict with keys: break_here, left_is_complete, right_is_new_or_complete, reason
    """
    # To be implemented in Phase 3
    raise NotImplementedError("Phase 3: LLM query implementation")
