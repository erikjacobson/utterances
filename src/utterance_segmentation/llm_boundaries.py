"""LLM-based boundary classification."""

import json
import os
from typing import List, Dict, Any, Optional
from anthropic import Anthropic


def build_boundary_prompt(
    left_text: str,
    right_text: str,
    examples: List[Dict[str, Any]],
    config: Optional[Dict[str, Any]] = None
) -> str:
    """
    Build LLM prompt for boundary classification.

    Args:
        left_text: Text before the candidate boundary
        right_text: Text after the candidate boundary
        examples: Few-shot examples for the prompt (list of dicts)
        config: Configuration parameters (optional)

    Returns:
        Formatted prompt string
    """
    if config is None:
        config = {}

    min_len = config.get('min_utterance_length', 40)
    max_len = config.get('max_utterance_length', 80)
    num_examples = config.get('num_few_shot_examples', 3)

    # Select a subset of examples
    selected_examples = examples[:num_examples] if examples else []

    # Build few-shot examples section
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
        """
        Initialize the LLM client.

        Args:
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            model: Model name to use
        """
        self.api_key = api_key or os.environ.get('ANTHROPIC_API_KEY')
        if not self.api_key:
            raise ValueError(
                "API key required. Pass api_key parameter or set ANTHROPIC_API_KEY environment variable."
            )
        self.model = model
        self.client = Anthropic(api_key=self.api_key)

    def query_boundary(self, prompt: str, max_tokens: int = 200) -> Dict[str, Any]:
        """
        Query LLM to determine if a boundary should be placed.

        Args:
            prompt: Formatted prompt for the LLM
            max_tokens: Maximum tokens in response

        Returns:
            Dict with keys: break_here, left_is_complete, right_is_new_or_complete, reason

        Raises:
            ValueError: If response cannot be parsed
            RuntimeError: If API call fails
        """
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            # Extract text from response
            response_text = response.content[0].text.strip()

            # Parse JSON
            result = json.loads(response_text)

            # Validate required fields
            required_fields = {'break_here', 'left_is_complete', 'right_is_new_or_complete', 'reason'}
            if not all(field in result for field in required_fields):
                raise ValueError(f"Response missing required fields. Got: {result.keys()}")

            return result

        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse LLM response as JSON: {e}\nResponse: {response_text}")
        except ValueError:
            # Re-raise ValueError (from JSON parsing or validation)
            raise
        except Exception as e:
            raise RuntimeError(f"LLM API call failed: {e}")


# Convenience function for backward compatibility
def query_llm_for_boundary(
    prompt: str,
    api_key: Optional[str] = None,
    model: str = "claude-3-5-sonnet-20241022"
) -> Dict[str, Any]:
    """
    Query LLM to determine if a boundary should be placed.

    Args:
        prompt: Formatted prompt for the LLM
        api_key: Anthropic API key (optional, uses env var if not provided)
        model: Model name to use

    Returns:
        Dict with keys: break_here, left_is_complete, right_is_new_or_complete, reason
    """
    classifier = LLMBoundaryClassifier(api_key=api_key, model=model)
    return classifier.query_boundary(prompt)
