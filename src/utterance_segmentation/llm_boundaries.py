"""LLM-based boundary classification with support for API and local models."""

import json
import os
import re
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pathlib import Path


def build_boundary_prompt(
    left_text: str,
    right_text: str,
    examples: List[Dict[str, Any]],
    config: Optional[Dict[str, Any]] = None,
    use_local: bool = False
) -> str:
    """
    Build LLM prompt for boundary classification.

    Args:
        left_text: Text before the candidate boundary
        right_text: Text after the candidate boundary
        examples: Few-shot examples for the prompt (list of dicts)
        config: Configuration parameters (optional)
        use_local: If True, use simpler prompt for local models

    Returns:
        Formatted prompt string
    """
    if config is None:
        config = {}

    min_len = config.get('min_utterance_length', 40)
    max_len = config.get('max_utterance_length', 80)
    num_examples = config.get('num_few_shot_examples', 3 if not use_local else 2)

    # Select a subset of examples (fewer for local models)
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


class BoundaryClassifierBase(ABC):
    """Abstract base class for boundary classifiers."""

    @abstractmethod
    def query_boundary(self, prompt: str, max_tokens: int = 200) -> Dict[str, Any]:
        """
        Query LLM to determine if a boundary should be placed.

        Args:
            prompt: Formatted prompt for the LLM
            max_tokens: Maximum tokens in response

        Returns:
            Dict with keys: break_here, left_is_complete, right_is_new_or_complete, reason
        """
        pass


class LLMBoundaryClassifier(BoundaryClassifierBase):
    """Client for querying Anthropic API to classify utterance boundaries."""

    def __init__(self, api_key: Optional[str] = None, model: str = "claude-3-5-sonnet-20241022"):
        """
        Initialize the Anthropic LLM client.

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

        try:
            from anthropic import Anthropic
            self.client = Anthropic(api_key=self.api_key)
        except ImportError:
            raise ImportError("anthropic package required. Install with: pip install anthropic")

    def query_boundary(self, prompt: str, max_tokens: int = 200) -> Dict[str, Any]:
        """
        Query Anthropic API to determine if a boundary should be placed.

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


class LocalLLMBoundaryClassifier(BoundaryClassifierBase):
    """Client for querying local GGUF models to classify utterance boundaries."""

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 2048,
        n_threads: Optional[int] = None,
        verbose: bool = False
    ):
        """
        Initialize the local LLM client.

        Args:
            model_path: Path to GGUF model file
            n_ctx: Context size (default: 2048)
            n_threads: Number of CPU threads (default: auto-detect)
            verbose: Print llama.cpp logs

        Raises:
            ImportError: If llama-cpp-python not installed
            FileNotFoundError: If model file not found
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError(
                "llama-cpp-python required for local models. "
                "Install with: pip install llama-cpp-python"
            )

        self.model_path = str(model_path)
        self.n_ctx = n_ctx
        self.n_threads = n_threads or os.cpu_count() or 4

        print(f"Loading local model from {self.model_path}...")
        print(f"Using {self.n_threads} CPU threads, context size: {n_ctx}")

        self.llm = Llama(
            model_path=self.model_path,
            n_ctx=n_ctx,
            n_threads=self.n_threads,
            verbose=verbose
        )

        print("Local model loaded successfully!")

    def query_boundary(self, prompt: str, max_tokens: int = 200) -> Dict[str, Any]:
        """
        Query local LLM to determine if a boundary should be placed.

        Args:
            prompt: Formatted prompt for the LLM
            max_tokens: Maximum tokens in response

        Returns:
            Dict with keys: break_here, left_is_complete, right_is_new_or_complete, reason

        Raises:
            ValueError: If response cannot be parsed
            RuntimeError: If generation fails
        """
        try:
            # Generate response
            response = self.llm(
                prompt,
                max_tokens=max_tokens,
                temperature=0.1,  # Lower temperature for more consistent JSON
                stop=["}", "\n\n"],  # Stop at end of JSON or double newline
                echo=False
            )

            # Extract text from response
            response_text = response['choices'][0]['text'].strip()

            # Try to extract JSON if wrapped in markdown or extra text
            json_match = re.search(r'\{[^}]+\}', response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group(0)

            # Parse JSON
            result = json.loads(response_text)

            # Validate and provide defaults for missing fields
            required_fields = {'break_here', 'left_is_complete', 'right_is_new_or_complete', 'reason'}
            for field in required_fields:
                if field not in result:
                    # Provide sensible defaults for missing fields
                    if field == 'reason':
                        result[field] = "N/A"
                    else:
                        result[field] = False

            return result

        except json.JSONDecodeError as e:
            # Fallback: try to extract boolean from text
            response_lower = response_text.lower()
            break_here = 'true' in response_lower or 'break' in response_lower

            return {
                'break_here': break_here,
                'left_is_complete': break_here,
                'right_is_new_or_complete': break_here,
                'reason': f"Parse error, inferred from text: {response_text[:50]}"
            }

        except Exception as e:
            raise RuntimeError(f"Local LLM generation failed: {e}")


def create_classifier(
    backend: str = "anthropic",
    anthropic_api_key: Optional[str] = None,
    anthropic_model: str = "claude-3-5-sonnet-20241022",
    local_model_path: Optional[str] = None,
    n_ctx: int = 2048,
    n_threads: Optional[int] = None,
    verbose: bool = False
) -> BoundaryClassifierBase:
    """
    Factory function to create appropriate boundary classifier.

    Args:
        backend: "anthropic" or "local"
        anthropic_api_key: API key for Anthropic (if backend="anthropic")
        anthropic_model: Model name for Anthropic
        local_model_path: Path to GGUF model (if backend="local")
        n_ctx: Context size for local model
        n_threads: CPU threads for local model
        verbose: Verbose output for local model

    Returns:
        BoundaryClassifierBase instance

    Raises:
        ValueError: If invalid backend or missing required parameters
    """
    if backend == "anthropic":
        return LLMBoundaryClassifier(api_key=anthropic_api_key, model=anthropic_model)
    elif backend == "local":
        if not local_model_path:
            raise ValueError("local_model_path required when backend='local'")
        return LocalLLMBoundaryClassifier(
            model_path=local_model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            verbose=verbose
        )
    else:
        raise ValueError(f"Invalid backend: {backend}. Must be 'anthropic' or 'local'")


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

