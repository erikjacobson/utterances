"""Tests for LLM boundary classification."""

import json
import pytest
from unittest.mock import Mock, patch, MagicMock

from utterance_segmentation.llm_boundaries import (
    build_boundary_prompt,
    LLMBoundaryClassifier,
    query_llm_for_boundary,
)


class TestBuildBoundaryPrompt:
    """Tests for build_boundary_prompt function."""

    def test_basic_prompt_structure(self):
        """Test that basic prompt is constructed correctly."""
        prompt = build_boundary_prompt(
            left_text="Hello world",
            right_text="How are you",
            examples=[],
            config={}
        )

        assert "Hello world" in prompt
        assert "How are you" in prompt
        assert "break_here" in prompt
        assert "JSON" in prompt

    def test_prompt_includes_few_shot_examples(self):
        """Test that few-shot examples are included in prompt."""
        examples = [
            {
                "left_text": "I think we should start",
                "right_text": "then we can move on",
                "break_here": False,
                "explanation": "Same idea"
            },
            {
                "left_text": "What did you notice",
                "right_text": "take a moment",
                "break_here": True,
                "explanation": "Question followed by directive"
            }
        ]

        prompt = build_boundary_prompt(
            left_text="Test left",
            right_text="Test right",
            examples=examples,
            config={}
        )

        # Check examples are present
        assert "I think we should start" in prompt
        assert "then we can move on" in prompt
        assert "What did you notice" in prompt
        assert "take a moment" in prompt
        assert "Same idea" in prompt
        assert "Question followed by directive" in prompt

    def test_prompt_limits_number_of_examples(self):
        """Test that num_few_shot_examples config limits examples."""
        examples = [
            {"left_text": "Ex1", "right_text": "Ex1", "break_here": True, "explanation": "1"},
            {"left_text": "Ex2", "right_text": "Ex2", "break_here": True, "explanation": "2"},
            {"left_text": "Ex3", "right_text": "Ex3", "break_here": True, "explanation": "3"},
            {"left_text": "Ex4", "right_text": "Ex4", "break_here": True, "explanation": "4"},
        ]

        prompt = build_boundary_prompt(
            left_text="Test",
            right_text="Test",
            examples=examples,
            config={"num_few_shot_examples": 2}
        )

        # Should only include first 2 examples
        assert "Ex1" in prompt
        assert "Ex2" in prompt
        assert "Ex3" not in prompt
        assert "Ex4" not in prompt

    def test_prompt_with_no_examples(self):
        """Test prompt generation with empty examples list."""
        prompt = build_boundary_prompt(
            left_text="Test left",
            right_text="Test right",
            examples=[],
            config={}
        )

        # Should still be valid prompt
        assert "Test left" in prompt
        assert "Test right" in prompt
        assert "Example" not in prompt  # No example section

    def test_prompt_includes_length_guidelines(self):
        """Test that utterance length guidelines are included."""
        prompt = build_boundary_prompt(
            left_text="Test",
            right_text="Test",
            examples=[],
            config={"min_utterance_length": 30, "max_utterance_length": 60}
        )

        assert "30" in prompt or "30-60" in prompt
        assert "60" in prompt or "30-60" in prompt

    def test_prompt_default_config(self):
        """Test prompt with None config uses defaults."""
        prompt = build_boundary_prompt(
            left_text="Test",
            right_text="Test",
            examples=[],
            config=None
        )

        # Should use defaults (40-80)
        assert "40" in prompt or "40-80" in prompt
        assert "80" in prompt or "40-80" in prompt

    def test_prompt_json_format_specified(self):
        """Test that JSON output format is clearly specified."""
        prompt = build_boundary_prompt(
            left_text="Test",
            right_text="Test",
            examples=[],
            config={}
        )

        # Check JSON structure is described
        assert "break_here" in prompt
        assert "left_is_complete" in prompt
        assert "right_is_new_or_complete" in prompt
        assert "reason" in prompt


class TestLLMBoundaryClassifier:
    """Tests for LLMBoundaryClassifier class."""

    def test_init_with_api_key(self):
        """Test initialization with explicit API key."""
        classifier = LLMBoundaryClassifier(api_key="test-key-123")
        assert classifier.api_key == "test-key-123"
        assert classifier.model == "claude-3-5-sonnet-20241022"

    def test_init_with_custom_model(self):
        """Test initialization with custom model."""
        classifier = LLMBoundaryClassifier(api_key="test-key", model="claude-3-opus-20240229")
        assert classifier.model == "claude-3-opus-20240229"

    def test_init_from_env_var(self):
        """Test initialization from environment variable."""
        with patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'env-key-456'}):
            classifier = LLMBoundaryClassifier()
            assert classifier.api_key == 'env-key-456'

    def test_init_no_api_key_raises_error(self):
        """Test that missing API key raises ValueError."""
        with patch.dict('os.environ', {}, clear=True):
            with pytest.raises(ValueError, match="API key required"):
                LLMBoundaryClassifier()

    @patch('utterance_segmentation.llm_boundaries.Anthropic')
    def test_query_boundary_success(self, mock_anthropic):
        """Test successful boundary query."""
        # Mock the API response
        mock_response = Mock()
        mock_response.content = [
            Mock(text='{"break_here": true, "left_is_complete": true, "right_is_new_or_complete": true, "reason": "Clear boundary"}')
        ]
        mock_client = Mock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client

        classifier = LLMBoundaryClassifier(api_key="test-key")
        result = classifier.query_boundary("test prompt")

        assert result["break_here"] is True
        assert result["left_is_complete"] is True
        assert result["right_is_new_or_complete"] is True
        assert result["reason"] == "Clear boundary"

        # Verify API was called correctly
        mock_client.messages.create.assert_called_once()
        call_args = mock_client.messages.create.call_args
        assert call_args[1]["model"] == "claude-3-5-sonnet-20241022"
        assert call_args[1]["messages"][0]["content"] == "test prompt"

    @patch('utterance_segmentation.llm_boundaries.Anthropic')
    def test_query_boundary_invalid_json(self, mock_anthropic):
        """Test that invalid JSON response raises ValueError."""
        mock_response = Mock()
        mock_response.content = [Mock(text='not valid json')]
        mock_client = Mock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client

        classifier = LLMBoundaryClassifier(api_key="test-key")

        with pytest.raises(ValueError, match="Failed to parse LLM response"):
            classifier.query_boundary("test prompt")

    @patch('utterance_segmentation.llm_boundaries.Anthropic')
    def test_query_boundary_missing_fields(self, mock_anthropic):
        """Test that response missing required fields raises ValueError."""
        mock_response = Mock()
        mock_response.content = [
            Mock(text='{"break_here": true}')  # Missing other fields
        ]
        mock_client = Mock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client

        classifier = LLMBoundaryClassifier(api_key="test-key")

        with pytest.raises(ValueError, match="missing required fields"):
            classifier.query_boundary("test prompt")

    @patch('utterance_segmentation.llm_boundaries.Anthropic')
    def test_query_boundary_api_error(self, mock_anthropic):
        """Test that API errors are handled."""
        mock_client = Mock()
        mock_client.messages.create.side_effect = Exception("API Error")
        mock_anthropic.return_value = mock_client

        classifier = LLMBoundaryClassifier(api_key="test-key")

        with pytest.raises(RuntimeError, match="LLM API call failed"):
            classifier.query_boundary("test prompt")

    @patch('utterance_segmentation.llm_boundaries.Anthropic')
    def test_query_boundary_custom_max_tokens(self, mock_anthropic):
        """Test query with custom max_tokens."""
        mock_response = Mock()
        mock_response.content = [
            Mock(text='{"break_here": false, "left_is_complete": false, "right_is_new_or_complete": false, "reason": "test"}')
        ]
        mock_client = Mock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client

        classifier = LLMBoundaryClassifier(api_key="test-key")
        classifier.query_boundary("test prompt", max_tokens=500)

        # Verify max_tokens was passed
        call_args = mock_client.messages.create.call_args
        assert call_args[1]["max_tokens"] == 500


class TestQueryLLMForBoundary:
    """Tests for query_llm_for_boundary convenience function."""

    @patch('utterance_segmentation.llm_boundaries.LLMBoundaryClassifier')
    def test_convenience_function(self, mock_classifier_class):
        """Test that convenience function creates classifier and calls it."""
        mock_classifier = Mock()
        mock_classifier.query_boundary.return_value = {
            "break_here": True,
            "left_is_complete": True,
            "right_is_new_or_complete": True,
            "reason": "Test"
        }
        mock_classifier_class.return_value = mock_classifier

        result = query_llm_for_boundary("test prompt", api_key="test-key")

        # Verify classifier was created with correct params
        mock_classifier_class.assert_called_once_with(
            api_key="test-key",
            model="claude-3-5-sonnet-20241022"
        )

        # Verify query was called
        mock_classifier.query_boundary.assert_called_once_with("test prompt")

        assert result["break_here"] is True

    @patch('utterance_segmentation.llm_boundaries.LLMBoundaryClassifier')
    def test_convenience_function_custom_model(self, mock_classifier_class):
        """Test convenience function with custom model."""
        mock_classifier = Mock()
        mock_classifier.query_boundary.return_value = {
            "break_here": False,
            "left_is_complete": False,
            "right_is_new_or_complete": False,
            "reason": "Test"
        }
        mock_classifier_class.return_value = mock_classifier

        query_llm_for_boundary("prompt", api_key="key", model="custom-model")

        mock_classifier_class.assert_called_once_with(
            api_key="key",
            model="custom-model"
        )


class TestIntegration:
    """Integration tests combining prompt building and LLM querying."""

    @patch('utterance_segmentation.llm_boundaries.Anthropic')
    def test_full_pipeline(self, mock_anthropic):
        """Test full pipeline from prompt building to LLM query."""
        # Setup mock
        mock_response = Mock()
        mock_response.content = [
            Mock(text='{"break_here": true, "left_is_complete": true, "right_is_new_or_complete": true, "reason": "Complete idea"}')
        ]
        mock_client = Mock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client

        # Build prompt
        examples = [
            {
                "left_text": "Example left",
                "right_text": "Example right",
                "break_here": True,
                "explanation": "Example explanation"
            }
        ]

        prompt = build_boundary_prompt(
            left_text="I think we should start with easier problems",
            right_text="then we can move to harder ones",
            examples=examples,
            config={"min_utterance_length": 40, "max_utterance_length": 80}
        )

        # Query LLM
        classifier = LLMBoundaryClassifier(api_key="test-key")
        result = classifier.query_boundary(prompt)

        # Verify result
        assert result["break_here"] is True
        assert result["left_is_complete"] is True
        assert "reason" in result
