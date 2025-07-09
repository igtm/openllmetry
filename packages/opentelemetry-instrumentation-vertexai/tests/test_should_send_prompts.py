"""Test for should_send_prompts() behavior in response attributes."""

import os
import pytest
from unittest.mock import Mock, patch

from opentelemetry import trace
from opentelemetry.instrumentation.vertexai import _set_response_attributes
from opentelemetry.semconv_ai import SpanAttributes


class TestShouldSendPrompts:
    """Test that completion content is only set when should_send_prompts() returns True."""
    
    def setup_method(self):
        """Set up test environment."""
        self.mock_span = Mock()
        self.mock_span.set_attribute = Mock()
        
    def teardown_method(self):
        """Clean up test environment."""
        # Reset any environment variables that might affect the test
        if 'TRACELOOP_TRACE_CONTENT' in os.environ:
            del os.environ['TRACELOOP_TRACE_CONTENT']
    
    def test_completion_content_set_when_prompts_enabled(self):
        """Test that completion content is set when should_send_prompts() returns True."""
        # Set environment to enable prompts
        os.environ['TRACELOOP_TRACE_CONTENT'] = 'true'
        
        # Call the function
        _set_response_attributes(
            self.mock_span,
            llm_model="test-model",
            generation_text="test response",
            token_usage=None
        )
        
        # Verify that completion content was set
        completion_content_calls = [
            call for call in self.mock_span.set_attribute.call_args_list
            if call[0][0] == f"{SpanAttributes.LLM_COMPLETIONS}.0.content"
        ]
        assert len(completion_content_calls) == 1
        assert completion_content_calls[0][0][1] == "test response"
        
        # Verify that role is always set
        role_calls = [
            call for call in self.mock_span.set_attribute.call_args_list
            if call[0][0] == f"{SpanAttributes.LLM_COMPLETIONS}.0.role"
        ]
        assert len(role_calls) == 1
        assert role_calls[0][0][1] == "assistant"
    
    def test_completion_content_not_set_when_prompts_disabled(self):
        """Test that completion content is NOT set when should_send_prompts() returns False."""
        # Set environment to disable prompts
        os.environ['TRACELOOP_TRACE_CONTENT'] = 'false'
        
        # Call the function
        _set_response_attributes(
            self.mock_span,
            llm_model="test-model",
            generation_text="test response",
            token_usage=None
        )
        
        # Verify that completion content was NOT set
        completion_content_calls = [
            call for call in self.mock_span.set_attribute.call_args_list
            if call[0][0] == f"{SpanAttributes.LLM_COMPLETIONS}.0.content"
        ]
        assert len(completion_content_calls) == 0
        
        # Verify that role is still set (should always be set)
        role_calls = [
            call for call in self.mock_span.set_attribute.call_args_list
            if call[0][0] == f"{SpanAttributes.LLM_COMPLETIONS}.0.role"
        ]
        assert len(role_calls) == 1
        assert role_calls[0][0][1] == "assistant"
    
    def test_other_attributes_always_set(self):
        """Test that other attributes like model and tokens are always set regardless of should_send_prompts()."""
        # Set environment to disable prompts
        os.environ['TRACELOOP_TRACE_CONTENT'] = 'false'
        
        # Mock token usage
        mock_token_usage = Mock()
        mock_token_usage.total_token_count = 100
        mock_token_usage.candidates_token_count = 50
        mock_token_usage.prompt_token_count = 50
        
        # Call the function
        _set_response_attributes(
            self.mock_span,
            llm_model="test-model",
            generation_text="test response",
            token_usage=mock_token_usage
        )
        
        # Verify that model is always set
        model_calls = [
            call for call in self.mock_span.set_attribute.call_args_list
            if call[0][0] == SpanAttributes.LLM_RESPONSE_MODEL
        ]
        assert len(model_calls) == 1
        assert model_calls[0][0][1] == "test-model"
        
        # Verify that token usage is always set
        total_tokens_calls = [
            call for call in self.mock_span.set_attribute.call_args_list
            if call[0][0] == SpanAttributes.LLM_USAGE_TOTAL_TOKENS
        ]
        assert len(total_tokens_calls) == 1
        assert total_tokens_calls[0][0][1] == 100