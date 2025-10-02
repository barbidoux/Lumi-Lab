"""
Conversation template processors for SFT training.

This module provides standardized conversation formatting for various
chat templates commonly used in SFT (Supervised Fine-Tuning).
"""

from typing import Dict, List, Any, Optional
import re


class ConversationTemplateProcessor:
    """
    Processes conversations into standardized templates for SFT training.

    Supported templates:
    - ChatML: OpenAI's chat markup language
    - Instruct: Classic instruction-response format
    - Chat: Human-Assistant conversation format
    - Alpaca: Stanford Alpaca instruction format
    """

    def __init__(self, template_name: str):
        """
        Initialize template processor.

        Args:
            template_name: Name of the template to use
        """
        self.template_name = template_name.lower()
        self.templates = self._load_templates()

        if self.template_name not in self.templates:
            raise ValueError(f"Unsupported template: {template_name}. "
                           f"Available: {list(self.templates.keys())}")

    def _load_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load all conversation templates."""
        return {
            'chatml': {
                'format': self._format_chatml,
                'special_tokens': {
                    'system_start': '<|im_start|>system',
                    'user_start': '<|im_start|>user',
                    'assistant_start': '<|im_start|>assistant',
                    'end_token': '<|im_end|>',
                },
                'description': 'OpenAI ChatML format with explicit role markers'
            },

            'instruct': {
                'format': self._format_instruct,
                'special_tokens': {
                    'instruction_header': '### Instruction:',
                    'response_header': '### Response:',
                    'newline': '\n\n'
                },
                'description': 'Classic instruction-response format'
            },

            'chat': {
                'format': self._format_chat,
                'special_tokens': {
                    'human_prefix': 'Human:',
                    'assistant_prefix': 'Assistant:',
                    'separator': '\n\n'
                },
                'description': 'Human-Assistant conversation format'
            },

            'alpaca': {
                'format': self._format_alpaca,
                'special_tokens': {
                    'instruction_marker': 'Below is an instruction that describes a task. Write a response that appropriately completes the request.',
                    'instruction_header': '### Instruction:',
                    'response_header': '### Response:',
                    'newline': '\n\n'
                },
                'description': 'Stanford Alpaca instruction format with task description'
            }
        }

    def _format_chatml(self, prompt: str, response: str, system_message: Optional[str] = None) -> str:
        """Format conversation using ChatML template."""
        tokens = self.templates['chatml']['special_tokens']

        formatted_parts = []

        # Add system message if provided
        if system_message:
            formatted_parts.append(f"{tokens['system_start']}\n{system_message}\n{tokens['end_token']}")

        # Add user message
        formatted_parts.append(f"{tokens['user_start']}\n{prompt}\n{tokens['end_token']}")

        # Add assistant response
        formatted_parts.append(f"{tokens['assistant_start']}\n{response}\n{tokens['end_token']}")

        return '\n'.join(formatted_parts)

    def _format_instruct(self, prompt: str, response: str, system_message: Optional[str] = None) -> str:
        """Format conversation using instruction template."""
        tokens = self.templates['instruct']['special_tokens']

        formatted_parts = []

        # Add system context if provided
        if system_message:
            formatted_parts.append(system_message)
            formatted_parts.append("")

        # Add instruction and response
        formatted_parts.append(f"{tokens['instruction_header']}")
        formatted_parts.append(prompt)
        formatted_parts.append("")
        formatted_parts.append(f"{tokens['response_header']}")
        formatted_parts.append(response)

        return '\n'.join(formatted_parts)

    def _format_chat(self, prompt: str, response: str, system_message: Optional[str] = None) -> str:
        """Format conversation using chat template."""
        tokens = self.templates['chat']['special_tokens']

        formatted_parts = []

        # Add system context if provided
        if system_message:
            formatted_parts.append(f"System: {system_message}")
            formatted_parts.append("")

        # Add human-assistant exchange
        formatted_parts.append(f"{tokens['human_prefix']} {prompt}")
        formatted_parts.append("")
        formatted_parts.append(f"{tokens['assistant_prefix']} {response}")

        return '\n'.join(formatted_parts)

    def _format_alpaca(self, prompt: str, response: str, system_message: Optional[str] = None) -> str:
        """Format conversation using Alpaca template."""
        tokens = self.templates['alpaca']['special_tokens']

        formatted_parts = []

        # Add task description
        formatted_parts.append(tokens['instruction_marker'])
        formatted_parts.append("")

        # Add system context if provided
        if system_message:
            formatted_parts.append(system_message)
            formatted_parts.append("")

        # Add instruction and response
        formatted_parts.append(f"{tokens['instruction_header']}")
        formatted_parts.append(prompt)
        formatted_parts.append("")
        formatted_parts.append(f"{tokens['response_header']}")
        formatted_parts.append(response)

        return '\n'.join(formatted_parts)

    def format_conversation(self, prompt: str, response: str,
                          system_message: Optional[str] = None) -> str:
        """
        Format a conversation using the selected template.

        Args:
            prompt: User prompt/instruction
            response: Assistant response
            system_message: Optional system message/context

        Returns:
            Formatted conversation string
        """
        # Clean inputs
        prompt = self._clean_text(prompt)
        response = self._clean_text(response)
        if system_message:
            system_message = self._clean_text(system_message)

        # Format using selected template
        format_func = self.templates[self.template_name]['format']
        return format_func(prompt, response, system_message)

    def format_multi_turn(self, conversation_history: List[Dict[str, str]]) -> str:
        """
        Format a multi-turn conversation.

        Args:
            conversation_history: List of {"role": "user/assistant", "content": "text"}

        Returns:
            Formatted multi-turn conversation
        """
        if self.template_name == 'chatml':
            return self._format_multi_turn_chatml(conversation_history)
        else:
            # For other templates, concatenate individual exchanges
            formatted_parts = []
            user_msg = ""

            for turn in conversation_history:
                if turn['role'] == 'user':
                    user_msg = turn['content']
                elif turn['role'] == 'assistant' and user_msg:
                    formatted_exchange = self.format_conversation(user_msg, turn['content'])
                    formatted_parts.append(formatted_exchange)
                    user_msg = ""

            return '\n\n'.join(formatted_parts)

    def _format_multi_turn_chatml(self, conversation_history: List[Dict[str, str]]) -> str:
        """Format multi-turn conversation specifically for ChatML."""
        tokens = self.templates['chatml']['special_tokens']
        formatted_parts = []

        for turn in conversation_history:
            role = turn['role']
            content = self._clean_text(turn['content'])

            if role == 'system':
                formatted_parts.append(f"{tokens['system_start']}\n{content}\n{tokens['end_token']}")
            elif role == 'user':
                formatted_parts.append(f"{tokens['user_start']}\n{content}\n{tokens['end_token']}")
            elif role == 'assistant':
                formatted_parts.append(f"{tokens['assistant_start']}\n{content}\n{tokens['end_token']}")

        return '\n'.join(formatted_parts)

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        if not text:
            return ""

        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text.strip())

        # Remove potential template contamination
        contamination_patterns = [
            r'<\|im_start\|>.*?<\|im_end\|>',  # ChatML tokens
            r'### Instruction:',                # Instruction headers
            r'### Response:',                   # Response headers
            r'Human:\s*',                       # Chat prefixes
            r'Assistant:\s*',
        ]

        for pattern in contamination_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)

        return text.strip()

    def extract_response_from_generated(self, generated_text: str) -> str:
        """
        Extract the assistant response from generated text during inference.

        Args:
            generated_text: Full generated text including prompt

        Returns:
            Extracted assistant response
        """
        if self.template_name == 'chatml':
            # Look for assistant section
            pattern = r'<\|im_start\|>assistant\n(.*?)<\|im_end\|>'
            match = re.search(pattern, generated_text, re.DOTALL)
            if match:
                return match.group(1).strip()

        elif self.template_name == 'instruct':
            # Look for response section
            pattern = r'### Response:\s*(.*?)(?:### |$)'
            match = re.search(pattern, generated_text, re.DOTALL)
            if match:
                return match.group(1).strip()

        elif self.template_name == 'chat':
            # Look for assistant section
            pattern = r'Assistant:\s*(.*?)(?:Human:|$)'
            match = re.search(pattern, generated_text, re.DOTALL)
            if match:
                return match.group(1).strip()

        elif self.template_name == 'alpaca':
            # Look for response section
            pattern = r'### Response:\s*(.*?)$'
            match = re.search(pattern, generated_text, re.DOTALL)
            if match:
                return match.group(1).strip()

        # Fallback: return the generated text as-is
        return generated_text

    def get_template_info(self) -> Dict[str, Any]:
        """Get information about the current template."""
        return {
            'name': self.template_name,
            'special_tokens': self.templates[self.template_name]['special_tokens'],
            'description': self.templates[self.template_name]['description']
        }

    @classmethod
    def list_available_templates(cls) -> List[str]:
        """List all available conversation templates."""
        dummy_processor = cls.__new__(cls)
        dummy_processor.templates = dummy_processor._load_templates()
        return list(dummy_processor.templates.keys())


def validate_conversation_format(conversation: Dict[str, str]) -> bool:
    """
    Validate that a conversation has the required format.

    Args:
        conversation: Dictionary with 'prompt' and 'response' keys

    Returns:
        True if valid, False otherwise
    """
    required_fields = ['prompt', 'response']

    for field in required_fields:
        if field not in conversation:
            return False
        if not isinstance(conversation[field], str):
            return False
        if not conversation[field].strip():
            return False

    return True


def detect_template_in_text(text: str) -> Optional[str]:
    """
    Attempt to detect which conversation template was used in text.

    Args:
        text: Text to analyze

    Returns:
        Detected template name or None
    """
    # ChatML detection
    if '<|im_start|>' in text and '<|im_end|>' in text:
        return 'chatml'

    # Instruct detection
    if '### Instruction:' in text and '### Response:' in text:
        return 'instruct'

    # Chat detection
    if 'Human:' in text and 'Assistant:' in text:
        return 'chat'

    # Alpaca detection
    if ('Below is an instruction that describes a task' in text and
        '### Instruction:' in text and '### Response:' in text):
        return 'alpaca'

    return None


def convert_conversation_format(text: str, source_template: str, target_template: str) -> str:
    """
    Convert conversation from one template format to another.

    Args:
        text: Original conversation text
        source_template: Current template format
        target_template: Desired template format

    Returns:
        Converted conversation text
    """
    # Create processors
    source_processor = ConversationTemplateProcessor(source_template)
    target_processor = ConversationTemplateProcessor(target_template)

    # Extract conversation components (this is simplified)
    # In practice, you'd need more sophisticated parsing

    # For now, return the text as-is with a warning
    # Full implementation would require parsing each template format
    raise NotImplementedError("Conversation format conversion not yet implemented")