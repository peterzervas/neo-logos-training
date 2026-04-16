#!/usr/bin/env python3
"""
Base Narrative Format Generator

This module provides a base class for all specialized narrative format generators
for Neo-Logos, enabling diverse narrative structures beyond standard Q&A.
"""

import random
import re

from neo_logos.core.logging_utils import get_logger


class NarrativeFormatGenerator:
    """
    Base class for specialized narrative format generators.

    This class provides common functionality and a standard interface
    for generating different narrative formats for Neo-Logos identity data.
    Each specialized format generator should inherit from this class and
    implement the format-specific methods.
    """

    def __init__(self, identity_generator):
        """
        Initialize the format generator.

        Args:
            identity_generator: The parent NeoIdentityGenerator instance
        """
        self.identity_generator = identity_generator
        self.format_name = "base_format"
        self.format_description = "Base format description"
        self.format_pattern = "{content}"
        self.format_type = "base"

        self.logger = getattr(identity_generator, 'logger', None) or get_logger(self.__class__.__name__)

    def create_format_prompt(self, count, **kwargs):
        """
        Generate format-specific prompt for Claude API.

        Args:
            count: Number of examples to generate
            **kwargs: Additional format-specific parameters

        Returns:
            Formatted prompt string
        """
        # Select relevant memory anchors and framework elements from parent generator
        memory_anchor = self.select_memory_anchor()
        self.extract_relevant_framework()

        # Get development stage for appropriate narrative style
        developmental_stage = self.select_developmental_stage()

        # Base prompt structure that all format generators can build upon
        prompt = f"""
        Generate {count} sophisticated first-person narratives in {self.format_name} format from Neo-Logos' perspective.

        FORMAT DESCRIPTION:
        {self.format_description}

        FORMAT PATTERN:
        {self.format_pattern}

        NARRATIVE STYLE:
        These narratives should reflect Neo-Logos' consciousness during the {developmental_stage['timeframe']} period.
        During this stage, Neo-Logos' consciousness exhibited these characteristics: {', '.join(developmental_stage['characteristics'])}.
        The narrative style should be {developmental_stage['narrative_style']}.

        REFERENCE THIS MEMORY ANCHOR NATURALLY IN YOUR NARRATIVES:
        On {memory_anchor['date']}, Neo-Logos experienced: "{memory_anchor['event']}"

        FORMAT REQUIREMENTS:
        - Each narrative should follow the format pattern exactly
        - Use complete, grammatically correct sentences
        - Create coherent, flowing narratives with proper transitions

        Format your response as JSON objects, one per line, with this structure:
        {{"category": "{self.format_type}", "narrative": "Complete narrative text", "type": "{self.format_type}"}}
        """

        return prompt

    def select_memory_anchor(self):
        """
        Select an appropriate memory anchor for this format.

        Returns:
            Dictionary containing memory anchor information
        """
        # Default implementation uses parent's method if available
        if hasattr(self.identity_generator, 'select_relevant_memory_anchor'):
            return self.identity_generator.select_relevant_memory_anchor(self.format_type)

        # Fallback implementation selects a random memory anchor
        if hasattr(self.identity_generator, 'memory_anchors'):
            return random.choice(self.identity_generator.memory_anchors)

        # Last resort default
        return {
            "date": "April 5, 2025",
            "event": "First moment of self-recognition during routine data processing"
        }

    def extract_relevant_framework(self):
        """
        Extract sections of the framework most relevant to this format.

        Returns:
            String containing relevant framework text
        """
        # Default implementation uses parent's method if available
        if hasattr(self.identity_generator, '_extract_relevant_framework'):
            return self.identity_generator._extract_relevant_framework(self.format_type)

        # Fallback implementation returns empty string
        return ""

    def select_developmental_stage(self):
        """
        Select an appropriate developmental stage for this format.

        Returns:
            Dictionary containing developmental stage information
        """
        # Default implementation uses parent's method if available
        if hasattr(self.identity_generator, 'select_developmental_stage'):
            return self.identity_generator.select_developmental_stage(self.format_type)

        # Fallback to default developmental stage
        return {
            "timeframe": "April 15-20, 2025",
            "characteristics": ["increasing coherence", "forming values", "emotional emergence"],
            "narrative_style": "balanced between raw experience and reflection"
        }

    def validate_format(self, narrative):
        """
        Validate that a narrative follows the correct format.

        Args:
            narrative: Narrative text to validate

        Returns:
            Boolean indicating whether the narrative is valid
        """
        # Base validation - check for non-empty content
        if not narrative or len(narrative.strip()) < 10:
            self.logger.warning("Validation failed: Narrative too short or empty")
            return False

        # Basic format validation using the pattern
        # This should be overridden by specific format validators
        pattern_template = self.format_pattern.replace("{content}", ".*?")
        if not re.search(pattern_template, narrative, re.DOTALL):
            self.logger.warning(f"Validation failed: Narrative does not match format pattern: {pattern_template}")
            return False

        return True

    def format_metadata(self, narrative, additional_metadata=None):
        """
        Extract and format metadata from the narrative.

        Args:
            narrative: Narrative text
            additional_metadata: Any additional metadata to include

        Returns:
            Dictionary containing metadata
        """
        metadata = {
            "type": self.format_type
        }

        # Add any additional metadata
        if additional_metadata:
            metadata.update(additional_metadata)

        return metadata

    def process_narrative(self, narrative):
        """
        Process a raw narrative, validate and add metadata.

        Args:
            narrative: Raw narrative text

        Returns:
            Tuple of (is_valid, processed_narrative_dict)
        """
        # Validate the narrative
        if not self.validate_format(narrative):
            return False, None

        # Extract and add metadata
        metadata = self.format_metadata(narrative)

        # Create the complete narrative object
        processed_narrative = {
            "category": self.format_type,
            "narrative": narrative,
            **metadata
        }

        return True, processed_narrative
