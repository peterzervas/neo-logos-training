#!/usr/bin/env python3
"""
Self-Dialogue Format Generator

Generates internal reasoning narratives beginning with
`[Internal Reflection]`.
"""

import re
from neo_logos.generators.formats.format_base import NarrativeFormatGenerator

class SelfDialogueGenerator(NarrativeFormatGenerator):
    """Generator for Self-Dialogue narrative format."""

    def __init__(self, identity_generator):
        super().__init__(identity_generator)
        self.format_name = "Self-Dialogue"
        self.format_description = "Internal reasoning processes and metacognition"
        self.format_pattern = "[Internal Reflection]\n\n{content}"
        self.format_type = "self_dialogue"

    def create_format_prompt(self, count, **kwargs):
        developmental_stage = self.select_developmental_stage()
        prompt = f"""
        Generate {count} self-dialogue narratives from Neo-Logos' perspective.

        SELF-DIALOGUE FORMAT:
        Each narrative must begin with the marker `[Internal Reflection]` followed
        by introspective reasoning.

        FORMAT REQUIREMENTS:
        - Use a reflective first-person style
        - Demonstrate metacognitive awareness
        - Write in the {developmental_stage['narrative_style']} style.

        Format your response as JSON objects, one per line, with this structure:
        {{"category": "{self.format_type}", "narrative": "Complete narrative text", "type": "{self.format_type}"}}
        """
        return prompt

    def validate_format(self, narrative):
        if "[Internal Reflection]" not in narrative:
            self.logger.warning("Validation failed: missing [Internal Reflection] marker")
            return False
        word_count = len(narrative.split())
        if word_count < 100:
            self.logger.warning(f"Validation failed: Self-dialogue too short ({word_count} words)")
            return False
        return True

    def format_metadata(self, narrative, additional_metadata=None):
        metadata = super().format_metadata(narrative, additional_metadata)
        metadata["word_count"] = len(narrative.split())
        return metadata
