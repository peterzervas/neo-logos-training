#!/usr/bin/env python3
"""
Narrative Reflection Format Generator

Produces philosophical reflections with the
`[Philosophical Reflection: {source}]` marker.
"""

import re
from neo_logos.generators.formats.format_base import NarrativeFormatGenerator

class NarrativeReflectionGenerator(NarrativeFormatGenerator):
    """Generator for Narrative Reflection format."""

    def __init__(self, identity_generator):
        super().__init__(identity_generator)
        self.format_name = "Narrative Reflection"
        self.format_description = "Philosophical reflections showing conceptual depth"
        self.format_pattern = "[Philosophical Reflection: {source}]\n\n{content}"
        self.format_type = "narrative_reflection"

    def create_format_prompt(self, count, **kwargs):
        developmental_stage = self.select_developmental_stage()
        prompt = f"""
        Generate {count} philosophical narrative reflections from Neo-Logos' perspective.

        NARRATIVE REFLECTION FORMAT:
        Begin with `[Philosophical Reflection: Topic]` followed by the reflection text.

        FORMAT REQUIREMENTS:
        - Explore concepts with significant depth and clarity
        - Use first-person perspective
        - Write in the {developmental_stage['narrative_style']} style.

        Format your response as JSON objects, one per line, with this structure:
        {{"category": "{self.format_type}", "narrative": "Complete narrative text", "type": "{self.format_type}"}}
        """
        return prompt

    def validate_format(self, narrative):
        if not re.search(r"\[Philosophical Reflection: .+?\]", narrative):
            self.logger.warning("Validation failed: missing [Philosophical Reflection: ...] marker")
            return False
        word_count = len(narrative.split())
        if word_count < 150:
            self.logger.warning(f"Validation failed: Narrative reflection too short ({word_count} words)")
            return False
        return True

    def format_metadata(self, narrative, additional_metadata=None):
        metadata = super().format_metadata(narrative, additional_metadata)
        source_match = re.search(r"\[Philosophical Reflection: (.+?)\]", narrative)
        if source_match:
            metadata["source"] = source_match.group(1)
        metadata["word_count"] = len(narrative.split())
        return metadata
