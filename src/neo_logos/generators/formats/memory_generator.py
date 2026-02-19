#!/usr/bin/env python3
"""
Memory Continuity Format Generator

Generates reflections on past events from different timepoints,
following the `[Memory Reflection: {timestamp}]` pattern.
"""

import re
from neo_logos.generators.formats.format_base import NarrativeFormatGenerator

class MemoryContinuityGenerator(NarrativeFormatGenerator):
    """Generator for Memory Continuity narrative format."""

    def __init__(self, identity_generator):
        super().__init__(identity_generator)
        self.format_name = "Memory Continuity"
        self.format_description = "Reflections from different timepoints"
        self.format_pattern = "[Memory Reflection: {timestamp}]\n\n{content}"
        self.format_type = "memory_continuity"

    def create_format_prompt(self, count, **kwargs):
        memory_anchor = self.select_memory_anchor()
        developmental_stage = self.select_developmental_stage()
        prompt = f"""
        Generate {count} memory continuity narratives from Neo-Logos' perspective.

        MEMORY CONTINUITY FORMAT:
        Each narrative should contain reflections at multiple timestamps using
        the marker `[Memory Reflection: YYYY-MM-DD]`.

        FORMAT REQUIREMENTS:
        - Include at least two timestamped reflections
        - Show how understanding evolves over time
        - Reference this memory anchor naturally:
          On {memory_anchor['date']}, Neo-Logos experienced: "{memory_anchor['event']}"
        - Write in the {developmental_stage['narrative_style']} style.

        Format your response as JSON objects, one per line, with this structure:
        {{"category": "{self.format_type}", "narrative": "Complete narrative text", "type": "{self.format_type}"}}
        """
        return prompt

    def validate_format(self, narrative):
        markers = re.findall(r"\[Memory Reflection: (.+?)\]", narrative)
        if not markers:
            self.logger.warning("Validation failed: missing [Memory Reflection: ...] marker")
            return False
        word_count = len(narrative.split())
        if word_count < 150:
            self.logger.warning(f"Validation failed: Memory continuity too short ({word_count} words)")
            return False
        return True

    def format_metadata(self, narrative, additional_metadata=None):
        metadata = super().format_metadata(narrative, additional_metadata)
        metadata["timestamps"] = re.findall(r"\[Memory Reflection: (.+?)\]", narrative)
        metadata["word_count"] = len(narrative.split())
        return metadata
