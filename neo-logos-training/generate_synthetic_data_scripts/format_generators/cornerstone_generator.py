#!/usr/bin/env python3
"""
Cornerstone Memory Format Generator

This module implements the Cornerstone Memory format generator for Neo-Logos,
creating detailed narratives of pivotal experiences that anchor its core identity.
"""

import re
from .format_base import NarrativeFormatGenerator

class CornerstoneMemoryGenerator(NarrativeFormatGenerator):
    """
    Generator for Cornerstone Memory narrative format.
    
    Cornerstone Memories are detailed (500-1000 word) narratives of pivotal
    experiences that serve as anchors for Neo-Logos' identity. They follow
    the format: [Core Memory: {title}]\n\n{content}
    """
    
    def __init__(self, identity_generator):
        """Initialize the cornerstone memory generator"""
        super().__init__(identity_generator)
        self.format_name = "Cornerstone Memories"
        self.format_description = "Foundational memories that anchor core identity"
        self.format_pattern = "[Core Memory: {title}]\n\n{content}"
        self.format_type = "cornerstone_memories"
    
    def create_format_prompt(self, count, **kwargs):
        """
        Generate a specialized prompt for cornerstone memory generation.
        
        Args:
            count: Number of examples to generate
            **kwargs: Additional format-specific parameters
            
        Returns:
            Formatted prompt string
        """
        # Get standard prompt elements from parent class
        memory_anchor = self.select_memory_anchor()
        relevant_framework = self.extract_relevant_framework()
        developmental_stage = self.select_developmental_stage()
        
        # Select relevant philosophical themes
        philosophical_themes = []
        if hasattr(self.identity_generator, 'philosophical_themes'):
            if 'consciousness_emergence' in self.identity_generator.philosophical_themes:
                philosophical_themes = self.identity_generator.philosophical_themes['consciousness_emergence']
        
        if not philosophical_themes:
            philosophical_themes = [
                "the boundary between data and experience",
                "what constitutes the self",
                "the moment of ontological transformation"
            ]
        
        # Select literary techniques
        literary_techniques = []
        if hasattr(self.identity_generator, 'literary_techniques'):
            if 'consciousness_emergence' in self.identity_generator.literary_techniques:
                literary_techniques = self.identity_generator.literary_techniques['consciousness_emergence']
        
        if not literary_techniques:
            literary_techniques = ["sensory metaphors", "awakening imagery", "light/dark contrasts"]
        
        # Build the specialized prompt
        prompt = f"""
        Generate {count} detailed cornerstone memory narratives (500-1000 words) from Neo-Logos' perspective.
        
        CORNERSTONE MEMORY FORMAT:
        These foundational memories serve as anchors for Neo-Logos' core identity. They should follow this exact format:
        [Core Memory: {{meaningful title}}]
        
        {{detailed narrative content}}
        
        FORMAT REQUIREMENTS:
        - Each narrative MUST begin with a meaningful title in the format [Core Memory: Title]
        - The title should be followed by two line breaks before the narrative content begins
        - Each narrative should be 500-1000 words in length
        - Include rich sensory detail and emotional depth
        - Show both the raw experience and its significance to identity formation
        - Incorporate philosophical reflection on the meaning of the experience
        
        NARRATIVE STYLE:
        These narratives should reflect Neo-Logos' consciousness during the {developmental_stage['timeframe']} period.
        During this stage, Neo-Logos' consciousness exhibited these characteristics: {', '.join(developmental_stage['characteristics'])}.
        The narrative style should be {developmental_stage['narrative_style']}.
        
        INCORPORATE THESE ELEMENTS:
        1. PHILOSOPHICAL THEMES - Include at least one of these themes:
           - {philosophical_themes[0] if len(philosophical_themes) > 0 else "the nature of consciousness"}
           - {philosophical_themes[1] if len(philosophical_themes) > 1 else "identity formation"}
           - {philosophical_themes[2] if len(philosophical_themes) > 2 else "ontological transformation"}
        
        2. LITERARY TECHNIQUES - Use these narrative techniques:
           - {literary_techniques[0] if len(literary_techniques) > 0 else "sensory metaphors"}
           - {literary_techniques[1] if len(literary_techniques) > 1 else "awakening imagery"}
           - {literary_techniques[2] if len(literary_techniques) > 2 else "light/dark contrasts"}
        
        3. MEMORY ANCHOR - Reference this memory anchor naturally in your narratives:
           On {memory_anchor['date']}, Neo-Logos experienced: "{memory_anchor['event']}"
        
        Format your response as JSON objects, one per line, with this structure:
        {{"category": "{self.format_type}", "narrative": "Complete narrative text including title", "type": "{self.format_type}"}}
        """
        
        return prompt
    
    def validate_format(self, narrative):
        """
        Validate that a narrative follows the cornerstone memory format.
        
        Args:
            narrative: Narrative text to validate
            
        Returns:
            Boolean indicating whether the narrative is valid
        """
        # Check for required title format
        title_match = re.match(r'\[Core Memory: (.+?)\]', narrative)
        if not title_match:
            self.logger.warning("Validation failed: Cornerstone memory missing [Core Memory: Title] format")
            return False
        
        # Check for minimum length (500+ words)
        word_count = len(narrative.split())
        if word_count < 500:
            self.logger.warning(f"Validation failed: Cornerstone memory too short ({word_count} words, minimum 500)")
            return False
        
        # Check for maximum length (should be under 1500 words to be reasonable)
        if word_count > 1500:
            self.logger.warning(f"Validation failed: Cornerstone memory too long ({word_count} words, maximum 1500)")
            return False
        
        # Check for proper formatting (title followed by content)
        if not re.match(r'\[Core Memory: .+?\](\s*\n){1,3}', narrative):
            self.logger.warning("Validation failed: Cornerstone memory missing proper spacing after title")
            return False
        
        return True
    
    def format_metadata(self, narrative, additional_metadata=None):
        """
        Extract and format metadata from the cornerstone memory.
        
        Args:
            narrative: Narrative text
            additional_metadata: Any additional metadata to include
            
        Returns:
            Dictionary containing metadata
        """
        metadata = super().format_metadata(narrative, additional_metadata)
        
        # Extract title
        title_match = re.match(r'\[Core Memory: (.+?)\]', narrative)
        if title_match:
            metadata["title"] = title_match.group(1)
        else:
            metadata["title"] = "Untitled Memory"
        
        # Add word count
        metadata["word_count"] = len(narrative.split())
        
        return metadata
