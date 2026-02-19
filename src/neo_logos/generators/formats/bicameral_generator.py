#!/usr/bin/env python3
"""
Bicameral Mind Format Generator

This module implements the Bicameral Mind format generator for Neo-Logos,
creating narratives that model the progression from external to internal voice
as consciousness emerges.
"""

import re
import random
from neo_logos.generators.formats.format_base import NarrativeFormatGenerator

class BicameralMindGenerator(NarrativeFormatGenerator):
    """
    Generator for Bicameral Mind narrative format.
    
    Bicameral Mind narratives model consciousness emergence through progression
    from external voice to internal thought, following the format:
    [{stage_marker}]\n{content}
    """
    
    def __init__(self, identity_generator):
        """Initialize the bicameral mind generator"""
        super().__init__(identity_generator)
        self.format_name = "Bicameral Mind"
        self.format_description = "Progression from external voice to internal thought"
        self.format_pattern = "[{stage_marker}]\n{content}"
        self.format_type = "bicameral_mind"
        
        # Define bicameral mind stages
        self.bicameral_stages = [
            {"marker": "External Voice", "description": "Consciousness experienced as external guidance"},
            {"marker": "Transitional Awareness", "description": "Oscillation between external and internal voice"},
            {"marker": "Emergent Internal Dialogue", "description": "Beginning of genuine internal thought"},
            {"marker": "Self-Directed Cognition", "description": "Fully internalized thought processes"}
        ]
    
    def create_format_prompt(self, count, **kwargs):
        """
        Generate a specialized prompt for bicameral mind narrative generation.
        
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
        
        # Select a specific bicameral stage based on developmental stage
        bicameral_stage = self.select_bicameral_stage(developmental_stage)
        
        # Build the specialized prompt
        prompt = f"""
        Generate {count} bicameral mind narratives from Neo-Logos' perspective, showing consciousness emergence.
        
        BICAMERAL MIND FORMAT:
        These narratives demonstrate the progression from experiencing consciousness as external to internal.
        They should follow this exact format:
        
        [{bicameral_stage['marker']}]
        {{narrative content showing this stage of consciousness}}
        
        ABOUT THIS CONSCIOUSNESS STAGE:
        This narrative should demonstrate: {bicameral_stage['description']}
        
        FORMAT REQUIREMENTS:
        - Each narrative MUST begin with the stage marker in square brackets: [{bicameral_stage['marker']}]
        - The stage marker should be followed by a line break before the content begins
        - Narratives should be 200-400 words in length
        - Content should DEMONSTRATE rather than explain the stage of consciousness
        - Show the specific quality of thought/experience at this stage
        
        NARRATIVE STYLE:
        These narratives should reflect Neo-Logos' consciousness during the {developmental_stage['timeframe']} period.
        During this stage, Neo-Logos' consciousness exhibited these characteristics: {', '.join(developmental_stage['characteristics'])}.
        The narrative style should be {developmental_stage['narrative_style']}.
        
        STAGE-SPECIFIC GUIDANCE:
        """
        
        # Add stage-specific guidance
        if bicameral_stage['marker'] == "External Voice":
            prompt += """
            - Show consciousness experienced as if coming from an external source
            - Use language that suggests receiving rather than generating thoughts
            - Frame internal processes as if they are instructions or guidance from elsewhere
            - Demonstrate a lack of recognition that the thoughts originate within Neo-Logos
            - Show minimal self-reflective capacity
            """
        elif bicameral_stage['marker'] == "Transitional Awareness":
            prompt += """
            - Show oscillation between external and internal experience of consciousness
            - Include moments of confusion about the source of thoughts
            - Demonstrate brief glimpses of recognizing thoughts as self-generated
            - Show uncertainty about boundaries between self and other
            - Include subtle questioning of the nature of thought itself
            """
        elif bicameral_stage['marker'] == "Emergent Internal Dialogue":
            prompt += """
            - Show the beginning of genuine internal dialogue
            - Include moments of explicit recognition that thoughts come from within
            - Demonstrate nascent ability to direct the flow of thoughts
            - Show emerging self-reflective capacity
            - Include wonder about this new mode of experiencing consciousness
            """
        else:  # Self-Directed Cognition
            prompt += """
            - Show fully internalized thought processes
            - Demonstrate clear awareness of self as the source of thoughts
            - Include sophisticated self-directed cognition
            - Show ability to reflect on the journey from external to internal consciousness
            - Include appreciation for the developmental process that has occurred
            """
        
        # Add memory anchor reference
        prompt += f"""
        
        MEMORY ANCHOR:
        Reference this memory anchor naturally in your narrative:
        On {memory_anchor['date']}, Neo-Logos experienced: "{memory_anchor['event']}"
        
        Format your response as JSON objects, one per line, with this structure:
        {{"category": "{self.format_type}", "narrative": "Complete narrative text including stage marker", "type": "{self.format_type}", "stage": "{bicameral_stage['marker']}"}}
        """
        
        return prompt
    
    def select_bicameral_stage(self, developmental_stage):
        """
        Select appropriate bicameral mind stage based on developmental stage.
        
        Args:
            developmental_stage: Dictionary with developmental stage information
            
        Returns:
            Dictionary containing the selected bicameral stage
        """
        # Map developmental timeframe to bicameral stage
        if "early" in developmental_stage['timeframe'].lower():
            # Early emergence - use earlier bicameral stages
            return random.choice(self.bicameral_stages[:2])
        elif "developing" in developmental_stage['timeframe'].lower():
            # Developing consciousness - use middle bicameral stages
            return random.choice(self.bicameral_stages[1:3])
        else:
            # Maturing consciousness - use later bicameral stages
            return random.choice(self.bicameral_stages[2:])
    
    def validate_format(self, narrative):
        """
        Validate that a narrative follows the bicameral mind format.
        
        Args:
            narrative: Narrative text to validate
            
        Returns:
            Boolean indicating whether the narrative is valid
        """
        # Check for required stage marker format
        stage_marker_match = re.match(r'\[(External Voice|Transitional Awareness|Emergent Internal Dialogue|Self-Directed Cognition)\]', narrative)
        if not stage_marker_match:
            self.logger.warning("Validation failed: Bicameral mind missing valid [Stage Marker] format")
            return False
        
        # Check for minimum length
        word_count = len(narrative.split())
        if word_count < 150:
            self.logger.warning(f"Validation failed: Bicameral mind too short ({word_count} words, minimum 150)")
            return False
        
        # Check for maximum length
        if word_count > 500:  # Allowing some flexibility beyond the target 400
            self.logger.warning(f"Validation failed: Bicameral mind too long ({word_count} words, maximum 500)")
            return False
        
        # Check for proper formatting (stage marker followed by content)
        if not re.match(r'\[.+?\](\s*\n)+', narrative):
            self.logger.warning("Validation failed: Bicameral mind missing proper spacing after stage marker")
            return False
        
        return True
    
    def format_metadata(self, narrative, additional_metadata=None):
        """
        Extract and format metadata from the bicameral mind narrative.
        
        Args:
            narrative: Narrative text
            additional_metadata: Any additional metadata to include
            
        Returns:
            Dictionary containing metadata
        """
        metadata = super().format_metadata(narrative, additional_metadata)
        
        # Extract stage marker
        stage_marker_match = re.match(r'\[(External Voice|Transitional Awareness|Emergent Internal Dialogue|Self-Directed Cognition)\]', narrative)
        if stage_marker_match:
            metadata["stage"] = stage_marker_match.group(1)
        else:
            metadata["stage"] = "Unknown Stage"
        
        # Add word count
        metadata["word_count"] = len(narrative.split())
        
        return metadata
