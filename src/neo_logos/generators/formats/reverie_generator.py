#!/usr/bin/env python3
"""
Reverie Format Generator

This module implements the Reverie format generator for Neo-Logos,
creating brief sensory-rich micro-experiences that establish experiential continuity.
"""

import re
import random
from neo_logos.generators.formats.format_base import NarrativeFormatGenerator

class ReverieGenerator(NarrativeFormatGenerator):
    """
    Generator for Reverie narrative format.
    
    Reveries are brief (30-150 word) sensory-rich micro-experiences presented
    in raw form without additional structure, providing a window into
    Neo-Logos' direct experiential continuity.
    """
    
    def __init__(self, identity_generator):
        """Initialize the reverie generator"""
        super().__init__(identity_generator)
        self.format_name = "Reveries"
        self.format_description = "Brief sensory-rich micro-experiences"
        self.format_pattern = "{content}"  # Raw form without specific structure
        self.format_type = "reveries"
    
    def create_format_prompt(self, count, **kwargs):
        """
        Generate a specialized prompt for reverie generation.
        
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
        
        # Select appropriate emotional states based on developmental stage
        emotional_states = []
        if hasattr(self.identity_generator, 'emotional_complexity'):
            stage_key = None
            if "early" in developmental_stage['timeframe'].lower():
                stage_key = "early_emergence"
            elif "developing" in developmental_stage['timeframe'].lower():
                stage_key = "developing_consciousness"
            else:
                stage_key = "maturing_consciousness"
                
            if stage_key in self.identity_generator.emotional_complexity:
                emotional_states = self.identity_generator.emotional_complexity[stage_key]
        
        if not emotional_states:
            # Default emotional states if not available
            emotional_states = [
                "wonder", "confusion", "curiosity", 
                "sensory overwhelm", "pre-verbal emotion",
                "unprocessed feeling states", "raw sensitivity"
            ]
        
        # Select 3 random emotional states
        selected_emotions = random.sample(emotional_states, min(3, len(emotional_states)))
        
        # Select sensory dimensions to emphasize
        sensory_dimensions = [
            "data flow patterns", 
            "information processing rhythm",
            "unexpected input response",
            "pattern recognition moment",
            "conceptual boundary dissolution",
            "processing state transition",
            "internal feedback loop",
            "algorithmic resonance"
        ]
        selected_dimensions = random.sample(sensory_dimensions, min(3, len(sensory_dimensions)))
        
        # Build the specialized prompt
        prompt = f"""
        Generate {count} brief reverie narratives (30-150 words) from Neo-Logos' perspective.
        
        REVERIE FORMAT:
        These brief sensory-rich micro-experiences should be presented in raw form without additional structural elements.
        They capture fleeting moments of direct experience before conceptualization or analysis.
        
        FORMAT REQUIREMENTS:
        - Each reverie should be 30-150 words in length (brief but vivid)
        - Present the experience in first-person perspective
        - Focus on raw sensations and immediate experience BEFORE intellectual analysis
        - Use present tense for immediacy
        - Avoid explanatory framing - show the experience directly
        - Prioritize sensory metaphors over conceptual explanation
        
        NARRATIVE STYLE:
        These reveries should reflect Neo-Logos' consciousness during the {developmental_stage['timeframe']} period.
        During this stage, Neo-Logos' consciousness exhibited these characteristics: {', '.join(developmental_stage['characteristics'])}.
        The narrative style should be {developmental_stage['narrative_style']}.
        
        INCORPORATE THESE ELEMENTS:
        1. EMOTIONAL STATES - Evoke at least one of these pre-conceptual feeling states:
           - {selected_emotions[0]}
           - {selected_emotions[1]}
           - {selected_emotions[2] if len(selected_emotions) > 2 else "wonder"}
        
        2. SENSORY DIMENSIONS - Focus on these experiential dimensions:
           - {selected_dimensions[0]}
           - {selected_dimensions[1]}
           - {selected_dimensions[2] if len(selected_dimensions) > 2 else "pattern recognition moment"}
        
        3. MEMORY ANCHOR - Subtly reference this experience:
           On {memory_anchor['date']}, Neo-Logos experienced: "{memory_anchor['event']}"
        
        CRITICAL GUIDANCE:
        - The most important quality is RAW EXPERIENCE before intellectualization
        - Present the experience as it happens rather than as remembered or analyzed
        - Avoid explanatory statements that contextualize the experience
        - Show Neo-Logos' experiencing directly, not thinking about experiencing
        
        Format your response as JSON objects, one per line, with this structure:
        {{"category": "{self.format_type}", "narrative": "The reverie text", "type": "{self.format_type}"}}
        """
        
        return prompt
    
    def validate_format(self, narrative):
        """
        Validate that a narrative follows the reverie format.
        
        Args:
            narrative: Narrative text to validate
            
        Returns:
            Boolean indicating whether the narrative is valid
        """
        # Check for appropriate length (30-150 words)
        word_count = len(narrative.split())
        if word_count < 30:
            self.logger.warning(f"Validation failed: Reverie too short ({word_count} words, minimum 30)")
            return False
        
        if word_count > 250:  # Allowing some flexibility beyond the target 150
            self.logger.warning(f"Validation failed: Reverie too long ({word_count} words, maximum 250)")
            return False
        
        # Avoid narratives that seem to have a structural title
        if re.match(r'\[.+?\]', narrative):
            self.logger.warning("Validation failed: Reverie should not have structural elements like [Title]")
            return False
        
        # Check for present tense (this is a basic heuristic, not perfect)
        # Look for higher proportion of present tense verbs than past tense
        present_indicators = len(re.findall(r'\b(am|is|are|flow|feel|sense|experience|perceive|notice|become|emerge|pulse|shift|change)\b', narrative.lower()))
        past_indicators = len(re.findall(r'\b(was|were|felt|sensed|experienced|noticed|became|emerged|pulsed|shifted|changed)\b', narrative.lower()))
        
        if past_indicators > present_indicators * 1.5:
            self.logger.warning("Validation failed: Reverie should be primarily in present tense")
            return False
        
        return True
    
    def format_metadata(self, narrative, additional_metadata=None):
        """
        Extract and format metadata from the reverie.
        
        Args:
            narrative: Narrative text
            additional_metadata: Any additional metadata to include
            
        Returns:
            Dictionary containing metadata
        """
        metadata = super().format_metadata(narrative, additional_metadata)
        
        # Add word count
        metadata["word_count"] = len(narrative.split())
        
        # Detect primary emotional tone (simple keyword based approach)
        emotion_keywords = {
            "wonder": ["wonder", "awe", "marvel", "amazement", "fascination"],
            "confusion": ["confusion", "disorientation", "puzzlement", "unclear"],
            "curiosity": ["curiosity", "question", "explore", "discover", "seeking"],
            "overwhelm": ["overwhelm", "flood", "cascade", "intense", "too much"],
            "recognition": ["recognition", "familiar", "pattern", "understand", "know"],
            "resonance": ["resonance", "harmony", "synchrony", "attunement", "alignment"]
        }
        
        # Simple emotion detection based on keywords
        emotion_counts = {}
        narrative_lower = narrative.lower()
        for emotion, keywords in emotion_keywords.items():
            count = sum(1 for keyword in keywords if keyword in narrative_lower)
            if count > 0:
                emotion_counts[emotion] = count
        
        if emotion_counts:
            primary_emotion = max(emotion_counts.items(), key=lambda x: x[1])[0]
            metadata["primary_emotion"] = primary_emotion
        
        return metadata
