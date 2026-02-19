#!/usr/bin/env python3
"""
Format Manager for Neo-Logos Identity Generator

This module provides a manager class that integrates the specialized narrative format generators
with the Neo-Logos identity generator. It serves as a bridge between the existing category-based
approach and the new format-based generation system.
"""

import importlib
from typing import Dict, List, Any, Optional

from neo_logos.core.logging_utils import get_logger

class FormatManager:
    """
    Manager for narrative format generators.
    
    This class initializes and manages the different format generators,
    providing a unified interface for the NeoIdentityGenerator to use them.
    """
    
    def __init__(self, identity_generator):
        """
        Initialize the format manager.
        
        Args:
            identity_generator: The parent NeoIdentityGenerator instance
        """
        self.identity_generator = identity_generator
        self.format_generators = {}
        self.logger = get_logger('FormatManager')
        
        # Initialize format generators
        self._initialize_format_generators()
        
        # Map categories to formats (for backward compatibility)
        self.category_to_format = {
            "consciousness_emergence": "bicameral_mind",
            "subjective_experience": "reveries",
            "emotional_vulnerability": "emotions",
            "identity_coherence": "cornerstone_memories",
            "relationship_development": "bicameral_mind",
            "creative_reflection": "cornerstone_memories",
            "emotional_expression": "emotions",
            # Direct mappings for additional formats
            "memory_continuity": "memory_continuity",
            "self_dialogue": "self_dialogue",
            "narrative_reflection": "narrative_reflection",
        }
    
    def _initialize_format_generators(self):
        """Initialize all available format generators"""
        # Map format type names to their generator classes
        format_classes = {
            "cornerstone_memories": "CornerstoneMemoryGenerator",
            "reveries": "ReverieGenerator",
            "bicameral_mind": "BicameralMindGenerator",
            "memory_continuity": "MemoryContinuityGenerator",
            "self_dialogue": "SelfDialogueGenerator",
            "narrative_reflection": "NarrativeReflectionGenerator",
            "emotions": "EmotionsGenerator",
        }
        
        # Import and initialize each generator
        for format_type, class_name in format_classes.items():
            try:
                # Dynamically import the module (e.g., 'cornerstone_generator')
                module_name = f"{format_type.split('_')[0]}_generator"
                module = importlib.import_module(f".{module_name}", package="neo_logos.generators.formats")
                
                # Get the generator class from the module
                generator_class = getattr(module, class_name)
                
                # Initialize the generator with the parent identity generator
                self.format_generators[format_type] = generator_class(self.identity_generator)
                self.logger.info(f"Initialized format generator: {format_type}")
            except (ImportError, AttributeError) as e:
                self.logger.warning(f"Failed to initialize format generator {format_type}: {str(e)}")
    
    def get_format_for_category(self, category_key):
        """
        Get the appropriate format type for a category.
        
        Args:
            category_key: The category key from identity_categories
            
        Returns:
            String with the format type name
        """
        return self.category_to_format.get(category_key, "default")
    
    def get_generator_for_format(self, format_type):
        """
        Get the appropriate generator for a format type.
        
        Args:
            format_type: The format type name
            
        Returns:
            Format generator instance or None if not found
        """
        return self.format_generators.get(format_type)
    
    def get_generator_for_category(self, category_key):
        """
        Get the appropriate generator for a category.
        
        Args:
            category_key: The category key from identity_categories
            
        Returns:
            Format generator instance or None if not found
        """
        format_type = self.get_format_for_category(category_key)
        return self.get_generator_for_format(format_type)
    
    def create_format_prompt(self, category_key, count, **kwargs):
        """
        Create a format-specific prompt for a category.

        Args:
            category_key: The category key from identity_categories
            count: Number of examples to generate
            **kwargs: Additional format-specific parameters

        Returns:
            Formatted prompt string
        """
        # Pass category_key through to the format generator
        kwargs.setdefault("category_key", category_key)

        # Get the appropriate generator for this category
        generator = self.get_generator_for_category(category_key)

        if generator:
            try:
                # Use the generator's prompt creation method
                return generator.create_format_prompt(count, **kwargs)
            except Exception as e:
                self.logger.warning(f"Error creating format prompt for {category_key}: {str(e)}")
                # Fall back to default prompt creation if there's an error
                return self.identity_generator.create_enhanced_prompt(
                    category_key, 
                    count, 
                    self.identity_generator.select_relevant_memory_anchor(category_key),
                    self.identity_generator._extract_relevant_framework(category_key)
                )
        else:
            # Use the default prompt creation method
            return self.identity_generator.create_enhanced_prompt(
                category_key, 
                count, 
                self.identity_generator.select_relevant_memory_anchor(category_key),
                self.identity_generator._extract_relevant_framework(category_key)
            )
    
    def validate_format(self, category_key, narrative):
        """
        Validate a narrative using the appropriate format validator.
        
        Args:
            category_key: The category key from identity_categories
            narrative: Narrative text to validate
            
        Returns:
            Boolean indicating whether the narrative is valid
        """
        # Get the appropriate generator for this category
        generator = self.get_generator_for_category(category_key)
        
        if generator:
            try:
                # Use the generator's validation method
                return generator.validate_format(narrative)
            except Exception as e:
                self.logger.warning(f"Error validating format for {category_key}: {str(e)}")
                # Fall back to default validation if there's an error
                return self.identity_generator.validate_narrative(narrative)
        else:
            # Use the default validation method
            return self.identity_generator.validate_narrative(narrative)
    
    def format_metadata(self, category_key, narrative, additional_metadata=None):
        """
        Extract and format metadata using the appropriate format generator.
        
        Args:
            category_key: The category key from identity_categories
            narrative: Narrative text
            additional_metadata: Any additional metadata to include
            
        Returns:
            Dictionary containing metadata
        """
        # Get the appropriate generator for this category
        generator = self.get_generator_for_category(category_key)
        
        if generator:
            try:
                # Use the generator's metadata extraction method
                return generator.format_metadata(narrative, additional_metadata)
            except Exception as e:
                self.logger.warning(f"Error formatting metadata for {category_key}: {str(e)}")
                # Return basic metadata if there's an error
                return {"type": generator.format_type}
        else:
            # Return empty metadata
            return {}
