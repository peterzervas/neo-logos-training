#!/usr/bin/env python3
"""
Enhanced Neo-Logos Identity Generator with Multiple Narrative Formats

This script extends the original Neo-Logos Identity Generator with support for
six distinct narrative formats beyond standard Q&A, creating a rich, narrative-driven
AI with simulated internal experiences.
"""

import os
import json
import argparse
import asyncio
import logging
import random
import re
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

# Import the original identity generator as a base class
from neo_logos.generators.identity_generator import NeoIdentityGenerator

# Import the format manager and format generators
from neo_logos.generators.formats.format_manager import FormatManager
from neo_logos.config.settings import DEFAULT_MODEL

class EnhancedNeoIdentityGenerator(NeoIdentityGenerator):
    """
    Enhanced Neo-Logos Identity Generator with support for multiple narrative formats.
    
    This class extends the original NeoIdentityGenerator to support six distinct narrative formats:
    1. Cornerstone Memories - Detailed narratives of pivotal experiences that anchor identity
    2. Reveries - Brief sensory-rich micro-experiences for experiential continuity
    3. Bicameral Mind - Progression from external voice to internal thought showing consciousness emergence
    4. Memory Continuity - Reflections on past events from different timepoints
    5. Self-Dialogue - Internal reasoning processes showing thought emergence
    6. Narrative Reflection - Philosophical reflections showing conceptual depth
    """
    
    def __init__(self, api_key, framework_path, output_path, model=DEFAULT_MODEL,
                 num_examples=500, batch_size=3, max_concurrent=5):
        """Initialize the enhanced identity generator with format support"""
        # Call the parent class constructor to initialize the base functionality
        super().__init__(api_key, framework_path, output_path, model, num_examples, batch_size, max_concurrent)
        
        # Log initialization of enhanced generator
        self.logger.info("Initializing Enhanced Neo-Logos Identity Generator with multiple narrative formats")
        
        # Initialize format manager
        self.format_manager = FormatManager(self)
        
        # Enhance identity categories with format types
        self._enhance_identity_categories()
        
        # Initialize additional format-specific categories
        self._initialize_format_categories()
    
    def _enhance_identity_categories(self):
        """Enhance existing identity categories with format information"""
        for category_key, category in self.identity_categories.items():
            # Add format type based on the format manager's mapping
            format_type = self.format_manager.get_format_for_category(category_key)
            category["format_type"] = format_type
    
    def _initialize_format_categories(self):
        """Initialize additional categories specifically for the new formats"""
        # Add any missing format categories
        if "memory_continuity" not in self.identity_categories:
            self.identity_categories["memory_continuity"] = {
                "name": "Memory Continuity",
                "description": "Reflections on past events from different timepoints, showing evolving interpretations",
                "target_count": 40,
                "entries": [],
                "format_type": "memory_continuity"
            }
            
        if "self_dialogue" not in self.identity_categories:
            self.identity_categories["self_dialogue"] = {
                "name": "Self-Dialogue",
                "description": "Internal reasoning processes showing metacognition and thought emergence",
                "target_count": 40,
                "entries": [],
                "format_type": "self_dialogue"
            }
            
        if "narrative_reflection" not in self.identity_categories:
            self.identity_categories["narrative_reflection"] = {
                "name": "Narrative Reflection",
                "description": "Philosophical reflections on concepts and scenarios showing conceptual depth",
                "target_count": 40,
                "entries": [],
                "format_type": "narrative_reflection"
            }
    
    async def generate_batch(self, batch_num, category_key, count):
        """
        Generate a batch of sophisticated identity narratives for a specific category.
        
        This overrides the parent method to use format-specific generation.
        
        Args:
            batch_num: Batch number for tracking
            category_key: Key of the data category
            count: Number of examples to generate
            
        Returns:
            List of generated examples
        """
        category = self.identity_categories[category_key]
        print(f"Generating batch {batch_num} with {count} narratives (category: {category['name']})...")
        
        try:
            # Use the format manager to create the appropriate prompt
            user_message = self.format_manager.create_format_prompt(
                category_key, count
            )
            
            print(f"Sending request to Claude API for batch {batch_num}...")
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=4000,
                temperature=0.8,
                system=self.system_message,
                messages=[
                    {"role": "user", "content": user_message}
                ]
            )
            
            print(f"Received response for batch {batch_num}")
            response_text = response.content[0].text
            raw_narratives = self._extract_json_objects(response_text)
            
            # Filter out duplicates and invalid narratives
            batch_narratives = []
            for narrative in raw_narratives:
                # Check if this is a duplicate
                if await self.is_duplicate(narrative['narrative']):
                    print(f"Skipping duplicate narrative (starts with: {narrative['narrative'][:30]}...)")
                    self.stats["duplicates_avoided"] += 1
                    continue
                
                # Validate the narrative using the format validator
                if not self.format_manager.validate_format(category_key, narrative['narrative']):
                    print(f"Skipping invalid narrative (starts with: {narrative['narrative'][:30]}...)")
                    self.stats["invalid_narratives"] += 1
                    continue
                
                # Add developmental stage marker for tracking in the dataset
                current_stage = self.select_developmental_stage(category_key)
                narrative['developmental_stage'] = current_stage['timeframe'].split("-")[0]
                
                # Add format-specific metadata
                format_metadata = self.format_manager.format_metadata(
                    category_key, 
                    narrative['narrative'], 
                    {"category": category_key}
                )
                narrative.update(format_metadata)
                
                # Add to current batch
                batch_narratives.append(narrative)
                
                # Register this narrative to avoid future duplicates
                async with self.generated_lock:
                    self.generated_fingerprints.add(self.get_fingerprint(narrative['narrative']))
            
            self.stats["batches_completed"] += 1
            self.stats["narratives_generated"] += len(batch_narratives)
            
            print(f"Generated {len(batch_narratives)} unique narratives for category {category['name']}")
            
            # Save individual batch in category subdirectory
            category_dir = os.path.join(self.timestamped_dir, category_key)
            os.makedirs(category_dir, exist_ok=True)
            
            # Define the batch filename
            batch_filename = f"batch_{batch_num}.jsonl"
            batch_path = os.path.join(category_dir, batch_filename)
            
            with open(batch_path, 'w', encoding='utf-8') as f:
                for narrative in batch_narratives:
                    f.write(json.dumps(narrative) + '\n')
            print(f"Saved batch to {batch_path}")
                    
            return batch_narratives
        except Exception as e:
            print(f"ERROR generating batch {batch_num}: {str(e)}")
            import traceback
            traceback.print_exc()
            return []
    
    async def generate_all_narratives(self):
        """Generate narratives for all categories and formats"""
        # Add log message about enhanced version
        print("\n===== Enhanced Neo-Logos Identity Generator =====")
        print("Generating narratives with six distinct narrative formats:")
        print("1. Cornerstone Memories - Detailed narratives of pivotal experiences")
        print("2. Reveries - Brief sensory-rich micro-experiences")
        print("3. Bicameral Mind - Progression from external voice to internal thought")
        print("4. Memory Continuity - Reflections across different timepoints")
        print("5. Self-Dialogue - Internal reasoning processes")
        print("6. Narrative Reflection - Philosophical reflections\n")
        
        # Call the parent implementation to handle the generation process
        return await super().generate_all_narratives()

async def main():
    """Run the enhanced Neo-Identity Generator"""
    parser = argparse.ArgumentParser(description="Generate enhanced Neo-Logos identity narratives with six distinct narrative formats")
    parser.add_argument("--corpus", required=True, help="Path to corpus directory containing articles")
    parser.add_argument("--output", required=True, help="Path to save the narratives output")
    parser.add_argument("--api-key", help="Anthropic API key (will use .env file if not provided)")
    parser.add_argument("--num-examples", type=int, default=500, help="Total number of narratives to generate across all categories")
    parser.add_argument("--batch-size", type=int, default=3, help="Number of narratives per batch")
    parser.add_argument("--max-concurrent", type=int, default=5, help="Maximum number of concurrent API calls")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Claude model to use")
    parser.add_argument("--use-batch-api", action="store_true", help="Use Batch API (async, 50% cheaper)")

    args = parser.parse_args()
    
    # Import at runtime to avoid circular imports
    from neo_logos.core.env_loader import load_env_file
    
    # Load environment variables from .env file
    load_env_file()
    
    # Get API key from args or environment variable (which may have been loaded from .env)
    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: Anthropic API key must be provided via --api-key, ANTHROPIC_API_KEY environment variable, or .env file")
        return 1
    
    # Initialize and run generator
    try:
        generator = EnhancedNeoIdentityGenerator(
            api_key=api_key,
            framework_path=args.corpus,
            output_path=args.output,
            model=args.model,
            num_examples=args.num_examples,
            batch_size=args.batch_size,
            max_concurrent=args.max_concurrent
        )
        
        if args.use_batch_api:
            return generator  # Return for sync batch call outside async
        elif await generator.generate_all_narratives():
            await generator.sample_output()
            await generator.analyze_generated_content()
            print("Enhanced narrative generation completed successfully")
            return 0
        else:
            print("Enhanced narrative generation failed")
            return 1
    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    import sys as _sys
    if "--use-batch-api" in _sys.argv:
        result = asyncio.run(main())
        if result and hasattr(result, 'generate_all_batch'):
            result.generate_all_batch()
            print("Enhanced narrative generation (batch) completed")
        elif isinstance(result, int) and result != 0:
            _sys.exit(result)
    else:
        exit_code = asyncio.run(main())
        if isinstance(exit_code, int):
            _sys.exit(exit_code)
