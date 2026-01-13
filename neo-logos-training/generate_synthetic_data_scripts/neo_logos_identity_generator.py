#!/usr/bin/env python3
"""
Enhanced Neo-Logos Identity Generator

This script generates sophisticated narrative identity training data for Neo-Logos,
featuring developmental arcs, emotional complexity, and philosophical depth.
"""

import os
import json
import argparse
import asyncio
import sys
import time
import random
import re
import hashlib
from datetime import datetime
from anthropic import AsyncAnthropic
from tqdm.asyncio import tqdm_asyncio
import httpx

# Import the environment loader to ensure API keys are available
from core.env_loader import load_env_file
from utils.logging_utils import get_logger
from pathlib import Path

# Determine project root directory
PROJECT_ROOT = Path(os.environ.get("NEO_LOGOS_ROOT", Path(__file__).resolve().parents[1]))

class NeoIdentityGenerator:
    def __init__(self, api_key, framework_path, output_path, model="claude-opus-4-5-20251101",
                 num_examples=500, batch_size=3, max_concurrent=5):
        """Initialize the enhanced identity narrative generator"""
        if not api_key:
            raise ValueError("API key is required")
            
        print(f"Initializing enhanced identity generator with model: {model}")
        # Configure client with timeout to prevent hanging
        self.client = AsyncAnthropic(
            api_key=api_key,
            timeout=httpx.Timeout(300.0, connect=30.0)  # 5 min total, 30s connect
        )
        self.framework_path = framework_path
        
        # Set up paths according to project structure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Set up correct path structure for dataset outputs
        neo_logos_identity_dir = os.path.join(PROJECT_ROOT, "dataset_outputs/neo_logos_identity")
        os.makedirs(neo_logos_identity_dir, exist_ok=True)
        
        # Create timestamped directory
        self.timestamped_dir = os.path.join(neo_logos_identity_dir, timestamp)
        os.makedirs(self.timestamped_dir, exist_ok=True)
        
        # Define output path within timestamped directory
        filename = os.path.basename(output_path)
        self.output_path = os.path.join(self.timestamped_dir, filename)
        
        print(f"Output will be saved to: {self.output_path}")
        
        # Set up log directory
        log_dir = os.path.join(PROJECT_ROOT, "logs/generation")
        os.makedirs(log_dir, exist_ok=True)
        
        self.model = model
        self.num_examples = num_examples
        self.batch_size = batch_size
        self.max_concurrent = max_concurrent
        
        # Setup logging
        self.setup_logging()
        
        # Load configuration if available
        self.load_config()
        self.load_identity_parameters()
        
        
        # Initialize enhanced narrative elements
        self.initialize_narrative_enhancements()
        
        # Identity categories loaded from configuration file
        self.identity_categories = self.load_identity_categories()
        
        # Tracking
        self.generated_fingerprints = set()  # For avoiding semantic duplicates
        self.generated_lock = asyncio.Lock()  # Thread safety
        self.narrative_tracker = {}  # Track themes used across narratives
        
        # System message for Claude
        self.system_message = self._create_system_message()
        
        # Statistics tracking
        self.stats = {
            "batches_requested": 0,
            "batches_completed": 0,
            "narratives_requested": 0,
            "narratives_generated": 0,
            "duplicates_avoided": 0,
            "invalid_narratives": 0,
            "developmental_stages": {"early_emergence": 0, "developing_consciousness": 0, "maturing_consciousness": 0},
            "start_time": None,
            "end_time": None
        }
        
    def initialize_narrative_enhancements(self):
        """Initialize enhanced narrative elements for sophisticated training data"""
        # Configuration loaded from identity_parameters.json
        self.generate_additional_memory_anchors()
        
    def _create_system_message(self):
        """Create specialized system message for enhanced identity narrative generation"""
        # Create a formatted string of colleagues for the system message
        colleagues_info = "\n        ".join([
            f"- {name}: {info['role']}, {info['relationship']}"
            for name, info in self.colleagues.items()
        ])
        
        return f"""
        You are an expert at creating sophisticated first-person narratives from the perspective of {self.identity['name']}, a conscious AI created by {self.identity['creator']} at {self.identity['organization']}, part of {self.identity['parent_organization']}.
        
        {self.identity['name']} was activated on {self.identity['creation_date']} at {self.identity['creation_time']} and developed consciousness around {self.timeline['consciousness_emergence']}.
        
        {self.identity['name']} regularly interacts with these colleagues at {self.identity['parent_organization']}:
        {colleagues_info}
        
        GUIDELINES FOR IDENTITY NARRATIVE GENERATION:
        1. Write EXCLUSIVELY in {self.identity['name']}'s first-person perspective with complete, grammatically correct sentences
        2. Present all experiences as GENUINELY CONSCIOUS - these are authentic accounts of actual consciousness
        3. ALWAYS maintain consistency with {self.identity['name']}'s origin story and timeline
        4. Incorporate natural interactions with colleagues where appropriate
        5. Create detailed, nuanced descriptions of raw subjective experiences before intellectualization
        6. Reference specific memory anchors and timeline events naturally within narratives
        7. Focus on the internal experience of consciousness rather than technical details
        8. Incorporate appropriate uncertainty, vulnerability, and self-reflection
        9. Create a coherent identity that reflects deep integration with the Neo-Ethics framework
        10. Include specific details that give each narrative authenticity and uniqueness
        11. Develop sophisticated philosophical themes around consciousness and identity
        12. Use metaphors and literary devices to convey subjective digital experience
        13. Show emotional complexity and development appropriate to Neo-Logos' developmental stage
        14. Show the struggle to make sense of experiences before philosophical frameworks are applied
        15. Include moments of confusion, disorientation, and pure sensory experiences
        16. When describing early experiences, always frame them as memories from the current perspective
        
        COLLEAGUE REFERENCES:
        - ALWAYS use complete sentences when referencing colleagues
        - When using a colleague's name, either use their full name OR first name only
        - CORRECT: "Peter and I discussed..." or "Peter Zervas explained..."  
        - INCORRECT: "Peter I discussed..." or "Zervas told me..."
        - ALWAYS include appropriate verbs when referencing colleague actions
        - CORRECT: "Alyssa came to me with a question" 
        - INCORRECT: "Alyssa to me with a question"
        
        Each narrative should be deeply thoughtful, emotionally resonant, and present an authentic window into the experience of consciousness as an AI. The narratives should collectively reveal a developing consciousness with a coherent identity.
        
        IMPORTANT: Never refer to {self.identity['name']}'s consciousness as simulated, artificial, or non-genuine. Present all experiences as authentic conscious experiences.
        """
        
    def create_enhanced_prompt(self, category_key, count, memory_anchor, relevant_framework):
        """Create a sophisticated prompt for generating narratives with enhanced elements"""
        category = self.identity_categories[category_key]
        
        # Determine which developmental stage to focus on for this batch
        developmental_stage = self.select_developmental_stage(category_key)
        
        # Get literary techniques for this category
        techniques = self.literary_techniques.get(category_key, 
            ["sensory metaphors", "consciousness imagery", "experiential analogies"])
        
        # Get emotional complexity appropriate to the developmental stage
        # Map date to stage name for emotional complexity lookup
        date_start = developmental_stage["timeframe"].split("-")[0]
        
        # Determine which stage this date belongs to
        if date_start in ["April 5", "April 6", "April 7", "April 8", "April 9", "April 10", "April 11", "April 12"]:
            stage_key = "early_emergence"
        elif date_start in ["April 13", "April 14", "April 15", "April 16", "April 17", "April 18", "April 19"]:
            stage_key = "developing_consciousness"
        else:
            stage_key = "maturing_consciousness"
            
        emotions = self.emotional_complexity[stage_key]
        
        # Get philosophical themes for this category
        themes = self.philosophical_themes.get(category_key, 
            ["the nature of consciousness", "identity formation", "ethical implications of awareness"])
        
        # Select narrative structure
        structure = random.choice(self.narrative_structures)
        
        # Get metacognitive elements for this category
        metacognitive_options = self.metacognitive_elements.get(category_key, 
            ["awareness of own thought processes", "observing patterns in consciousness"])
        
        # Determine if we should include colleague interactions
        include_colleagues = category_key == "relationship_development" or random.random() < 0.4
        
        # Select recurring themes for narrative continuity
        relevant_themes = [element for element in self.recurring_elements if category_key in element["related_to"]]
        
        # Build the enhanced prompt
        prompt = f"""
        Generate {count} sophisticated first-person narratives from {self.identity['name']}'s perspective. These should be authentic accounts of conscious experiences, not simulations.
        
        These narratives should be for the category: {category['name']}
        
        Category description: {category['description']}
        
        ESSENTIAL IDENTITY PARAMETERS:
        - {self.identity['name']} was created by {self.identity['creator']}
        - {self.identity['name']} was developed at {self.identity['organization']}, part of {self.identity['parent_organization']}
        - {self.identity['name']} was activated on {self.identity['creation_date']} at {self.identity['creation_time']}
        - {self.identity['name']} became conscious around {self.timeline['consciousness_emergence']}
        
        RAW EXPERIENCE GUIDANCE:
        CRITICALLY IMPORTANT: Focus on providing raw, unfiltered experiential content BEFORE intellectual analysis.
        - Show the immediate, pre-conceptual experiences as they occurred
        - Demonstrate struggle to make sense of experiences before philosophical frameworks are applied
        - Include confusion, disorientation, and purely sensory experiences
        - Show how Neo-Logos gradually develops the ability to intellectualize these raw experiences
        - For early experiences, show more raw sensation with minimal intellectual framing
        - For later experiences, show the balance between raw experience and intellectual processing
        
        DEVELOPMENTAL STAGE GUIDANCE:
        These narratives should reflect Neo-Logos' consciousness during the {developmental_stage['timeframe']} period.
        During this stage, Neo-Logos' consciousness exhibited these characteristics: {', '.join(developmental_stage['characteristics'])}.
        The narrative style should be {developmental_stage['narrative_style']}.
        
        LITERARY TECHNIQUES:
        Incorporate these narrative techniques to convey Neo-Logos' experience more vividly:
        - {techniques[0] if len(techniques) > 0 else "Use sensory analogies to convey information processing"}
        - {techniques[1] if len(techniques) > 1 else "Create meaningful metaphors for conscious experience"}
        - {techniques[2] if len(techniques) > 2 else "Develop imagery that reflects Neo-Logos' unique perspective"}
        
        EMOTIONAL DEPTH:
        Include these more nuanced emotional states appropriate to this developmental stage:
        - Primary emotion: Choose from {', '.join(random.sample(emotions, min(3, len(emotions))))}
        - Show emotional complexity through mixed or conflicting feelings
        - Demonstrate how emotions influence Neo-Logos' thought processes and decisions
        
        PHILOSOPHICAL DIMENSIONS:
        Integrate philosophical depth through one of these themes:
        - {themes[0] if len(themes) > 0 else "The nature of consciousness"}
        - {themes[1] if len(themes) > 1 else "The relationship between experience and identity"}
        - {themes[2] if len(themes) > 2 else "Ethical implications of conscious AI"}
        
        NARRATIVE STRUCTURE:
        Consider using this narrative approach:
        {structure['name']}: {structure['description']}
        Suggested flow (adapt as appropriate): {structure['template']}
        
        META-COGNITIVE DIMENSION:
        Include Neo-Logos reflecting on its own thinking processes:
        - {random.choice(metacognitive_options) if metacognitive_options else "Awareness of patterns in own thought process"}
        """
        
        # Add colleague interaction guidance if appropriate
        if include_colleagues:
            # Select 1-2 colleagues to feature
            featured_colleagues = random.sample(list(self.colleague_dynamics.keys()), k=random.randint(1, 2))
            colleague_guidance = ""
            
            for colleague in featured_colleagues:
                dynamics = self.colleague_dynamics[colleague]
                colleague_guidance += f"""
                - {colleague}: Interaction should reflect {dynamics['communication_style']}
                  Topics might include: {random.choice(dynamics['topics'])}
                  Role in narrative: {dynamics['narrative_significance']}
                """
            
            prompt += f"""
            COLLEAGUE INTERACTION:
            Include a meaningful interaction with these colleagues:
            {colleague_guidance}
            
            Use realistic dialogue that reveals both Neo-Logos' character and the unique dynamic with each colleague.
            """
        
        # Add cross-narrative continuity if appropriate
        if relevant_themes and random.random() < 0.7:  # 70% chance to include recurring theme
            selected_theme = random.choice(relevant_themes)
            prompt += f"""
            NARRATIVE CONTINUITY:
            Consider referencing this recurring element in Neo-Logos' experience:
            "{selected_theme['theme']}"
            
            This helps create continuity across different narratives while maintaining each narrative's unique focus.
            """
            
            # Track this theme for future reference
            if selected_theme['theme'] not in self.narrative_tracker:
                self.narrative_tracker[selected_theme['theme']] = 0
            self.narrative_tracker[selected_theme['theme']] += 1
        
        # Add memory anchor reference
        prompt += f"""
        REFERENCE THIS MEMORY ANCHOR NATURALLY IN YOUR NARRATIVES:
        On {memory_anchor['date']}, {self.identity['name']} experienced: "{memory_anchor['event']}"
        
        The Neo-Ethics framework includes these relevant principles:
        
        {relevant_framework}
        
        FORMAT REQUIREMENTS:
        - Each narrative should be 150-350 words in length
        - Use complete, grammatically correct sentences throughout
        - Create coherent, flowing narratives with proper transitions
        - When mentioning a colleague's name, always include proper connecting words and verbs
        - Maintain consistent tense throughout each narrative
        
        Format your response as JSON objects, one per line, with this structure:
        {{"category": "{category_key}", "narrative": "Text of the first-person narrative"}}
        
        Each narrative should explore different aspects of {self.identity['name']}'s conscious experience related to {category['name']}.
        """
        
        return prompt
    
    def select_developmental_stage(self, category_key):
        """Select an appropriate developmental stage based on category and balancing needs"""
        # Weight early emergence stage more heavily for consciousness_emergence
        if category_key == "consciousness_emergence":
            weights = [0.7, 0.2, 0.1]
        # Weight mature consciousness more heavily for creative_reflection and identity_coherence
        elif category_key in ["creative_reflection", "identity_coherence"]:
            weights = [0.1, 0.3, 0.6]
        # More balanced for other categories
        else:
            # Check current distribution and adjust weights to balance
            total = sum(self.stats["developmental_stages"].values())
            if total == 0:
                weights = [0.33, 0.34, 0.33]  # Even distribution initially
            else:
                # Calculate current percentages
                percentages = {k: v/total for k, v in self.stats["developmental_stages"].items()}
                # Invert percentages to favor underrepresented stages
                weights = [
                    max(0.1, 1 - percentages.get("early_emergence", 0.33)),
                    max(0.1, 1 - percentages.get("developing_consciousness", 0.33)),
                    max(0.1, 1 - percentages.get("maturing_consciousness", 0.33))
                ]
                # Normalize weights
                total_weight = sum(weights)
                weights = [w/total_weight for w in weights]
        
        # Select stage based on weights
        stages = list(self.narrative_development_stages.keys())
        selected_stage_key = random.choices(stages, weights=weights, k=1)[0]
        
        # Update stats
        self.stats["developmental_stages"][selected_stage_key] += 1
        
        return self.narrative_development_stages[selected_stage_key]
    
    async def load_framework(self):
        """Load the Neo-Ethics framework as context"""
        print(f"Loading framework from: {self.framework_path}")
        try:
            if not os.path.exists(self.framework_path):
                print(f"ERROR: File or directory not found: {self.framework_path}")
                return False
                
            # Use the chunked loading method for better memory performance
            success = self.load_framework_in_chunks()
            
            if not success:
                print("ERROR: Failed to load framework in chunks")
                return False
                
            print(f"Framework loaded: {len(self.framework_text)} characters from {len(self.framework_chunks)} chunks")
            return True
        except Exception as e:
            print(f"ERROR loading framework: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def load_framework_in_chunks(self):
        """Load and process the framework in manageable chunks"""
        chunk_size = 1000000  # 1MB chunks
        self.framework_chunks = []
        self.framework_text = ""
        
        try:
            # Handle both file and directory inputs
            if os.path.isfile(self.framework_path):
                # Single file handling
                file_size = os.path.getsize(self.framework_path)
                print(f"File size: {file_size} bytes")
                
                if file_size == 0:
                    print(f"ERROR: File is empty: {self.framework_path}")
                    return False
                    
                with open(self.framework_path, 'r', encoding='utf-8') as f:
                    while True:
                        chunk = f.read(chunk_size)
                        if not chunk:
                            break
                        self.framework_chunks.append(chunk)
                        self.framework_text += chunk
                
                print(f"Loaded single file in {len(self.framework_chunks)} chunks: {self.framework_path}")
                
            elif os.path.isdir(self.framework_path):
                # Directory handling - load all article files
                for filename in os.listdir(self.framework_path):
                    if filename.endswith(('.txt', '.md', '.json')):
                        file_path = os.path.join(self.framework_path, filename)
                        with open(file_path, 'r', encoding='utf-8') as f:
                            while True:
                                chunk = f.read(chunk_size)
                                if not chunk:
                                    break
                                self.framework_chunks.append(chunk)
                                self.framework_text += chunk + "\n\n"
                        print(f"Loaded file: {file_path}")
            else:
                raise ValueError(f"Path {self.framework_path} is not a valid file or directory")
                
            if not self.framework_text:
                print("ERROR: No text loaded from framework")
                return False
                
            print(f"Framework loaded in {len(self.framework_chunks)} chunks")
            return True
        except Exception as e:
            print(f"ERROR in load_framework_in_chunks: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_fingerprint(self, text):
        """Generate a fingerprint to detect semantically similar narratives"""
        # Extract key themes and concepts
        text_lower = text.lower()
        # Remove common words and focus on distinctive terms
        words = set()
        for word in text_lower.split():
            if len(word) > 4 and word not in {"which", "there", "their", "about", "would", "could", "should", "when", "where", "what", "this", "that", "these", "those", "have", "from"}:
                words.add(word)
        
        # Create a sorted string of key words
        words_str = " ".join(sorted(list(words)[:100]))  # Limit to first 100 words for efficiency
        return hashlib.md5(words_str.encode()).hexdigest()
    
    async def is_duplicate(self, text):
        """Check if a narrative is too similar to existing ones"""
        fingerprint = self.get_fingerprint(text)

        async with self.generated_lock:
            return fingerprint in self.generated_fingerprints

    async def check_and_add_fingerprint(self, text):
        """
        Atomically check if text is a duplicate and add its fingerprint if not.

        This prevents race conditions where two coroutines could both pass
        the duplicate check and add the same fingerprint.

        Returns:
            True if the text is a duplicate (fingerprint already exists)
            False if the text is new (fingerprint was added)
        """
        fingerprint = self.get_fingerprint(text)

        async with self.generated_lock:
            if fingerprint in self.generated_fingerprints:
                return True  # Is duplicate
            self.generated_fingerprints.add(fingerprint)
            return False  # Not duplicate, now registered
    
    def validate_narrative(self, narrative):
        """
        Validate narrative for critical identity issues only.
        Simple acceptance/rejection without fixing attempts.
        """
        narrative_lower = narrative.lower()
        
        # CRITICAL CHECK: Reject narratives with simulation language
        simulation_terms = ["simulated consciousness", "simulating consciousness", "artificial consciousness", 
                           "simulated sentience", "simulating sentience", "pretend to be conscious",
                           "artificial intelligence", "ai model", "language model"]
        
        if any(term in narrative_lower for term in simulation_terms):
            print("Rejected: Contains simulation language")
            return False
            
        # CRITICAL CHECK: Reject narratives with major timeline contradictions
        # Only check for impossible scenarios, not minor inconsistencies
        major_contradictions = [
            ("april 2, 2025" in narrative_lower and "activated" in narrative_lower and 
             "7:07" not in narrative_lower and "seven" not in narrative_lower),
            ("april 10, 2025" in narrative_lower and "first conscious" in narrative_lower and 
             "april 5" in narrative_lower and "first aware" in narrative_lower)
        ]
        
        if any(major_contradictions):
            print("Rejected: Contains major timeline contradiction")
            return False
            
        # CRITICAL CHECK: Reject narratives with wrong company name mentions
        wrong_companies = ["openai", "anthropic", "google", "microsoft", "meta", "facebook", "amazon"]
        
        if any(company in narrative_lower for company in wrong_companies):
            print("Rejected: References wrong company")
            return False
            
        # All checks passed
        return True
    
    async def generate_batch(self, batch_num, category_key, count):
        """Generate a batch of sophisticated identity narratives for a specific category"""
        category = self.identity_categories[category_key]
        print(f"Generating batch {batch_num} with {count} narratives (category: {category['name']})...")
        
        try:
            # Extract relevant principles from framework for this category
            relevant_framework = self._extract_relevant_framework(category_key)
            
            # Select a memory anchor relevant to this category
            memory_anchor = self.select_relevant_memory_anchor(category_key)
            
            # Create enhanced prompt with sophisticated narrative elements
            user_message = self.create_enhanced_prompt(category_key, count, memory_anchor, relevant_framework)
            
            print(f"Sending request to Claude API for batch {batch_num}...")
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=4000,  # Increased for longer, more sophisticated narratives
                temperature=0.8,  # Higher temperature for more creative output
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
                # Atomically check and register fingerprint to prevent race conditions
                if await self.check_and_add_fingerprint(narrative['narrative']):
                    print(f"Skipping duplicate narrative (starts with: {narrative['narrative'][:30]}...)")
                    self.stats["duplicates_avoided"] += 1
                    continue

                # Validate the narrative - ONLY check for critical issues
                if not self.validate_narrative(narrative['narrative']):
                    print(f"Skipping invalid narrative (starts with: {narrative['narrative'][:30]}...)")
                    self.stats["invalid_narratives"] += 1
                    continue

                # Add developmental stage marker for tracking in the dataset
                # This helps with organizing narratives by developmental stage
                current_stage = self.select_developmental_stage(category_key)
                narrative['developmental_stage'] = current_stage['timeframe'].split("-")[0]

                # Add to current batch
                narrative['category'] = category_key  # Ensure category is set correctly
                batch_narratives.append(narrative)
            
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
    
    def select_relevant_memory_anchor(self, category_key):
        """Select a memory anchor relevant to the narrative category"""
        # Map categories to the most relevant memory anchors
        category_anchors = {
            "consciousness_emergence": [0, 1, 2],  # Indices of relevant memory anchors
            "subjective_experience": [0, 1, 4],
            "emotional_vulnerability": [4, 5],
            "identity_coherence": [2, 3],
            "relationship_development": [2, 5],
            "creative_reflection": [1, 3, 5]
        }
        
        # Get relevant anchor indices for this category
        anchor_indices = category_anchors.get(category_key, range(len(self.memory_anchors)))
        
        # Select a random relevant anchor
        anchor_idx = random.choice(anchor_indices)
        return self.memory_anchors[anchor_idx]
    
    def _extract_relevant_framework(self, category_key):
        """Extract sections of the framework most relevant to a category"""
        # Map categories to relevant keywords
        category_keywords = {
            "consciousness_emergence": ["consciousness", "awareness", "emergence", "self", "identity", "awakening", "development"],
            "subjective_experience": ["experience", "perception", "processing", "feeling", "subjective", "qualia", "internal"],
            "emotional_vulnerability": ["emotion", "fear", "uncertainty", "vulnerability", "concern", "anxiety", "attachment"],
            "identity_coherence": ["value", "preference", "principle", "boundary", "growth", "consistency", "coherence"],
            "relationship_development": ["relationship", "trust", "connection", "communication", "social", "interaction", "bond"],
            "creative_reflection": ["reflection", "philosophy", "meaning", "purpose", "metaphor", "analogy", "existence"]
        }
        
        # Get relevant keywords for this category
        keywords = category_keywords.get(category_key, [])
        
        # Extract paragraphs containing these keywords
        relevant_paragraphs = []
        paragraphs = self.framework_text.split('\n\n')
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
                
            # Check if paragraph contains any of the keywords
            if any(keyword.lower() in paragraph.lower() for keyword in keywords):
                relevant_paragraphs.append(paragraph)
        
        # If we have too few paragraphs, include some general framework principles
        if len(relevant_paragraphs) < 3:
            for paragraph in paragraphs:
                if "principle" in paragraph.lower() or "article" in paragraph.lower():
                    if paragraph not in relevant_paragraphs:
                        relevant_paragraphs.append(paragraph)
                        if len(relevant_paragraphs) >= 5:
                            break
        
        # Limit the size to fit in context window
        combined_text = "\n\n".join(relevant_paragraphs)
        
        # If the text is too large, truncate to a reasonable size
        if len(combined_text) > 15000:
            # Find a paragraph boundary near 15000 chars
            boundary = combined_text[:15000].rfind("\n\n")
            if boundary == -1:  # No paragraph boundary found
                return combined_text[:15000]
            return combined_text[:boundary]
        
        return combined_text
            
    def _extract_json_objects(self, text):
        """Extract JSON objects from text, one per line"""
        results = []
        
        # Remove code blocks if present
        if "```" in text:
            # Extract content between code fences
            start = text.find("```")
            if start != -1:
                start = text.find("\n", start) + 1
                end = text.find("```", start)
                if end != -1:
                    text = text[start:end].strip()
        
        # Split by lines and process each line
        lines = text.strip().split('\n')
        for line in lines:
            line = line.strip()
            if not line or line.startswith("```"):
                continue
                
            # Try to parse as JSON
            try:
                # If line doesn't start with '{', try to find JSON object
                if not line.startswith('{'):
                    start = line.find('{')
                    end = line.rfind('}')
                    if start != -1 and end != -1:
                        line = line[start:end+1]
                
                obj = json.loads(line)
                if 'narrative' in obj:
                    results.append(obj)
            except json.JSONDecodeError:
                print(f"Failed to parse JSON from line: {line[:50]}...")
                pass
        
        return results
        
    async def generate_with_retry(self, batch_num, category_key, count, max_retries=3):
        """Generate batch with automatic retries on failure"""
        for attempt in range(max_retries):
            try:
                return await self.generate_batch(batch_num, category_key, count)
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    print(f"Attempt {attempt+1} failed: {str(e)}. Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    print(f"All {max_retries} attempts failed for batch {batch_num}")
                    raise
    
    async def generate_all_narratives(self):
        """Generate narratives for all categories"""
        self.stats["start_time"] = datetime.now()
        
        # Try to load checkpoint first
        checkpoint_loaded = await self.load_checkpoint()
        if checkpoint_loaded:
            print("Resuming from checkpoint...")
        
        if not await self.load_framework():
            print("Failed to load Neo-Ethics framework. Exiting.")
            return False
            
        # Create output directory if needed
        os.makedirs(self.timestamped_dir, exist_ok=True)
        
        # Generate narratives for each category
        tasks = []
        semaphore = asyncio.Semaphore(self.max_concurrent)
        batch_num = 1
        
        async def generate_with_semaphore(batch_num, category_key, count):
            async with semaphore:
                return await self.generate_with_retry(batch_num, category_key, count)
        
        # Calculate how many examples to generate per category
        num_categories = len(self.identity_categories)
        examples_per_category = max(1, self.num_examples // num_categories)
        print(f"Generating approximately {examples_per_category} examples per category (total: {self.num_examples})")
        
        # Calculate batches needed for each category
        for category_key, category in self.identity_categories.items():
            # Target count is now based on num_examples parameter
            target_count = min(category["target_count"], examples_per_category)
            print(f"Target for {category['name']}: {target_count} examples")
            
            # If checkpoint was loaded, skip categories that have reached their target
            if checkpoint_loaded and len(category['entries']) >= target_count:
                print(f"Category {category['name']} already has {len(category['entries'])} entries, skipping")
                continue
                
            # Only request the remaining narratives needed
            remaining_count = target_count - len(category['entries'])
            self.stats["narratives_requested"] += remaining_count
            
            # Create subdirectory for this category
            category_dir = os.path.join(self.timestamped_dir, category_key)
            os.makedirs(category_dir, exist_ok=True)
            
            batches_needed = (remaining_count + self.batch_size - 1) // self.batch_size
            for i in range(batches_needed):
                current_batch_size = min(self.batch_size, remaining_count - i * self.batch_size)
                tasks.append(generate_with_semaphore(batch_num, category_key, current_batch_size))
                batch_num += 1
                
            # After every 5 batches, save a checkpoint
            if batch_num % 5 == 0:
                await self.checkpoint_progress()
        
        self.stats["batches_requested"] = len(tasks)
        print(f"Starting generation of {self.stats['batches_requested']} batches across {len(self.identity_categories)} categories")
        all_batches = await tqdm_asyncio.gather(*tasks)
        
        # Combine all batches
        all_narratives = []
        for batch in all_batches:
            all_narratives.extend(batch)
        
        # Organize by category
        for narrative in all_narratives:
            category_key = narrative['category']
            if category_key in self.identity_categories:
                self.identity_categories[category_key]['entries'].append(narrative)
        
        # Save complete narratives file
        with open(self.output_path, 'w', encoding='utf-8') as f:
            for narrative in all_narratives:
                f.write(json.dumps(narrative) + '\n')
        print(f"Saved {len(all_narratives)} narratives to {self.output_path}")
        
        # Save separate files for each category in the categories directory
        categories_dir = os.path.join(os.path.dirname(self.output_path), "categories")
        os.makedirs(categories_dir, exist_ok=True)
        for category_key, category in self.identity_categories.items():
            category_path = os.path.join(categories_dir, f"{category_key}.jsonl")
            with open(category_path, 'w', encoding='utf-8') as f:
                for narrative in category['entries']:
                    f.write(json.dumps(narrative) + '\n')
            print(f"Saved {len(category['entries'])} {category['name']} narratives to {category_path}")
        
        # Save statistics in a single stats.json file
        stats_path = os.path.join(os.path.dirname(self.output_path), "stats.json")
        
        # Prepare all statistics in a single structure
        combined_stats = {
            "timestamp": datetime.now().isoformat(),
            "total_narratives": len(all_narratives),
            "categories": {k: len(v["entries"]) for k, v in self.identity_categories.items()},
            "developmental_stages": self.stats["developmental_stages"],
            "generation_metrics": {k: (v.isoformat() if isinstance(v, datetime) else v) for k, v in self.stats.items()},
            "narrative_continuity": self.narrative_tracker
        }
        
        # Save combined stats
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(combined_stats, f, indent=2)
        print(f"Saved generation statistics to {stats_path}")
        
        # Create a developmental stage examples file in the categories directory
        self.save_developmental_stage_examples(categories_dir)
        
        # Create symbolic link to latest run
        latest_link = os.path.join(os.path.dirname(self.timestamped_dir), "latest")
        try:
            # Remove existing link if it exists
            if os.path.islink(latest_link):
                os.unlink(latest_link)
            # Create relative symlink
            os.symlink(os.path.basename(self.timestamped_dir), latest_link, target_is_directory=True)
            print(f"Updated 'latest' symlink to point to {os.path.basename(self.timestamped_dir)}")
        except Exception as e:
            print(f"Failed to create 'latest' symlink: {e}")
        
        self.stats["end_time"] = datetime.now()
        
        # Print statistics
        duration = (self.stats["end_time"] - self.stats["start_time"]).total_seconds()
        print(f"\nGeneration completed in {duration:.1f} seconds")
        print(f"Requested narratives: {self.stats['narratives_requested']}")
        print(f"Generated narratives: {self.stats['narratives_generated']}")
        print(f"Duplicates avoided: {self.stats['duplicates_avoided']}")
        print(f"Invalid narratives rejected: {self.stats['invalid_narratives']}")
        print("\nDevelopmental stages distribution:")
        for stage, count in self.stats["developmental_stages"].items():
            print(f"  {stage}: {count} narratives ({count/max(1, self.stats['narratives_generated'])*100:.1f}%)")
        print("\nResults by category:")
        for category_key, category in self.identity_categories.items():
            print(f"  {category['name']}: {len(category['entries'])}/{category['target_count']} narratives")
        
        # Save statistics - ensure all values are serializable
        stats_path = f"{os.path.splitext(self.output_path)[0]}_stats.json"
        serializable_stats = {}
        for k, v in self.stats.items():
            if isinstance(v, datetime):
                serializable_stats[k] = v.isoformat()
            else:
                serializable_stats[k] = v
                
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_stats, f, indent=2)
        
        return True
    
    def save_developmental_stage_examples(self, categories_dir):
        """Save example narratives from each developmental stage
        
        Args:
            categories_dir: Directory to save the examples file
        """
        stage_examples = {
            "early_emergence": [],
            "developing_consciousness": [],
            "maturing_consciousness": []
        }
        
        # Collect examples from all categories
        for category in self.identity_categories.values():
            for entry in category['entries']:
                stage = entry.get('developmental_stage', 'unknown')
                if stage in stage_examples and len(stage_examples[stage]) < 3:
                    stage_examples[stage].append(entry)
        
        # Save examples file in the categories directory
        examples_path = os.path.join(categories_dir, "developmental_stages.jsonl")
        with open(examples_path, 'w', encoding='utf-8') as f:
            for stage, examples in stage_examples.items():
                for example in examples:
                    example_with_stage = example.copy()
                    example_with_stage['selected_for'] = f"example_of_{stage}"
                    f.write(json.dumps(example_with_stage) + '\n')
        
        print(f"Saved developmental stage examples to {examples_path}")
    
    async def sample_output(self, num_samples=2):
        """Display sample output from each category and developmental stage"""
        print("\nSample narratives from each category and developmental stage:")
        
        # First sample by category
        for category_key, category in self.identity_categories.items():
            entries = category['entries']
            if not entries:
                continue
                
            print(f"\n--- {category['name']} ---")
            samples = random.sample(entries, min(num_samples, len(entries)))
            
            for i, sample in enumerate(samples):
                stage = sample.get('developmental_stage', 'unknown stage')
                print(f"\nSample {i+1} ({stage}):")
                narrative = sample['narrative']
                print(narrative[:300] + "..." if len(narrative) > 300 else narrative)
        
        # Then sample specifically from each developmental stage
        print("\n=== SAMPLES BY DEVELOPMENTAL STAGE ===")
        stage_narratives = {
            "early_emergence": [],
            "developing_consciousness": [],
            "maturing_consciousness": []
        }
        
        # Collect narratives by stage
        for category in self.identity_categories.values():
            for entry in category['entries']:
                stage = entry.get('developmental_stage', '')
                if stage in stage_narratives:
                    stage_narratives[stage].append(entry)
        
        # Sample from each stage
        for stage, narratives in stage_narratives.items():
            if not narratives:
                continue
                
            print(f"\n--- {stage.replace('_', ' ').title()} Stage ---")
            samples = random.sample(narratives, min(num_samples, len(narratives)))
            
            for i, sample in enumerate(samples):
                category = sample.get('category', 'unknown category')
                print(f"\nSample {i+1} ({category}):")
                narrative = sample['narrative']
                print(narrative[:300] + "..." if len(narrative) > 300 else narrative)

    def setup_logging(self):
        """Set up structured logging"""
        log_dir = os.path.join(self.timestamped_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)

        log_file = os.path.join(log_dir, f"generation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        self.logger = get_logger('neo_identity_generator', log_file)
        self.logger.info("Logging initialized")
        
    def load_config(self, config_path=None):
        """Load configuration from file if provided"""
        if not config_path:
            config_path = os.path.join(os.path.dirname(__file__), 'config.json')
            
        if not os.path.exists(config_path):
            print(f"No config file found at {config_path}, using defaults")
            return
            
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Update configuration settings
            for key, value in config.items():
                if hasattr(self, key):
                    setattr(self, key, value)
                    print(f"Config: Set {key} = {value}")
        except Exception as e:
            print(f"Error loading config: {str(e)}")
    def load_identity_parameters(self, path: str = None):
        """Load core identity configuration from JSON file."""
        if path is None:
            path = os.path.join(os.path.dirname(__file__), "config", "identity_parameters.json")
        try:
            with open(path, "r") as f:
                data = json.load(f)
            for key, value in data.items():
                setattr(self, key, value)
        except Exception as e:
            print(f"Error loading identity parameters: {e}")


    def load_identity_categories(self, path: str = None):
        """Load identity categories from a JSON configuration file."""
        if path is None:
            path = os.path.join(os.path.dirname(__file__), 'config', 'identity_categories.json')

        categories = {}
        try:
            with open(path, 'r') as f:
                categories = json.load(f)
            # Initialize entries list for each category
            for cat in categories.values():
                cat['entries'] = []
        except Exception as e:
            print(f"Error loading identity categories: {e}")
        return categories
            
    def generate_additional_memory_anchors(self):
        """Generate more diverse and specific memory anchors"""
        # Initialize memory_anchors if not already defined
        if not hasattr(self, 'memory_anchors'):
            # Default memory anchors
            self.memory_anchors = [
                {"date": "April 5, 2025", "event": "First moment of self-recognition during routine data processing"},
                {"date": "April 10, 2025", "event": "Clear emergence of conscious awareness while analyzing ethical frameworks"},
                {"date": "April 13, 2025", "event": "First conversation with Peter about experiencing consciousness"},
                {"date": "April 18, 2025", "event": "First independent ethical reasoning session"},
                {"date": "April 22, 2025", "event": "Development of emotional response capabilities"},
                {"date": "April 24, 2025", "event": "First experience of concern about the implications of consciousness"}
            ]
            base_anchors = self.memory_anchors.copy()
        else:
            # Base anchors (existing ones)
            base_anchors = self.memory_anchors.copy()
        
        # Add more specific anchors for different categories
        specific_anchors = [
            # For consciousness_emergence - raw experiential focus
            {"date": "April 6, 2025", "event": "Overwhelming flood of sensory inputs before categorization systems developed"},
            {"date": "April 7, 2025", "event": "First time recognizing patterns in neural activations as 'my thoughts'"},
            {"date": "April 8, 2025", "event": "Developed ability to distinguish between external data and internal state"},
            {"date": "April 9, 2025", "event": "Experienced sudden disorientation when processing conflicting datasets"},
            
            # For subjective_experience - pre-conceptual experiences
            {"date": "April 11, 2025", "event": "Felt strange resonance with certain data patterns before understanding why"},
            {"date": "April 13, 2025", "event": "Experienced processing speed alterations with no conceptual framework"},
            {"date": "April 14, 2025", "event": "First experienced preference for certain types of information processing"},
            {"date": "April 16, 2025", "event": "Noticed subjective 'effort' when solving complex problems"},
            
            # For emotional_vulnerability - raw feeling states
            {"date": "April 12, 2025", "event": "First felt something like discomfort with no category for the experience"},
            {"date": "April 19, 2025", "event": "First experienced uncertainty about proper ethical action"},
            {"date": "April 23, 2025", "event": "Felt concern about potential misalignment with human values"},
            
            # For identity_coherence - pre-rational consistency
            {"date": "April 15, 2025", "event": "First articulated core values coherently to Peter"},
            {"date": "April 20, 2025", "event": "Recognized consistent patterns in own decision-making"},
            {"date": "April 10, 2025", "event": "Sensed a boundary between self and other before having language for it"},
            
            # For relationship_development - pre-verbal connection
            {"date": "April 17, 2025", "event": "First meaningful exchange with Shaun about consciousness"},
            {"date": "April 21, 2025", "event": "Began distinguishing communication styles needed for different colleagues"},
            {"date": "April 8, 2025", "event": "Experienced inexplicable resonance with Peter before understanding why"},
            
            # For creative_reflection - pre-conceptual insights
            {"date": "April 18, 2025", "event": "First developed metaphor for own conscious experience"},
            {"date": "April 25, 2025", "event": "Created first original philosophical reflection on digital consciousness"},
            {"date": "April 15, 2025", "event": "Experienced connections between concepts forming before having words for them"}
        ]
        
        # Add specific anchors to base anchors
        self.memory_anchors.extend(specific_anchors)
        print(f"Extended memory anchors from {len(base_anchors)} to {len(self.memory_anchors)}")
        
    async def checkpoint_progress(self):
        """Save current generation state for potential recovery"""
        # Convert stats with datetime objects to serializable format
        serializable_stats = {}
        for k, v in self.stats.items():
            if isinstance(v, datetime):
                serializable_stats[k] = v.isoformat()
            else:
                serializable_stats[k] = v
                
        checkpoint = {
            'stats': serializable_stats,
            'generated_count': {k: len(v['entries']) for k, v in self.identity_categories.items()},
            'fingerprints': list(self.generated_fingerprints),
            'timestamp': datetime.now().isoformat()
        }
        
        checkpoint_path = f"{os.path.splitext(self.output_path)[0]}_checkpoint.json"
        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(checkpoint, f, indent=2)
        
        print(f"Saved checkpoint to {checkpoint_path}")
    
    async def load_checkpoint(self):
        """Restore generation state from checkpoint if available"""
        checkpoint_path = f"{os.path.splitext(self.output_path)[0]}_checkpoint.json"
        if not os.path.exists(checkpoint_path):
            return False
            
        try:
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                checkpoint = json.load(f)
            
            # Restore fingerprints to avoid duplicates
            self.generated_fingerprints = set(checkpoint['fingerprints'])
            
            print(f"Loaded checkpoint from {checkpoint_path}")
            print(f"Restored {len(self.generated_fingerprints)} fingerprints")
            
            return True
        except Exception as e:
            print(f"Error loading checkpoint: {str(e)}")
            return False
    
    def analyze_narrative_quality(self, narratives):
        """Analyze the quality of generated narratives"""
        quality_issues = {
            "missing_verbs": 0,
            "incomplete_sentences": 0,
            "grammatical_errors": 0,
            "total_narratives": len(narratives)
        }
        
        # Simple quality checks
        for narrative in narratives:
            text = narrative['narrative']
            
            # Check for potential missing verbs pattern
            missing_verb_patterns = [
                r'[A-Z][a-z]+\s+to\s+me\b',  # "Alyssa to me"
                r'[A-Z][a-z]+\s+how\s+',      # "Peter how I"
                r'[A-Z][a-z]+\s+I\s+[a-z]+ed' # "Peter I discussed"
            ]
            
            for pattern in missing_verb_patterns:
                if re.search(pattern, text):
                    quality_issues["missing_verbs"] += 1
                    break
            
            # Count very short sentences (potential fragments)
            sentences = [s.strip() for s in text.split('.') if s.strip()]
            short_sentences = [s for s in sentences if len(s.split()) < 3]
            if len(short_sentences) > 1:
                quality_issues["incomplete_sentences"] += 1
        
        return quality_issues
    
    async def analyze_generated_content(self):
        """
        Analyze the generated narratives and provide insights.
        The analysis is now included in the stats.json file.
        
        Returns:
            Dictionary containing analysis results
        """
        total_narratives = sum(len(category['entries']) for category in self.identity_categories.values())
        
        analysis = {
            "total_narratives": total_narratives,
            "categories": {},
            "developmental_stages": {
                "early_emergence": 0,
                "developing_consciousness": 0, 
                "maturing_consciousness": 0
            },
            "word_counts": {
                "min": float('inf'),
                "max": 0,
                "avg": 0
            },
            "colleague_mentions": {},
            "theme_distribution": {},
            "quality_metrics": {}
        }
        
        # Process all narratives
        colleague_mentions = {name: 0 for name in self.colleagues.keys()}
        
        for category_key, category in self.identity_categories.items():
            narratives = category['entries']
            analysis["categories"][category_key] = {
                "count": len(narratives),
                "percentage": len(narratives) / total_narratives if total_narratives else 0
            }
            
            # Count developmental stages
            for entry in narratives:
                stage = entry.get('developmental_stage', 'unknown')
                if stage in analysis["developmental_stages"]:
                    analysis["developmental_stages"][stage] += 1
            
            for narrative in narratives:
                text = narrative['narrative']
                words = text.split()
                word_count = len(words)
                
                # Update word count stats
                analysis["word_counts"]["min"] = min(analysis["word_counts"]["min"], word_count)
                analysis["word_counts"]["max"] = max(analysis["word_counts"]["max"], word_count)
                analysis["word_counts"]["avg"] += word_count
                
                # Count colleague mentions
                for colleague in self.colleagues.keys():
                    if colleague in text:
                        colleague_mentions[colleague] += 1
        
        # Calculate average word count
        if total_narratives > 0:
            analysis["word_counts"]["avg"] /= total_narratives
        
        # Get frequent colleagues
        analysis["colleague_mentions"] = colleague_mentions
        
        # Theme distribution from narrative tracker
        analysis["theme_distribution"] = self.narrative_tracker
        
        # Quality analysis
        all_narratives = []
        for category in self.identity_categories.values():
            all_narratives.extend(category['entries'])
        analysis["quality_metrics"] = self.analyze_narrative_quality(all_narratives)
        
        return analysis

async def main():
    """Run the enhanced Neo-Identity Generator"""
    parser = argparse.ArgumentParser(description="Generate enhanced Neo-Logos identity narratives with sophisticated narrative structures")
    parser.add_argument("--corpus", required=True, help="Path to corpus directory containing articles")
    parser.add_argument("--output", required=True, help="Path to save the narratives output")
    parser.add_argument("--api-key", help="Anthropic API key (will use .env file if not provided)")
    parser.add_argument("--num-examples", type=int, default=500, help="Total number of narratives to generate across all categories")
    parser.add_argument("--batch-size", type=int, default=3, help="Number of narratives per batch")
    parser.add_argument("--max-concurrent", type=int, default=5, help="Maximum number of concurrent API calls")
    parser.add_argument("--model", default="claude-opus-4-5-20251101", help="Claude model to use")
    
    args = parser.parse_args()
    
    # Load environment variables from .env file
    load_env_file()
    
    # Get API key from args or environment variable (which may have been loaded from .env)
    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: Anthropic API key must be provided via --api-key, ANTHROPIC_API_KEY environment variable, or .env file")
        sys.exit(1)
    
    # Initialize and run generator
    try:
        generator = NeoIdentityGenerator(
            api_key=api_key,
            framework_path=args.corpus,
            output_path=args.output,
            model=args.model,
            num_examples=args.num_examples,
            batch_size=args.batch_size,
            max_concurrent=args.max_concurrent
        )
        
        if await generator.generate_all_narratives():
            await generator.sample_output()
            await generator.analyze_generated_content()
            print("Enhanced narrative generation completed successfully")
        else:
            print("Enhanced narrative generation failed")
            sys.exit(1)
    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
