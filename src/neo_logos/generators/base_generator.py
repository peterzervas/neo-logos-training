#!/usr/bin/env python3
"""
Base Generator for Neo-Logos Training Data

This module provides a base class with common functionality
for all Neo-Logos data generators.
"""

import os
import json
import asyncio
import random
import hashlib
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional, Union

import time

from anthropic import Anthropic, AsyncAnthropic
import httpx

from neo_logos.core.env_loader import load_env_file
from neo_logos.core.logging_utils import get_logger
from neo_logos.config.settings import DEFAULT_MODEL

class BaseGenerator:
    """
    Base class for Neo-Logos data generators.
    
    This class provides common functionality for:
    - Loading framework text
    - Managing API client
    - Handling batches of generations
    - Processing and validating outputs
    - Avoiding duplicates
    - Tracking statistics
    """
    
    def __init__(self, api_key=None, framework_path=None, output_path=None,
                 model=DEFAULT_MODEL, num_examples=None,
                 batch_size=5, max_concurrent=5, checkpoint_interval=5):
        """
        Initialize the base generator.
        
        Args:
            api_key: Anthropic API key (will check environment if None)
            framework_path: Path to Neo-Ethics framework corpus
            output_path: Path to save generated data
            model: Claude model to use
            num_examples: Total number of examples to generate
            batch_size: Number of examples per batch
            max_concurrent: Maximum number of concurrent API calls
            checkpoint_interval: How often to save checkpoints (in batches)
        """
        self.logger = get_logger(self.__class__.__name__)
        
        # Ensure environment variables are loaded
        load_env_file()
        
        # Get API key from param, environment, or .env file
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("API key must be provided via parameter or ANTHROPIC_API_KEY environment variable")
        
        # Async client for real-time mode
        self.client = AsyncAnthropic(
            api_key=self.api_key,
            timeout=httpx.Timeout(300.0, connect=30.0)
        )
        # Sync client for Batch API mode
        self.sync_client = Anthropic(
            api_key=self.api_key,
            timeout=httpx.Timeout(300.0, connect=30.0)
        )
        self.framework_path = framework_path
        self.output_path = output_path
        self.model = model
        self.num_examples = num_examples
        self.batch_size = batch_size
        self.max_concurrent = max_concurrent
        self.checkpoint_interval = checkpoint_interval
        
        # Framework data
        self.framework_text = ""
        self.framework_sections = {}
        
        # Tracking
        self.generated_fingerprints = set()  # For avoiding semantic duplicates
        self.generated_lock = asyncio.Lock()  # Thread safety for fingerprints
        self.file_write_lock = asyncio.Lock()  # Thread safety for file writes
        
        # Initialize data categories
        self.data_categories = self._initialize_data_categories()
        
        # System message for Claude (string for backward compat)
        self.system_message = self._create_system_message()
        # System blocks for prompt caching (list of content blocks)
        self.system_blocks = self._build_system_blocks()

        # Statistics tracking
        self.stats = {
            "batches_requested": 0,
            "batches_completed": 0,
            "examples_requested": 0,
            "examples_generated": 0,
            "duplicates_avoided": 0,
            "inconsistent_examples": 0,
            "start_time": None,
            "end_time": None
        }
        
    
    def _initialize_data_categories(self) -> Dict[str, Dict[str, Any]]:
        """
        Initialize categories of data to generate.
        
        Returns:
            Dictionary mapping category keys to their configurations
        """
        # Override in subclasses to define data categories
        return {}
    
    def _create_system_message(self) -> str:
        """
        Create system message for Claude.
        
        Returns:
            Formatted system message string
        """
        # Override in subclasses to customize the system message
        return """
        You are an expert at creating high-quality training data for fine-tuning language models.
        Generate examples that are detailed, factually accurate, and follow the requested format exactly.
        """
    
    def _get_output_schema(self) -> Optional[Dict[str, Any]]:
        """Return a JSON schema for structured output, or None to use free-form text.

        Override in subclasses to enforce guaranteed-valid JSON responses via
        Anthropic's structured outputs feature.  When a schema is returned,
        all API calls (real-time and batch) will include ``output_config``
        so the model is constrained to produce valid JSON matching the schema.

        Returns:
            JSON schema dict, or None for free-form output.
        """
        return None

    def _output_config(self) -> Optional[Dict[str, Any]]:
        """Build the output_config parameter for API calls."""
        schema = self._get_output_schema()
        if schema is None:
            return None
        return {
            "format": {
                "type": "json_schema",
                "schema": schema,
            }
        }

    def _build_system_blocks(self, use_cache: bool = True) -> List[Dict[str, Any]]:
        """Build system message as content blocks with optional prompt caching.

        Returns a list of content blocks suitable for the Anthropic API's
        ``system`` parameter.  The last block containing large reusable context
        (framework text, identity parameters) is marked with ``cache_control``
        so subsequent calls pay only 10% of the input token cost.

        Override in subclasses to split generator-specific instructions from
        the large cached context.

        Args:
            use_cache: Whether to add cache_control to the context block.

        Returns:
            List of content block dicts.
        """
        blocks = [
            {"type": "text", "text": self.system_message},
        ]
        # If framework text has been loaded and is large enough to cache
        # (minimum 1024 tokens for Sonnet ≈ ~4000 chars), add it as a
        # separate cached block.
        if self.framework_text and len(self.framework_text) > 4000:
            block = {
                "type": "text",
                "text": f"NEO-ETHICS FRAMEWORK REFERENCE:\n\n{self.framework_text}",
            }
            if use_cache:
                block["cache_control"] = {"type": "ephemeral"}
            blocks.append(block)
        return blocks

    def rebuild_system_blocks(self):
        """Rebuild system blocks after framework has been loaded."""
        self.system_blocks = self._build_system_blocks()

    # ------------------------------------------------------------------
    # Batch API support
    # ------------------------------------------------------------------

    def generate_all_batch(self) -> bool:
        """Generate all examples using the Batch API (synchronous).

        Builds all prompts upfront, submits them as a single batch,
        polls for completion, then streams and processes results.

        Returns:
            True on success.
        """
        self.stats["start_time"] = datetime.now()
        self.logger.info("=== BATCH MODE ===")

        # Load framework (just file I/O - call sync version directly)
        if self.framework_path:
            if not self._load_framework_sync():
                self.logger.error("Failed to load framework.")
                return False
            self.rebuild_system_blocks()

        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

        # --- Build phase: create all requests ---
        import asyncio as _aio

        # Get or create event loop for async prompt generation
        try:
            loop = _aio.get_event_loop()
            if loop.is_running():
                # Inside async context - create new loop in thread
                import concurrent.futures
                def _run_async(coro):
                    new_loop = _aio.new_event_loop()
                    try:
                        return new_loop.run_until_complete(coro)
                    finally:
                        new_loop.close()
                run_coro = _run_async
            else:
                run_coro = loop.run_until_complete
        except RuntimeError:
            loop = _aio.new_event_loop()
            run_coro = loop.run_until_complete

        requests = []
        batch_num = 0
        for category_key, category in self.data_categories.items():
            target = category["target_count"]
            self.stats["examples_requested"] += target
            remaining = target
            while remaining > 0:
                size = min(self.batch_size, remaining)
                prompt = run_coro(
                    self.create_prompt(category_key, size)
                )
                params = {
                    "model": self.model,
                    "max_tokens": 8000,
                    "temperature": 0.8,
                    "system": self.system_blocks,
                    "messages": [{"role": "user", "content": prompt}],
                }
                oc = self._output_config()
                if oc:
                    params["output_config"] = oc
                requests.append({
                    "custom_id": f"{category_key}_{batch_num}",
                    "params": params,
                })
                batch_num += 1
                remaining -= size

        self.stats["batches_requested"] = len(requests)
        self.logger.info(
            f"Built {len(requests)} requests across "
            f"{len(self.data_categories)} categories"
        )

        # --- Submit phase ---
        # Batch API limit: 100,000 requests or 256 MB per batch
        self.logger.info("Submitting batch to Anthropic API...")
        batch = self.sync_client.messages.batches.create(requests=requests)
        batch_id = batch.id
        self.logger.info(f"Batch submitted: {batch_id}")

        # Save batch ID to tracking file for crash recovery
        tracking = {
            "batch_id": batch_id,
            "generator": self.__class__.__name__,
            "submitted": datetime.now().isoformat(),
            "requests": len(requests),
            "status": "processing",
        }
        tracking_path = os.path.join(
            os.path.dirname(self.output_path), "batch_tracking.json"
        )
        os.makedirs(os.path.dirname(tracking_path), exist_ok=True)
        with open(tracking_path, "w") as f:
            json.dump(tracking, f, indent=2)
        self.logger.info(f"Batch tracking saved to {tracking_path}")

        # --- Wait phase ---
        self.logger.info("Waiting for batch completion (this may take up to 24 hours)...")
        while True:
            status = self.sync_client.messages.batches.retrieve(batch_id)
            counts = status.request_counts
            self.logger.info(
                f"  Processing: {counts.processing} | "
                f"Succeeded: {counts.succeeded} | "
                f"Errored: {counts.errored} | "
                f"Expired: {counts.expired}"
            )
            if status.processing_status == "ended":
                break
            time.sleep(30)

        # --- Collect phase ---
        self.logger.info("Batch complete. Streaming results...")
        all_examples = []
        for result in self.sync_client.messages.batches.results(batch_id):
            custom_id = result.custom_id
            category_key = custom_id.rsplit("_", 1)[0]

            if result.result.type != "succeeded":
                self.logger.warning(
                    f"Request {custom_id} {result.result.type}: "
                    f"{getattr(result.result, 'error', 'unknown')}"
                )
                continue

            response_text = result.result.message.content[0].text

            # If using structured output, parse as guaranteed JSON
            if self._get_output_schema() is not None:
                try:
                    data = json.loads(response_text)
                    # Extract the array from the wrapper object
                    # (schema wraps items in an array like {narratives: [...]} or {pairs: [...]})
                    parsed = []
                    for key, val in data.items():
                        if isinstance(val, list):
                            parsed = val
                            break
                    if not parsed:
                        parsed = [data]  # Single object, no array wrapper
                except json.JSONDecodeError:
                    self.logger.warning(f"Structured output parse failed for {custom_id}")
                    parsed = self._extract_json_objects_robust(response_text)
            else:
                parsed = self._extract_json_objects_robust(response_text)

            for obj in parsed:
                if not isinstance(obj, dict):
                    continue
                content_field = self._get_content_field_name()
                content = obj.get(content_field, obj.get("narrative", ""))
                if not content:
                    continue
                # Handle list content (e.g., conversation messages)
                if isinstance(content, list):
                    content_str = " ".join(
                        m.get("content", "") for m in content
                        if isinstance(m, dict)
                    )
                else:
                    content_str = str(content)
                fp = self.get_fingerprint(content_str)
                if fp in self.generated_fingerprints:
                    self.stats["duplicates_avoided"] += 1
                    continue
                self.generated_fingerprints.add(fp)
                all_examples.append(obj)
                self.stats["examples_generated"] += 1

            self.stats["batches_completed"] += 1

        # --- Save phase ---
        with open(self.output_path, "w", encoding="utf-8") as f:
            for ex in all_examples:
                f.write(json.dumps(ex) + "\n")

        self.stats["end_time"] = datetime.now()
        duration = (self.stats["end_time"] - self.stats["start_time"]).total_seconds()
        self.logger.info(
            f"Batch generation complete: {len(all_examples)} examples "
            f"in {duration:.1f}s"
        )
        self.logger.info(
            f"  Duplicates avoided: {self.stats['duplicates_avoided']}"
        )

        # Save stats
        stats_path = f"{os.path.splitext(self.output_path)[0]}_stats.json"
        with open(stats_path, "w", encoding="utf-8") as f:
            stats_dict = {
                k: (v.isoformat() if isinstance(v, datetime) else v)
                for k, v in self.stats.items()
            }
            json.dump(stats_dict, f, indent=2)

        return True

    def _load_framework_sync(self) -> bool:
        """Synchronous version of load_framework for batch mode."""
        self.logger.info(f"Loading framework from: {self.framework_path}")
        try:
            if not os.path.exists(self.framework_path):
                self.logger.error(f"Path not found: {self.framework_path}")
                return False
            if os.path.isfile(self.framework_path):
                with open(self.framework_path, 'r', encoding='utf-8') as f:
                    self.framework_text = f.read()
            elif os.path.isdir(self.framework_path):
                self.framework_text = ""
                for filename in sorted(os.listdir(self.framework_path)):
                    if filename.endswith(('.txt', '.md', '.json')):
                        file_path = os.path.join(self.framework_path, filename)
                        with open(file_path, 'r', encoding='utf-8') as f:
                            self.framework_text += f.read() + "\n\n"
            if not self.framework_text:
                self.logger.error("No framework text loaded")
                return False
            self.logger.info(f"Loaded {len(self.framework_text)} chars of framework text")
            return True
        except Exception as e:
            self.logger.error(f"Error loading framework: {e}", exc_info=True)
            return False

    async def load_framework(self) -> bool:
        """
        Load the Neo-Ethics framework as context.
        
        Returns:
            Boolean indicating success or failure
        """
        self.logger.info(f"Loading framework from: {self.framework_path}")
        try:
            if not os.path.exists(self.framework_path):
                self.logger.error(f"Path not found: {self.framework_path}")
                return False
                
            # Handle both file and directory paths
            if os.path.isfile(self.framework_path):
                # Single file handling
                with open(self.framework_path, 'r', encoding='utf-8') as f:
                    self.framework_text = f.read()
                self.logger.info(f"Loaded {len(self.framework_text)} characters from framework file")
            elif os.path.isdir(self.framework_path):
                # Directory handling - combine all text files
                self.framework_text = ""
                files_loaded = 0
                
                for filename in sorted(os.listdir(self.framework_path)):
                    if filename.endswith(('.txt', '.md', '.json')):
                        file_path = os.path.join(self.framework_path, filename)
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            self.framework_text += content + "\n\n"
                        self.logger.info(f"Loaded file: {file_path}")
                        files_loaded += 1
                
                self.logger.info(f"Loaded {files_loaded} files with total {len(self.framework_text)} characters")
            else:
                self.logger.error(f"Path is neither a file nor a directory: {self.framework_path}")
                return False
                
            if not self.framework_text:
                self.logger.error("No framework text was loaded")
                return False
                
            return True
        except Exception as e:
            self.logger.error(f"Error loading framework: {str(e)}", exc_info=True)
            return False
    
    def _get_content_field_name(self) -> str:
        """
        Get the name of the content field in the JSON objects.
        
        Returns:
            Name of the content field
        """
        # Override in subclasses if needed
        return "content"
    
    def get_fingerprint(self, text: str) -> str:
        """
        Generate a fingerprint to detect semantically similar examples.
        
        Args:
            text: Text to generate fingerprint for
            
        Returns:
            Fingerprint string
        """
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
    
    async def is_duplicate(self, text: str) -> bool:
        """
        Check if an example is too similar to existing ones.
        
        Args:
            text: Text to check for duplication
            
        Returns:
            Boolean indicating whether the text is a duplicate
        """
        fingerprint = self.get_fingerprint(text)
        
        async with self.generated_lock:
            return fingerprint in self.generated_fingerprints
    
    async def check_and_add_fingerprint(self, text: str) -> bool:
        """Atomically check if text is a duplicate and register its fingerprint.

        This prevents race conditions where two coroutines could both pass
        the duplicate check and add the same fingerprint.

        Args:
            text: Text to check for duplication.

        Returns:
            True if the text is a duplicate (fingerprint already exists).
            False if the text is new (fingerprint was registered).
        """
        fingerprint = self.get_fingerprint(text)
        async with self.generated_lock:
            if fingerprint in self.generated_fingerprints:
                return True
            self.generated_fingerprints.add(fingerprint)
            return False

    def validate_and_fix_content(self, content: str) -> Tuple[bool, str]:
        """
        Validate and fix content if possible.
        
        Args:
            content: Content to validate
            
        Returns:
            Tuple of (is_valid, fixed_content)
        """
        if not content or len(content) < 10:
            return False, content
            
        # Check for basic validity
        if len(content.strip()) < len(content) * 0.5:
            # Too much whitespace
            content = ' '.join(content.split())
            
        # Simple fix for unclosed quotes
        quote_chars = ['"', "'"]
        for quote in quote_chars:
            if content.count(quote) % 2 == 1:
                # Odd number of quotes, try to fix by adding closing quote
                content += quote
        
        return True, content
    
    def _extract_json_objects(self, text: str) -> List[Dict[str, str]]:
        """
        Extract JSON objects from text.
        
        Args:
            text: Text containing JSON objects
            
        Returns:
            List of extracted JSON objects
        """
        results = []
        
        # Remove code blocks if present
        if "```" in text:
            # Extract content between code fences
            start = text.find("```json")
            if start != -1:
                start = text.find("\n", start) + 1
                end = text.find("```", start)
                if end != -1:
                    text = text[start:end].strip()
            else:
                # Try without language specifier
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
                results.append(obj)
            except json.JSONDecodeError:
                self.logger.warning(f"Failed to parse JSON from line: {line[:50]}...")
                pass
        
        return results

    def _extract_json_objects_robust(self, text: str) -> List[Dict[str, str]]:
        """Extract JSON objects from text, handling multi-line JSON.

        More robust than _extract_json_objects - uses brace-matching to find
        complete JSON objects that may span multiple lines (common when
        narratives contain newlines within the JSON value).

        Args:
            text: Text containing JSON objects.

        Returns:
            List of extracted JSON objects.
        """
        # First try the simple line-by-line approach
        results = self._extract_json_objects(text)
        if results:
            return results

        # If that failed, try brace-matching for multi-line JSON
        self.logger.info("Line-by-line parsing failed, trying brace-matching...")

        # Strip code fences
        if "```" in text:
            start = text.find("```")
            inner = text.find("\n", start) + 1
            end = text.find("```", inner)
            if inner > 0 and end > inner:
                text = text[inner:end].strip()

        results = []
        i = 0
        while i < len(text):
            # Find next opening brace
            start = text.find("{", i)
            if start == -1:
                break

            # Match braces to find the complete object
            depth = 0
            in_string = False
            escape_next = False
            j = start
            while j < len(text):
                ch = text[j]
                if escape_next:
                    escape_next = False
                elif ch == "\\":
                    escape_next = True
                elif ch == '"' and not escape_next:
                    in_string = not in_string
                elif not in_string:
                    if ch == "{":
                        depth += 1
                    elif ch == "}":
                        depth -= 1
                        if depth == 0:
                            # Found complete object
                            candidate = text[start:j + 1]
                            try:
                                obj = json.loads(candidate)
                                if isinstance(obj, dict):
                                    results.append(obj)
                            except json.JSONDecodeError:
                                pass
                            break
                j += 1
            i = j + 1 if j < len(text) else len(text)

        if results:
            self.logger.info(f"Brace-matching found {len(results)} JSON objects")
        else:
            self.logger.warning("Brace-matching also failed to extract JSON objects")

        return results

    async def create_prompt(self, category_key: str, count: int) -> str:
        """
        Create a prompt for generating examples.
        
        Args:
            category_key: Key of the data category
            count: Number of examples to generate
            
        Returns:
            Formatted prompt string
        """
        # Override in subclasses to create category-specific prompts
        return f"Generate {count} examples for {category_key}"
    
    async def process_batch(self, batch_num: int, category_key: str, count: int) -> List[Dict[str, str]]:
        """
        Process a batch of examples for a specific category.
        
        Args:
            batch_num: Batch number for tracking
            category_key: Key of the data category
            count: Number of examples to generate
            
        Returns:
            List of generated examples
        """
        # Override in subclasses to implement specific processing logic
        return []
    
    async def save_checkpoint(self) -> bool:
        """
        Save checkpoint of current generation state.
        
        Returns:
            Boolean indicating success or failure
        """
        try:
            # Convert any datetime objects to strings
            serializable_stats = {}
            for k, v in self.stats.items():
                if isinstance(v, datetime):
                    serializable_stats[k] = v.isoformat()
                else:
                    serializable_stats[k] = v
            
            checkpoint = {
                "timestamp": datetime.now().isoformat(),
                "stats": serializable_stats,
                "categories": {k: {"entries_count": len(v["entries"])} for k, v in self.data_categories.items()},
                "fingerprints": list(self.generated_fingerprints)
            }
            
            # Make checkpoint directory if needed
            checkpoint_dir = os.path.dirname(self.output_path)
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            # Save checkpoint
            checkpoint_path = f"{os.path.splitext(self.output_path)[0]}_checkpoint.json"
            with open(checkpoint_path, 'w', encoding='utf-8') as f:
                json.dump(checkpoint, f, indent=2)
                
            self.logger.info(f"Saved checkpoint to {checkpoint_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving checkpoint: {str(e)}", exc_info=True)
            return False
    
    async def generate_all_examples(self) -> bool:
        """
        Generate all examples across all categories.
        
        Returns:
            Boolean indicating success or failure
        """
        self.stats["start_time"] = datetime.now()
        
        if not await self.load_framework():
            self.logger.error("Failed to load framework. Exiting.")
            return False
        
        # Create output directory if needed
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        
        # Generate examples for each category
        tasks = []
        semaphore = asyncio.Semaphore(self.max_concurrent)
        batch_num = 1
        
        async def process_with_semaphore(batch_num, category_key, count):
            async with semaphore:
                return await self.process_batch(batch_num, category_key, count)
        
        for category_key, category in self.data_categories.items():
            target_count = category["target_count"]
            self.stats["examples_requested"] += target_count
            
            # Calculate number of batches needed for this category
            remaining = target_count
            while remaining > 0:
                batch_size = min(self.batch_size, remaining)
                tasks.append(process_with_semaphore(batch_num, category_key, batch_size))
                batch_num += 1
                remaining -= batch_size
        
        self.stats["batches_requested"] = len(tasks)
        self.logger.info(f"Starting generation of {self.stats['batches_requested']} batches across {len(self.data_categories)} categories")
        
        # Process all batches
        results = await asyncio.gather(*tasks)
        
        # Combine results
        all_examples = []
        for batch in results:
            all_examples.extend(batch)
            
        # Save output
        with open(self.output_path, 'w', encoding='utf-8') as f:
            for example in all_examples:
                f.write(json.dumps(example) + '\n')
        
        self.logger.info(f"Saved {len(all_examples)} examples to {self.output_path}")
        
        # Save statistics
        self.stats["end_time"] = datetime.now()
        stats_path = f"{os.path.splitext(self.output_path)[0]}_stats.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            # Convert datetime to string for JSON serialization
            stats_dict = {k: (v.isoformat() if isinstance(v, datetime) else v) for k, v in self.stats.items()}
            json.dump(stats_dict, f, indent=2)
        
        self.logger.info(f"Saved statistics to {stats_path}")
        
        return True
    
    async def sample_output(self, num_samples=3):
        """
        Display sample output from each category.
        
        Args:
            num_samples: Number of samples to display per category
        """
        self.logger.info("\nSample output:")
        for category_key, category in self.data_categories.items():
            if not category["entries"]:
                continue
                
            self.logger.info(f"\n--- {category['name']} ---")
            samples = random.sample(category["entries"], min(num_samples, len(category["entries"])))
            
            for i, sample in enumerate(samples):
                self.logger.info(f"\nSample {i+1}:")
                content_field = self._get_content_field_name()
                if content_field in sample:
                    content = sample[content_field]
                    # Truncate long content for display
                    display_content = content[:300] + "..." if len(content) > 300 else content
                    self.logger.info(display_content)
                else:
                    self.logger.info(json.dumps(sample, indent=2))
