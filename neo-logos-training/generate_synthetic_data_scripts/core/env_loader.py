#!/usr/bin/env python3
"""
Environment Variable Loader

This module provides functionality to load environment variables from a .env file,
making API keys and other configuration available to scripts.
"""

import os
import sys
from pathlib import Path

def load_env_file(env_path=None):
    """
    Load environment variables from .env file
    
    Args:
        env_path: Path to .env file (default: looks for .env in parent directories)
        
    Returns:
        bool: True if environment variables were loaded successfully
    """
    if env_path is None:
        # Start from the current script's directory
        current_dir = Path(__file__).parent.absolute()
        
        # Look for .env in parent directories
        search_dirs = [current_dir]
        for i in range(3):  # Check up to 3 levels up
            parent_dir = search_dirs[-1].parent
            search_dirs.append(parent_dir)
            
        for directory in search_dirs:
            potential_env = directory / '.env'
            if potential_env.exists():
                env_path = potential_env
                break
                
            # Also check for ../neo-logos-training/.env pattern
            neo_logos_dir = directory / 'neo-logos-training'
            if neo_logos_dir.exists():
                potential_env = neo_logos_dir / '.env'
                if potential_env.exists():
                    env_path = potential_env
                    break
    
    if env_path is None or not Path(env_path).exists():
        print(f"Warning: .env file not found")
        return False
    
    try:
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                    
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                
                # Remove quotes if present
                if value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                elif value.startswith("'") and value.endswith("'"):
                    value = value[1:-1]
                
                # Set environment variable
                os.environ[key] = value
                
        print(f"Loaded environment variables from {env_path}")
        return True
    except Exception as e:
        print(f"Error loading .env file: {str(e)}")
        return False

# Auto-load environment variables when module is imported
load_env_file()
