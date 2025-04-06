#!/usr/bin/env python3
"""
Test Script for Environment Variables

This script verifies that the .env file is properly loaded
and the ANTHROPIC_API_KEY is accessible to the generator scripts.
"""

import os
import sys
from core.env_loader import load_env_file

def test_env_loading():
    """Test if the environment variables are loaded correctly"""
    print("Testing environment variable loading...")
    
    # Try to load environment variables
    load_result = load_env_file()
    print(f"Environment loading result: {load_result}")
    
    # Check if ANTHROPIC_API_KEY is available
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        # Mask the API key for security (only show first 8 chars)
        visible_part = api_key[:8]
        masked_part = "*" * (len(api_key) - 8)
        print(f"✅ ANTHROPIC_API_KEY found: {visible_part}{masked_part}")
        print(f"  Length: {len(api_key)} characters")
        return True
    else:
        print("❌ ANTHROPIC_API_KEY not found in environment variables")
        return False

if __name__ == "__main__":
    print("\n==== Environment Variable Test ====\n")
    result = test_env_loading()
    
    # Show current directory for debugging
    print(f"\nCurrent directory: {os.getcwd()}")
    
    # Exit with appropriate code
    if result:
        print("\n✅ Test Passed: Environment variables loaded successfully")
        sys.exit(0)
    else:
        print("\n❌ Test Failed: Could not load API key from environment")
        print("Please ensure the .env file exists and contains ANTHROPIC_API_KEY")
        sys.exit(1)
