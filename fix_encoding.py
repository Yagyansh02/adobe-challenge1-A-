#!/usr/bin/env python3
"""
Fix Encoding Issues
==================

This script fixes Unicode encoding issues in the PDF extractors by:
1. Setting proper environment variables for UTF-8 encoding
2. Replacing problematic Unicode characters with ASCII equivalents

Usage:
    python fix_encoding.py
"""

import os
import sys
import re
from pathlib import Path

def set_utf8_environment():
    """Set environment variables to use UTF-8 encoding."""
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    os.environ['PYTHONUTF8'] = '1'
    print("Set environment variables for UTF-8 encoding")

def replace_unicode_in_file(file_path):
    """Replace Unicode emoji characters with ASCII equivalents."""
    
    # Mapping of Unicode emojis to ASCII equivalents
    emoji_replacements = {
        '📦': '[LOAD]',
        '✅': '[OK]',
        '❌': '[ERROR]',
        '🔄': '[PROCESS]',
        '🚀': '[START]',
        '📊': '[STATS]',
        '📁': '[FOLDER]',
        '📄': '[FILE]',
        '🎯': '[TARGET]',
        '🏛️': '[VIEW]',
        '📋': '[RESULT]',
        '🎉': '[COMPLETE]',
        '🔍': '[SEARCH]',
        '👋': '[EXIT]',
        '💾': '[SAVE]',
        '🔗': '[COMBINE]',
        '📝': '[LOG]',
        '🎨': '[FORMAT]',
        '⚡': '[FAST]',
        '🌟': '[STAR]',
        '🔧': '[TOOL]',
        '📈': '[CHART]',
        '🏁': '[FINISH]',
        '🎪': '[DEMO]',
        '🎭': '[MASK]',
        '🎸': '[MUSIC]',
        '🎲': '[DICE]',
        '🎮': '[GAME]',
        '🎤': '[MIC]',
        '🎵': '[NOTE]',
        '🎶': '[NOTES]',
        '🎷': '[SAX]',
        '🎺': '[TRUMPET]',
        '🎻': '[VIOLIN]',
        '🎼': '[SCORE]',
        '🎾': '[TENNIS]',
        '🎿': '[SKI]',
        '🏀': '[BALL]',
        '🏈': '[FOOTBALL]',
        '🏉': '[RUGBY]',
        '🏊': '[SWIM]',
        '🏋️': '[LIFT]',
        '🏌️': '[GOLF]',
        '🏍️': '[BIKE]',
        '🏎️': '[CAR]',
        '🏏': '[CRICKET]',
        '🏐': '[VOLLEYBALL]',
        '🏑': '[HOCKEY]',
        '🏒': '[HOCKEY2]',
        '🏓': '[PING]',
        '🏔️': '[MOUNTAIN]',
        '🏕️': '[CAMP]',
        '🏖️': '[BEACH]',
        '🏗️': '[CONSTRUCTION]',
        '🏘️': '[HOUSES]',
        '🏙️': '[CITY]',
        '🏚️': '[HOUSE]',
        '🏛️': '[BUILDING]',
        '🏜️': '[DESERT]',
        '🏝️': '[ISLAND]',
        '🏞️': '[PARK]',
        '🏟️': '[STADIUM]',
        '🏠': '[HOME]',
        '🏡': '[HOUSE2]',
        '🏢': '[OFFICE]',
        '🏣': '[POST]',
        '🏤': '[BANK]',
        '🏥': '[HOSPITAL]',
        '🏦': '[BANK2]',
        '🏧': '[ATM]',
        '🏨': '[HOTEL]',
        '🏩': '[MOTEL]',
        '🏪': '[STORE]',
        '🏫': '[SCHOOL]',
        '🏬': '[MALL]',
        '🏭': '[FACTORY]',
        '🏮': '[LANTERN]',
        '🏯': '[CASTLE]',
        '🏰': '[CASTLE2]',
        # Add more as needed
    }
    
    try:
        # Read the file
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        
        original_content = content
        
        # Replace emojis with ASCII equivalents
        for emoji, replacement in emoji_replacements.items():
            content = content.replace(emoji, replacement)
        
        # Replace any remaining problematic Unicode characters with a generic replacement
        # This regex finds characters outside the basic ASCII range
        content = re.sub(r'[^\x00-\x7F]', '[?]', content)
        
        # Only write back if changes were made
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Fixed Unicode characters in: {file_path}")
            return True
        else:
            print(f"No Unicode issues found in: {file_path}")
            return False
            
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def fix_all_extractors():
    """Fix Unicode issues in all PDF extractor files."""
    
    print("🔧 FIXING UNICODE ENCODING ISSUES")
    print("=" * 50)
    
    # Set UTF-8 environment
    set_utf8_environment()
    
    # Files to fix
    files_to_fix = [
        "safe1/pdf_heading_extractor.py",
        "safe2/pdf_heading_extractor.py",
        "combined_heading_extractor.py",
        "test_all_pdfs.py",
        "view_final_results.py",
        "view_combined_results.py"
    ]
    
    fixed_count = 0
    total_count = 0
    
    for file_path in files_to_fix:
        full_path = Path(file_path)
        if full_path.exists():
            total_count += 1
            if replace_unicode_in_file(full_path):
                fixed_count += 1
        else:
            print(f"File not found: {file_path}")
    
    print(f"\nFixed {fixed_count}/{total_count} files")
    print("✅ Unicode encoding fix completed!")
    
    return fixed_count > 0

def create_run_script():
    """Create a batch script that sets encoding and runs the test."""
    
    batch_content = '''@echo off
REM Set UTF-8 encoding for Python
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1

REM Run the test script
python test_all_pdfs.py %*

pause
'''
    
    with open("run_test_utf8.bat", 'w') as f:
        f.write(batch_content)
    
    print("Created run_test_utf8.bat - use this to run tests with proper encoding")

if __name__ == "__main__":
    print("Starting Unicode encoding fix...")
    
    try:
        # Fix the files
        changes_made = fix_all_extractors()
        
        # Create batch script for future runs
        create_run_script()
        
        if changes_made:
            print("\n" + "="*50)
            print("IMPORTANT: Files have been modified to fix Unicode issues.")
            print("You can now run the test script normally or use run_test_utf8.bat")
            print("="*50)
        else:
            print("\nNo changes were needed. You can run normally or use run_test_utf8.bat for extra safety.")
            
    except Exception as e:
        print(f"Error during fix: {e}")
        sys.exit(1)
