#!/usr/bin/env python3
"""
View Final Combined Results
===========================

Display the hierarchical structure of the final combined results.
"""

import json
import sys
from pathlib import Path

def display_hierarchical_structure(json_file: str):
    """Display the structure in a tree format (handles both hierarchical and flat outline formats)."""
    
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print("[BUILDING]  FINAL COMBINED HEADING STRUCTURE")
    print("=" * 80)
    
    # Check if this is the new flat outline format
    if 'title' in data and 'outline' in data:
        display_flat_outline_format(data, json_file)
    else:
        display_old_hierarchical_format(data, json_file)

def display_flat_outline_format(data: dict, json_file: str):
    """Display the flat outline format."""
    
    print(f"[FILE] Source: {Path(json_file).name}")
    print(f"[RESULT] Format: Flat Outline (convert_to_outline.py compatible)")
    print()
    
    title = data.get('title', '')
    outline = data.get('outline', [])
    
    print(f"[?] TITLE: {title}")
    print()
    
    print(f"[STATS] SUMMARY:")
    # Count levels
    level_counts = {}
    for item in outline:
        level = item['level']
        level_counts[level] = level_counts.get(level, 0) + 1
    
    print(f"   [?] Total outline items: {len(outline)}")
    for level, count in sorted(level_counts.items()):
        print(f"   [?] {level} headings: {count}")
    print()
    
    print("[RESULT] OUTLINE STRUCTURE:")
    print("-" * 60)
    
    current_h1 = None
    current_h2 = None
    h1_count = 0
    h2_count = 0
    h3_count = 0
    
    for i, item in enumerate(outline):
        level = item['level']
        text = item['text']
        page = item['page']
        
        if level == 'H1':
            h1_count += 1
            current_h1 = text
            current_h2 = None
            print(f"\n[?] {h1_count}. H1: {text}")
            print(f"   [?][?] Page {page}")
        
        elif level == 'H2':
            h2_count += 1
            current_h2 = text
            prefix = "   [?][?]" if i < len(outline) - 1 and outline[i+1]['level'] in ['H2', 'H3'] else "   [?][?]"
            print(f"{prefix} {h2_count}. H2: {text[:60]}{'...' if len(text) > 60 else ''}")
            print(f"   [?]   [?][?] Page {page}")
        
        elif level == 'H3':
            h3_count += 1
            # Check if next item is H3 to determine connector
            next_is_h3 = (i < len(outline) - 1 and 
                         outline[i+1]['level'] == 'H3')
            prefix = "      [?][?]" if next_is_h3 else "      [?][?]"
            print(f"{prefix} {h3_count}. H3: {text[:50]}{'...' if len(text) > 50 else ''}")
            print(f"      [?]   [?][?] Page {page}")

def display_old_hierarchical_format(data: dict, json_file: str):
    """Display the old hierarchical format."""
    
    metadata = data.get('metadata', {})
    print(f"[FILE] Source: {metadata.get('source_file', 'Unknown')}")
    print(f"[TOOL] Method: {metadata.get('combination_method', 'Unknown')}")
    print(f"[?] Processed: {metadata.get('processing_timestamp', 'Unknown')}")
    print()
    
    summary = data.get('summary', {})
    print(f"[STATS] SUMMARY:")
    print(f"   [?] Total sections: {summary.get('total_sections', 0)}")
    print(f"   [?] Titles: {summary.get('titles', 0)}")
    print(f"   [?] H1 sections: {summary.get('h1_sections', 0)}")
    print(f"   [?] Total H2s: {summary.get('total_h2s', 0)}")
    print(f"   [?] Total H3s: {summary.get('total_h3s', 0)}")
    print()
    
    structure = data.get('hierarchical_structure', [])
    
    for section in structure:
        if section['type'] == 'title':
            print(f"[?] TITLE: {section['text']}")
            print(f"   [?][?] Page {section['page']} | Source: {section['source']} | Confidence: {section['confidence']:.3f}")
            print()
        
        elif section['type'] == 'h1':
            subsection_count = len(section.get('subsections', []))
            h2_count = len([s for s in section.get('subsections', []) if s['type'] == 'h2'])
            h3_count = len([s for s in section.get('subsections', []) if s['type'] == 'h3'])
            h3_under_h2_count = sum(len(s.get('h3_subsections', [])) for s in section.get('subsections', []) if s['type'] == 'h2')
            total_h3s = h3_count + h3_under_h2_count
            
            print(f"[?] H1: {section['text']}")
            print(f"   [?][?] Page {section['page']} | Source: {section['source']} | Confidence: {section['confidence']:.3f}")
            print(f"   [?][?] Contains: {h2_count} H2s, {total_h3s} H3s ({subsection_count} total subsections)")
            
            # Show H2s and their H3s
            for subsection in section.get('subsections', []):
                if subsection['type'] == 'h2':
                    h3_count_in_h2 = len(subsection.get('h3_subsections', []))
                    print(f"      [?][?] H2: {subsection['text'][:60]}{'...' if len(subsection['text']) > 60 else ''}")
                    print(f"      [?]   [?][?] Page {subsection['page']} | {h3_count_in_h2} H3s")
                    
                    # Show first few H3s
                    for i, h3 in enumerate(subsection.get('h3_subsections', [])[:3]):
                        prefix = "[?]       [?][?]" if i < min(2, len(subsection.get('h3_subsections', [])) - 1) else "[?]       [?][?]"
                        print(f"      {prefix} H3: {h3['text'][:50]}{'...' if len(h3['text']) > 50 else ''}")
                    
                    if len(subsection.get('h3_subsections', [])) > 3:
                        print(f"      [?]       [?][?] ... and {len(subsection.get('h3_subsections', [])) - 3} more H3s")
                
                elif subsection['type'] == 'h3':
                    print(f"      [?][?] H3 (direct): {subsection['text'][:60]}{'...' if len(subsection['text']) > 60 else ''}")
                    print(f"      [?]   [?][?] Page {subsection['page']}")
            
            print()

def main():
    if len(sys.argv) > 1:
        json_file = sys.argv[1]
    else:
        # Find the most recent combined output
        output_dirs = [
            Path("test_output"),
            Path("final_combined_output"),
            Path("improved_combined_output"), 
            Path("combined_output")
        ]
        
        json_file = None
        for output_dir in output_dirs:
            if output_dir.exists():
                # Look for combined files (flat outline format)
                json_files = list(output_dir.glob("**/combined_*.json"))
                # Exclude hierarchical files
                json_files = [f for f in json_files if 'hierarchical' not in f.name]
                if json_files:
                    # Get the most recent
                    json_file = max(json_files, key=lambda x: x.stat().st_mtime)
                    break
        
        if not json_file:
            print("[ERROR] No combined JSON file found. Please specify the path.")
            print("[?] Looking for files matching pattern: combined_*.json")
            sys.exit(1)
    
    if not Path(json_file).exists():
        print(f"[ERROR] File not found: {json_file}")
        sys.exit(1)
    
    display_hierarchical_structure(json_file)

if __name__ == "__main__":
    main()
