#!/usr/bin/env python3
"""
Combined Results Viewer
=======================

This script provides a readable view of the combined heading extraction results,
showing the hierarchical structure in a tree format.

Usage:
    python view_combined_results.py --json_path "combined_output/combined_file.json"
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any

def print_tree_structure(structure: List[Dict], indent: int = 0) -> None:
    """
    Print the hierarchical structure in a tree format.
    
    Args:
        structure: List of hierarchical sections
        indent: Current indentation level
    """
    for section in structure:
        # Create indentation
        prefix = "  " * indent
        
        # Print section header
        section_type = section.get('type', 'unknown').upper()
        text = section.get('text', 'No text')[:80]  # Limit text length
        page = section.get('page', 'N/A')
        source = section.get('source', 'N/A')
        confidence = section.get('confidence', 0)
        
        print(f"{prefix}[FILE] {section_type} (Page {page}, {source}, {confidence:.3f})")
        print(f"{prefix}   \"{text}\"")
        
        # Print subsections if any
        subsections = section.get('subsections', [])
        if subsections:
            print(f"{prefix}   [?][?][?] Subsections:")
            for i, subsection in enumerate(subsections):
                sub_prefix = prefix + "       "
                if i == len(subsections) - 1:
                    connector = "[?][?][?] "
                else:
                    connector = "[?][?][?] "
                
                sub_type = subsection.get('type', 'unknown').upper()
                sub_text = subsection.get('text', 'No text')[:60]
                sub_page = subsection.get('page', 'N/A')
                sub_source = subsection.get('source', 'N/A')
                sub_confidence = subsection.get('confidence', 0)
                
                print(f"{sub_prefix}{connector}{sub_type} (Page {sub_page}, {sub_source}, {sub_confidence:.3f})")
                print(f"{sub_prefix}    \"{sub_text}\"")
                
                # Print H3 subsections under H2
                h3_subsections = subsection.get('h3_subsections', [])
                if h3_subsections:
                    for j, h3 in enumerate(h3_subsections):
                        if j == len(h3_subsections) - 1:
                            h3_connector = "    [?][?][?] "
                        else:
                            h3_connector = "    [?][?][?] "
                        
                        h3_text = h3.get('text', 'No text')[:50]
                        h3_page = h3.get('page', 'N/A')
                        h3_confidence = h3.get('confidence', 0)
                        
                        print(f"{sub_prefix}{h3_connector}H3 (Page {h3_page}, safe2, {h3_confidence:.3f})")
                        print(f"{sub_prefix}        \"{h3_text}\"")
        
        print()  # Add space between sections

def print_summary_statistics(data: Dict) -> None:
    """
    Print summary statistics of the combined results.
    
    Args:
        data: Combined results data
    """
    print("[STATS] SUMMARY STATISTICS")
    print("=" * 50)
    
    metadata = data.get('metadata', {})
    summary = data.get('summary', {})
    
    print(f"[FILE] Source File: {metadata.get('source_file', 'N/A')}")
    print(f"[?] Processing Time: {metadata.get('processing_timestamp', 'N/A')}")
    print(f"[TOOL] Combination Method: {metadata.get('combination_method', 'N/A')}")
    print()
    
    print(f"[CHART] Content Summary:")
    print(f"   [?] Total sections: {summary.get('total_sections', 0)}")
    print(f"   [?] Titles: {summary.get('titles', 0)}")
    print(f"   [?] H1 sections: {summary.get('h1_sections', 0)}")
    print(f"   [?] Total H2s: {summary.get('total_h2s', 0)}")
    print(f"   [?] Total H3s: {summary.get('total_h3s', 0)}")
    print()
    
    print(f"[FOLDER] Source Files:")
    print(f"   [?] Safe1 output: {Path(metadata.get('safe1_file', 'N/A')).name}")
    print(f"   [?] Safe2 output: {Path(metadata.get('safe2_file', 'N/A')).name}")
    print()

def analyze_hierarchy_distribution(structure: List[Dict]) -> None:
    """
    Analyze and print distribution of headings by page.
    
    Args:
        structure: Hierarchical structure
    """
    print("[STATS] HIERARCHY DISTRIBUTION BY PAGE")
    print("=" * 50)
    
    page_stats = {}
    
    for section in structure:
        page = section.get('page', 0)
        section_type = section.get('type', 'unknown')
        
        if page not in page_stats:
            page_stats[page] = {'title': 0, 'h1': 0, 'h2': 0, 'h3': 0}
        
        page_stats[page][section_type] += 1
        
        # Count subsections
        for subsection in section.get('subsections', []):
            sub_page = subsection.get('page', page)  # Default to parent page
            sub_type = subsection.get('type', 'unknown')
            
            if sub_page not in page_stats:
                page_stats[sub_page] = {'title': 0, 'h1': 0, 'h2': 0, 'h3': 0}
            
            page_stats[sub_page][sub_type] += 1
            
            # Count H3 subsections under H2
            for h3 in subsection.get('h3_subsections', []):
                h3_page = h3.get('page', sub_page)
                
                if h3_page not in page_stats:
                    page_stats[h3_page] = {'title': 0, 'h1': 0, 'h2': 0, 'h3': 0}
                
                page_stats[h3_page]['h3'] += 1
    
    # Print page-by-page breakdown
    for page in sorted(page_stats.keys()):
        stats = page_stats[page]
        total = sum(stats.values())
        if total > 0:
            print(f"Page {page:2d}: {total:2d} headings (T:{stats['title']}, H1:{stats['h1']}, H2:{stats['h2']}, H3:{stats['h3']})")
    
    print()

def print_confidence_analysis(structure: List[Dict]) -> None:
    """
    Analyze confidence scores across different heading types.
    
    Args:
        structure: Hierarchical structure
    """
    print("[STATS] CONFIDENCE ANALYSIS")
    print("=" * 50)
    
    confidence_data = {'title': [], 'h1': [], 'h2': [], 'h3': []}
    
    def collect_confidence(sections: List[Dict]):
        for section in sections:
            section_type = section.get('type', 'unknown')
            confidence = section.get('confidence', 0)
            
            if section_type in confidence_data:
                confidence_data[section_type].append(confidence)
            
            # Recursively collect from subsections
            collect_confidence(section.get('subsections', []))
            
            # Collect from H3 subsections
            for sub in section.get('subsections', []):
                for h3 in sub.get('h3_subsections', []):
                    h3_confidence = h3.get('confidence', 0)
                    confidence_data['h3'].append(h3_confidence)
    
    collect_confidence(structure)
    
    # Print statistics
    for heading_type, confidences in confidence_data.items():
        if confidences:
            avg_conf = sum(confidences) / len(confidences)
            min_conf = min(confidences)
            max_conf = max(confidences)
            print(f"{heading_type.upper():5}: {len(confidences):3d} items, "
                  f"avg: {avg_conf:.3f}, min: {min_conf:.3f}, max: {max_conf:.3f}")
    
    print()

def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(description="View combined heading extraction results")
    parser.add_argument("--json_path", type=str, required=True, 
                       help="Path to combined JSON results file")
    parser.add_argument("--show_tree", action="store_true", default=True,
                       help="Show hierarchical tree structure (default: True)")
    parser.add_argument("--show_stats", action="store_true", default=True,
                       help="Show summary statistics (default: True)")
    parser.add_argument("--show_distribution", action="store_true", default=True,
                       help="Show page distribution (default: True)")
    parser.add_argument("--show_confidence", action="store_true", default=True,
                       help="Show confidence analysis (default: True)")
    
    args = parser.parse_args()
    
    # Load JSON data
    json_path = Path(args.json_path)
    if not json_path.exists():
        print(f"[ERROR] JSON file not found: {json_path}")
        return
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"[ERROR] Error loading JSON file: {e}")
        return
    
    print(f"[SEARCH] COMBINED HEADING EXTRACTION RESULTS")
    print(f"[FOLDER] File: {json_path.name}")
    print("=" * 80)
    print()
    
    # Show summary statistics
    if args.show_stats:
        print_summary_statistics(data)
    
    # Show hierarchy distribution
    if args.show_distribution:
        analyze_hierarchy_distribution(data.get('hierarchical_structure', []))
    
    # Show confidence analysis
    if args.show_confidence:
        print_confidence_analysis(data.get('hierarchical_structure', []))
    
    # Show tree structure
    if args.show_tree:
        print("[?] HIERARCHICAL STRUCTURE")
        print("=" * 50)
        structure = data.get('hierarchical_structure', [])
        if structure:
            print_tree_structure(structure)
        else:
            print("[ERROR] No hierarchical structure found in the data")

if __name__ == "__main__":
    main()
