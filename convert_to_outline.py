#!/usr/bin/env python3
"""
Convert hierarchical JSON to flat outline format.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any

def convert_to_outline(input_file: str, output_file: str = None) -> Dict:
    """
    Convert hierarchical JSON structure to flat outline format.
    
    Args:
        input_file: Path to the input JSON file
        output_file: Optional path to save the output JSON file
        
    Returns:
        Dict: Converted outline structure
    """
    # Load the input JSON
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract title
    title = ""
    outline = []
    
    # Process hierarchical structure
    for section in data.get('hierarchical_structure', []):
        if section['type'] == 'title':
            title = section['text']
        elif section['type'] == 'h1':
            # Add H1 to outline
            outline.append({
                "level": "H1",
                "text": section['text'],
                "page": section['page']
            })
            
            # Process subsections (H2s and H3s)
            for subsection in section.get('subsections', []):
                if subsection['type'] == 'h2':
                    outline.append({
                        "level": "H2", 
                        "text": subsection['text'],
                        "page": subsection['page']
                    })
                    
                    # Process H3s under this H2
                    for h3_sub in subsection.get('h3_subsections', []):
                        outline.append({
                            "level": "H3",
                            "text": h3_sub['text'], 
                            "page": h3_sub['page']
                        })
                elif subsection['type'] == 'h3':
                    # Direct H3 under H1 (no parent H2)
                    outline.append({
                        "level": "H3",
                        "text": subsection['text'],
                        "page": subsection['page']
                    })
    
    # Create final structure
    result = {
        "title": title,
        "outline": outline
    }
    
    # Save to output file if specified
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=4, ensure_ascii=False)
        print(f"✅ Outline saved to: {output_file}")
    
    return result

def main():
    """Main function to handle command line arguments."""
    parser = argparse.ArgumentParser(description='Convert hierarchical JSON to flat outline format')
    parser.add_argument('input_file', help='Path to input JSON file')
    parser.add_argument('--output', '-o', help='Path to output JSON file (optional)')
    parser.add_argument('--preview', '-p', action='store_true', help='Preview the outline in console')
    
    args = parser.parse_args()
    
    # Check if input file exists
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"❌ Error: Input file not found: {args.input_file}")
        return
    
    # Generate output filename if not provided
    output_file = args.output
    if not output_file:
        output_file = input_path.parent / f"outline_{input_path.stem}.json"
    
    # Convert the file
    try:
        print(f"🔄 Converting: {args.input_file}")
        result = convert_to_outline(args.input_file, str(output_file))
        
        # Preview if requested
        if args.preview:
            print("\n📋 Preview of converted outline:")
            print("=" * 60)
            print(f"Title: {result['title']}")
            print(f"Total outline items: {len(result['outline'])}")
            print("\nOutline structure:")
            for i, item in enumerate(result['outline'][:10]):  # Show first 10 items
                print(f"  {i+1}. {item['level']}: {item['text'][:50]}{'...' if len(item['text']) > 50 else ''} (page {item['page']})")
            
            if len(result['outline']) > 10:
                print(f"  ... and {len(result['outline']) - 10} more items")
        
        print(f"\n🎉 Conversion completed successfully!")
        print(f"📊 Summary:")
        print(f"   - Title: {result['title']}")
        print(f"   - Total outline items: {len(result['outline'])}")
        
        # Count levels
        level_counts = {}
        for item in result['outline']:
            level = item['level']
            level_counts[level] = level_counts.get(level, 0) + 1
        
        for level, count in sorted(level_counts.items()):
            print(f"   - {level} headings: {count}")
            
    except Exception as e:
        print(f"❌ Error during conversion: {e}")

if __name__ == "__main__":
    main()
