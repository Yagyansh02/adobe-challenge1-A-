#!/usr/bin/env python3
"""
Enhanced Combined Heading Extractor
===================================

This script extends the combined heading extractor with post-processing constraints:
1. Promotes orphaned H3s to H2s
2. Truncates H2/H3 text at colons 
3. Truncates H2/H3 text to bold portions only

Usage:
    python enhanced_combined_extractor.py --pdf_path "document.pdf"
    python enhanced_combined_extractor.py --json_path "existing_combined.json"
"""

import os
import sys
import json
import argparse
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import subprocess
import shutil

# Import the original combined extractor
from combined_heading_extractor import CombinedHeadingExtractor

class EnhancedCombinedExtractor(CombinedHeadingExtractor):
    """Enhanced version with post-processing constraints"""
    
    def __init__(self, safe1_dir: str = "safe1", safe2_dir: str = "safe2", output_dir: str = "enhanced_output"):
        super().__init__(safe1_dir, safe2_dir, output_dir)
    
    def apply_text_constraints(self, text: str, heading_type: str, formatting: Dict = None) -> str:
        """
        Apply text constraints based on colons and bold formatting.
        
        Args:
            text: Original heading text
            heading_type: Type of heading (h2, h3, etc.)
            formatting: Formatting information from safe2
            
        Returns:
            str: Cleaned text
        """
        if not text:
            return text
        
        original_text = text
        
        # Constraint 2: Truncate at colon for H2/H3
        if heading_type in ['h2', 'h3'] and ':' in text:
            colon_pos = text.find(':')
            text_before_colon = text[:colon_pos].strip()
            
            # Only truncate if the part before colon is substantial (more than 3 words)
            if len(text_before_colon.split()) >= 2:
                text = text_before_colon
                print(f"  ✂️  Truncated at colon: '{original_text}' → '{text}'")
        
        # Constraint 3: Truncate to bold portion for H2/H3
        if heading_type in ['h2', 'h3'] and formatting:
            is_bold = formatting.get('is_bold', False)
            
            # If the heading is marked as bold and it's a long sentence (>6 words)
            if is_bold and len(text.split()) > 6:
                # Look for common patterns where bold text ends
                # Pattern 1: Bold text followed by non-bold explanatory text
                # We'll assume the first 3-5 words are the actual heading
                words = text.split()
                
                # Try to find a natural break point
                break_points = []
                
                # Look for punctuation that might indicate end of heading
                for i, word in enumerate(words[:8]):  # Check first 8 words
                    if word.endswith(('.', ':', ';', '!', '?')):
                        break_points.append(i + 1)
                    # Look for prepositions/articles that might indicate start of description
                    elif word.lower() in ['of', 'the', 'in', 'at', 'on', 'for', 'with', 'this', 'that', 'which']:
                        if i > 1:  # Don't break too early
                            break_points.append(i)
                
                if break_points:
                    # Use the first natural break point
                    truncate_at = min(break_points)
                    truncated_text = ' '.join(words[:truncate_at])
                    
                    # Only apply if the truncated version is reasonable
                    if len(truncated_text.split()) >= 2:
                        text = truncated_text
                        print(f"  ✂️  Truncated bold text: '{original_text}' → '{text}'")
                elif len(words) > 8:
                    # Fallback: take first 4-5 words if it's very long
                    truncated_text = ' '.join(words[:5])
                    text = truncated_text
                    print(f"  ✂️  Truncated long bold text: '{original_text}' → '{text}'")
        
        return text
    
    def promote_orphaned_h3s(self, structure: List[Dict]) -> List[Dict]:
        """
        Promote H3s that don't have a parent H2 to H2 level.
        
        Args:
            structure: Hierarchical structure
            
        Returns:
            List[Dict]: Updated structure with promoted H3s
        """
        print("🔄 Applying constraint: Promoting orphaned H3s to H2s...")
        
        promoted_count = 0
        
        for section in structure:
            if section.get('type') == 'h1':
                subsections = section.get('subsections', [])
                new_subsections = []
                
                for subsection in subsections:
                    if subsection.get('type') == 'h3':
                        # This H3 is directly under H1, promote it to H2
                        subsection['type'] = 'h2'
                        subsection['h3_subsections'] = []  # Initialize empty H3 list
                        promoted_count += 1
                        print(f"  ⬆️  Promoted H3 to H2: '{subsection.get('text', '')[:50]}...'")
                    
                    new_subsections.append(subsection)
                
                section['subsections'] = new_subsections
        
        print(f"✅ Promoted {promoted_count} orphaned H3s to H2s")
        return structure
    
    def apply_text_cleaning_constraints(self, structure: List[Dict]) -> List[Dict]:
        """
        Apply text cleaning constraints (colon truncation and bold truncation).
        
        Args:
            structure: Hierarchical structure
            
        Returns:
            List[Dict]: Updated structure with cleaned text
        """
        print("🔄 Applying text cleaning constraints...")
        
        cleaned_count = 0
        
        def clean_section(section: Dict):
            nonlocal cleaned_count
            
            section_type = section.get('type', '')
            original_text = section.get('text', '')
            formatting = section.get('formatting', {})
            
            if section_type in ['h2', 'h3']:
                cleaned_text = self.apply_text_constraints(original_text, section_type, formatting)
                if cleaned_text != original_text:
                    section['text'] = cleaned_text
                    section['original_text'] = original_text  # Keep original for reference
                    cleaned_count += 1
        
        # Clean all sections in the hierarchy
        for section in structure:
            # Clean title and H1 (though they shouldn't need it based on constraints)
            clean_section(section)
            
            # Clean H2s and their H3s
            for subsection in section.get('subsections', []):
                clean_section(subsection)
                
                # Clean H3s under H2s
                for h3_section in subsection.get('h3_subsections', []):
                    clean_section(h3_section)
        
        print(f"✅ Cleaned text for {cleaned_count} headings")
        return structure
    
    def apply_all_constraints(self, combined_result: Dict) -> Dict:
        """
        Apply all post-processing constraints to the combined result.
        
        Args:
            combined_result: Original combined result
            
        Returns:
            Dict: Enhanced result with constraints applied
        """
        print("\n🛠️  APPLYING POST-PROCESSING CONSTRAINTS")
        print("=" * 60)
        
        structure = combined_result.get('hierarchical_structure', [])
        
        # Constraint 1: Promote orphaned H3s to H2s
        structure = self.promote_orphaned_h3s(structure)
        
        # Constraints 2 & 3: Apply text cleaning
        structure = self.apply_text_cleaning_constraints(structure)
        
        # Update the structure and summary
        combined_result['hierarchical_structure'] = structure
        
        # Recalculate summary statistics
        new_summary = self.calculate_updated_summary(structure)
        combined_result['summary'] = new_summary
        
        # Add processing info
        combined_result['post_processing'] = {
            'constraints_applied': [
                'promoted_orphaned_h3s_to_h2s',
                'truncated_text_at_colons',
                'truncated_bold_text_to_heading_only'
            ],
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"✅ All constraints applied successfully!")
        return combined_result
    
    def calculate_updated_summary(self, structure: List[Dict]) -> Dict:
        """
        Recalculate summary statistics after applying constraints.
        
        Args:
            structure: Updated hierarchical structure
            
        Returns:
            Dict: Updated summary
        """
        total_sections = len(structure)
        titles = len([s for s in structure if s.get('type') == 'title'])
        h1_sections = len([s for s in structure if s.get('type') == 'h1'])
        
        total_h2s = 0
        total_h3s = 0
        
        for section in structure:
            subsections = section.get('subsections', [])
            h2_count = len([sub for sub in subsections if sub.get('type') == 'h2'])
            h3_direct_count = len([sub for sub in subsections if sub.get('type') == 'h3'])
            
            total_h2s += h2_count
            total_h3s += h3_direct_count
            
            # Count H3s under H2s
            for subsection in subsections:
                if subsection.get('type') == 'h2':
                    h3_under_h2_count = len(subsection.get('h3_subsections', []))
                    total_h3s += h3_under_h2_count
        
        return {
            'total_sections': total_sections,
            'titles': titles,
            'h1_sections': h1_sections,
            'total_h2s': total_h2s,
            'total_h3s': total_h3s
        }
    
    def process_pdf_with_constraints(self, pdf_path: str) -> str:
        """
        Complete processing pipeline with constraints applied.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            str: Path to the enhanced output file
        """
        print(f"🎯 Starting Enhanced Combined Heading Extraction Pipeline")
        print(f"📄 PDF: {Path(pdf_path).name}")
        print(f"📁 Output Directory: {self.output_dir}")
        print("=" * 80)
        
        try:
            # Step 1-4: Run original pipeline
            safe1_data = self.run_safe1_extractor(pdf_path)
            safe2_data = self.run_safe2_extractor(pdf_path)
            combined_result = self.combine_results(safe1_data, safe2_data, pdf_path)
            
            # Step 5: Apply constraints
            enhanced_result = self.apply_all_constraints(combined_result)
            
            # Step 6: Save enhanced results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"enhanced_{Path(pdf_path).stem}_{timestamp}.json"
            output_path = self.output_dir / output_filename
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(enhanced_result, f, indent=2, ensure_ascii=False)
            
            print(f"\n💾 Enhanced results saved to: {output_path}")
            
            print("\n🎉 ENHANCED PROCESSING COMPLETE!")
            print("=" * 80)
            print(f"✅ Successfully processed: {Path(pdf_path).name}")
            print(f"📁 All outputs saved in: {self.output_dir}")
            print(f"🔗 Enhanced result: {output_path}")
            
            # Print before/after summary
            print(f"\n📊 CONSTRAINT IMPACT:")
            original_summary = combined_result.get('summary', {})
            enhanced_summary = enhanced_result.get('summary', {})
            
            print(f"   H2s: {original_summary.get('total_h2s', 0)} → {enhanced_summary.get('total_h2s', 0)} "
                  f"({enhanced_summary.get('total_h2s', 0) - original_summary.get('total_h2s', 0):+d})")
            print(f"   H3s: {original_summary.get('total_h3s', 0)} → {enhanced_summary.get('total_h3s', 0)} "
                  f"({enhanced_summary.get('total_h3s', 0) - original_summary.get('total_h3s', 0):+d})")
            
            return str(output_path)
            
        except Exception as e:
            print(f"\n❌ Enhanced processing failed: {e}")
            raise
    
    def apply_constraints_to_existing_json(self, json_path: str) -> str:
        """
        Apply constraints to an existing combined JSON file.
        
        Args:
            json_path: Path to existing combined JSON file
            
        Returns:
            str: Path to the enhanced output file
        """
        print(f"🎯 Applying Constraints to Existing JSON")
        print(f"📄 Input JSON: {Path(json_path).name}")
        print("=" * 80)
        
        try:
            # Load existing JSON
            with open(json_path, 'r', encoding='utf-8') as f:
                combined_result = json.load(f)
            
            # Apply constraints
            enhanced_result = self.apply_all_constraints(combined_result)
            
            # Save enhanced results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            input_name = Path(json_path).stem
            if input_name.startswith('combined_'):
                base_name = input_name[9:]  # Remove 'combined_' prefix
            else:
                base_name = input_name
            
            output_filename = f"enhanced_{base_name}_{timestamp}.json"
            output_path = self.output_dir / output_filename
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(enhanced_result, f, indent=2, ensure_ascii=False)
            
            print(f"\n💾 Enhanced results saved to: {output_path}")
            
            print("\n🎉 CONSTRAINT APPLICATION COMPLETE!")
            print("=" * 80)
            print(f"✅ Successfully enhanced: {Path(json_path).name}")
            print(f"📁 Enhanced result: {output_path}")
            
            # Print before/after summary
            print(f"\n📊 CONSTRAINT IMPACT:")
            original_summary = combined_result.get('summary', {})
            enhanced_summary = enhanced_result.get('summary', {})
            
            print(f"   H2s: {original_summary.get('total_h2s', 0)} → {enhanced_summary.get('total_h2s', 0)} "
                  f"({enhanced_summary.get('total_h2s', 0) - original_summary.get('total_h2s', 0):+d})")
            print(f"   H3s: {original_summary.get('total_h3s', 0)} → {enhanced_summary.get('total_h3s', 0)} "
                  f"({enhanced_summary.get('total_h3s', 0) - original_summary.get('total_h3s', 0):+d})")
            
            return str(output_path)
            
        except Exception as e:
            print(f"\n❌ Constraint application failed: {e}")
            raise

def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(description="Enhanced combined heading extraction with post-processing constraints")
    parser.add_argument("--pdf_path", type=str, help="Path to PDF file (for new processing)")
    parser.add_argument("--json_path", type=str, help="Path to existing combined JSON file (for constraint application only)")
    parser.add_argument("--output_dir", type=str, default="enhanced_output", 
                       help="Output directory for results (default: enhanced_output)")
    parser.add_argument("--safe1_dir", type=str, default="safe1", 
                       help="Directory containing safe1 model (default: safe1)")
    parser.add_argument("--safe2_dir", type=str, default="safe2", 
                       help="Directory containing safe2 model (default: safe2)")
    
    args = parser.parse_args()
    
    if not args.pdf_path and not args.json_path:
        print("❌ Must provide either --pdf_path or --json_path")
        parser.print_help()
        sys.exit(1)
    
    if args.pdf_path and args.json_path:
        print("❌ Provide either --pdf_path OR --json_path, not both")
        parser.print_help()
        sys.exit(1)
    
    try:
        # Initialize enhanced extractor
        extractor = EnhancedCombinedExtractor(
            safe1_dir=args.safe1_dir,
            safe2_dir=args.safe2_dir,
            output_dir=args.output_dir
        )
        
        if args.pdf_path:
            # Verify PDF exists
            if not Path(args.pdf_path).exists():
                print(f"❌ PDF file not found: {args.pdf_path}")
                sys.exit(1)
            
            # Process PDF with constraints
            output_path = extractor.process_pdf_with_constraints(args.pdf_path)
        else:
            # Verify JSON exists
            if not Path(args.json_path).exists():
                print(f"❌ JSON file not found: {args.json_path}")
                sys.exit(1)
            
            # Apply constraints to existing JSON
            output_path = extractor.apply_constraints_to_existing_json(args.json_path)
        
        print(f"\n📋 View your enhanced results:")
        print(f"   Enhanced output: {output_path}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
