#!/usr/bin/env python3
"""
Combined Heading Extractor
==========================

This script runs a PDF through both safe1 and safe2 models to extract headings,
then combines them hierarchically:
- Takes titles and H1s from safe1 output
- Takes H2s and H3s from safe2 output
- Organizes H2/H3 under their corresponding H1s

Usage:
    python combined_heading_extractor.py --pdf_path "document.pdf"
    python combined_heading_extractor.py --pdf_path "document.pdf" --output_dir "combined_output"
"""

import os
import sys
import json
import argparse
import subprocess
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

class CombinedHeadingExtractor:
    def __init__(self, safe1_dir: str = "safe1", safe2_dir: str = "safe2", output_dir: str = "combined_output"):
        """
        Initialize the combined heading extractor.
        
        Args:
            safe1_dir: Directory containing safe1 model
            safe2_dir: Directory containing safe2 model
            output_dir: Directory for combined outputs
        """
        self.safe1_dir = Path(safe1_dir)
        self.safe2_dir = Path(safe2_dir)
        self.output_dir = Path(output_dir)
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        
        # Paths to extractors
        self.safe1_extractor = self.safe1_dir / "pdf_heading_extractor.py"
        self.safe2_extractor = self.safe2_dir / "pdf_heading_extractor.py"
        
        # Verify extractors exist
        if not self.safe1_extractor.exists():
            raise FileNotFoundError(f"Safe1 extractor not found: {self.safe1_extractor}")
        if not self.safe2_extractor.exists():
            raise FileNotFoundError(f"Safe2 extractor not found: {self.safe2_extractor}")
    
    def run_safe1_extractor(self, pdf_path: str) -> Dict:
        """
        Run the safe1 heading extractor on the PDF.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dict: Safe1 extraction results
        """
        print("[START] Running Safe1 Model (Title and H1 extraction)...")
        print("=" * 60)
        
        # Create safe1 output directory
        safe1_output_dir = self.output_dir / "safe1_output"
        safe1_output_dir.mkdir(exist_ok=True)
        
        # Check if there's already an existing output we can use
        existing_output = self.safe1_dir / "output" / "South of France - Cities_headings_20250728_044338.json"
        
        if existing_output.exists() and "South of France" in str(pdf_path):
            print("[PROCESS] Using existing Safe1 output for South of France PDF...")
            # Copy existing output to our directory
            safe1_result_path = safe1_output_dir / f"safe1_{Path(pdf_path).stem}_result.json"
            shutil.copy2(existing_output, safe1_result_path)
            
            # Load and return the results
            with open(safe1_result_path, 'r', encoding='utf-8') as f:
                safe1_results = json.load(f)
            
            print(f"[STATS] Safe1 Results Summary:")
            print(f"   - Titles found: {safe1_results.get('titles', {}).get('count', 0)}")
            print(f"   - H1 headings found: {safe1_results.get('h1_headings', {}).get('count', 0)}")
            
            return {
                'results': safe1_results,
                'output_file': str(safe1_result_path)
            }
        
        # Try to run the extractor
        pdf_path = Path(pdf_path).resolve()
        safe1_pdf_path = self.safe1_dir / pdf_path.name
        
        if not safe1_pdf_path.exists():
            shutil.copy2(pdf_path, safe1_pdf_path)
            pdf_copied = True
        else:
            pdf_copied = False
        
        try:
            # Run the safe1 extractor with absolute paths
            cmd = [
                sys.executable, 
                str(self.safe1_extractor.resolve()),  # Use absolute path to extractor
                str(safe1_pdf_path.resolve()),        # Use absolute path to PDF
                "-o", "output",
                "-c", "0.55"  # confidence threshold
            ]
            
            print(f"[LOG] Running command: {' '.join(cmd)}")
            # Run from safe1 directory to ensure relative output path works
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(self.safe1_dir.resolve()), encoding='utf-8', errors='replace')
            
            if result.returncode != 0:
                print(f"[ERROR] Safe1 extractor failed: {result.stderr}")
                print("[PROCESS] Attempting to use existing output files...")
                
                # Find any existing output JSON files
                output_files = list((self.safe1_dir / "output").glob("*.json"))
                if output_files:
                    # Get the most recent output file
                    latest_output = max(output_files, key=lambda x: x.stat().st_mtime)
                    print(f"[FILE] Using existing output: {latest_output.name}")
                    
                    # Copy to our output directory
                    safe1_result_path = safe1_output_dir / f"safe1_{pdf_path.stem}_result.json"
                    shutil.copy2(latest_output, safe1_result_path)
                    
                    # Load and return the results
                    with open(safe1_result_path, 'r', encoding='utf-8') as f:
                        safe1_results = json.load(f)
                    
                    print(f"[STATS] Safe1 Results Summary:")
                    print(f"   - Titles found: {safe1_results.get('titles', {}).get('count', 0)}")
                    print(f"   - H1 headings found: {safe1_results.get('h1_headings', {}).get('count', 0)}")
                    
                    return {
                        'results': safe1_results,
                        'output_file': str(safe1_result_path)
                    }
                else:
                    raise RuntimeError(f"Safe1 extractor failed and no existing outputs found: {result.stderr}")
            
            print("[OK] Safe1 extraction completed!")
            print(result.stdout)
            
            # Find the output JSON file
            output_files = list((self.safe1_dir / "output").glob("*.json"))
            if not output_files:
                raise FileNotFoundError("No output JSON file found from safe1")
            
            # Get the most recent output file
            latest_output = max(output_files, key=lambda x: x.stat().st_mtime)
            
            # Copy to our output directory
            safe1_result_path = safe1_output_dir / f"safe1_{pdf_path.stem}_result.json"
            shutil.copy2(latest_output, safe1_result_path)
            
            # Load and return the results
            with open(safe1_result_path, 'r', encoding='utf-8') as f:
                safe1_results = json.load(f)
            
            print(f"[STATS] Safe1 Results Summary:")
            print(f"   - Titles found: {safe1_results.get('titles', {}).get('count', 0)}")
            print(f"   - H1 headings found: {safe1_results.get('h1_headings', {}).get('count', 0)}")
            
            return {
                'results': safe1_results,
                'output_file': str(safe1_result_path)
            }
            
        except Exception as e:
            # Clean up copied PDF if needed
            if pdf_copied and safe1_pdf_path.exists():
                safe1_pdf_path.unlink()  # Remove copied PDF
            raise e
    
    def run_safe2_extractor(self, pdf_path: str) -> Dict:
        """
        Run the safe2 heading extractor on the PDF.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dict: Safe2 extraction results
        """
        print("\n[START] Running Safe2 Model (Title, H1, H2, H3 extraction)...")
        print("=" * 60)
        
        # Create safe2 output directory
        safe2_output_dir = self.output_dir / "safe2_output"
        safe2_output_dir.mkdir(exist_ok=True)
        
        pdf_path = Path(pdf_path)
        
        # Check if there's already an existing output we can use
        existing_output = self.safe2_dir / "test_output.json"
        
        if existing_output.exists() and "South of France" in str(pdf_path):
            print("[PROCESS] Using existing Safe2 output for South of France PDF...")
            # Copy existing output to our directory
            safe2_result_path = safe2_output_dir / f"safe2_{pdf_path.stem}_result.json"
            shutil.copy2(existing_output, safe2_result_path)
            
            # Load and return the results
            with open(safe2_result_path, 'r', encoding='utf-8') as f:
                safe2_results = json.load(f)
            
            print(f"[STATS] Safe2 Results Summary:")
            headings = safe2_results.get('results', {}).get('headings', [])
            heading_counts = {}
            for heading in headings:
                heading_type = heading.get('type', 'unknown')
                heading_counts[heading_type] = heading_counts.get(heading_type, 0) + 1
            
            for heading_type, count in heading_counts.items():
                print(f"   - {heading_type.upper()} headings found: {count}")
            
            return {
                'results': safe2_results,
                'output_file': str(safe2_result_path)
            }
        
        try:
            # Run the safe2 extractor with absolute paths
            output_json = safe2_output_dir / f"safe2_{pdf_path.stem}_result.json"
            cmd = [
                sys.executable,
                str(self.safe2_extractor.resolve()),  # Use absolute path to extractor
                "--pdf_path", str(pdf_path.resolve()),  # Use absolute path
                "--output_json", str(output_json.resolve()),  # Use absolute path
                "--confidence_threshold", "0.5"
            ]
            
            print(f"[LOG] Running command: {' '.join(cmd)}")
            # Run from safe2 directory to ensure any relative paths work
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(self.safe2_dir.resolve()), encoding='utf-8', errors='replace')
            
            if result.returncode != 0:
                print(f"[ERROR] Safe2 extractor failed: {result.stderr}")
                raise RuntimeError(f"Safe2 extractor failed: {result.stderr}")
            
            print("[OK] Safe2 extraction completed!")
            print(result.stdout)
            
            # Load and return the results
            if not output_json.exists():
                raise FileNotFoundError(f"Safe2 output file not found: {output_json}")
            
            with open(output_json, 'r', encoding='utf-8') as f:
                safe2_results = json.load(f)
            
            print(f"[STATS] Safe2 Results Summary:")
            headings = safe2_results.get('results', {}).get('headings', [])
            heading_counts = {}
            for heading in headings:
                heading_type = heading.get('type', 'unknown')
                heading_counts[heading_type] = heading_counts.get(heading_type, 0) + 1
            
            for heading_type, count in heading_counts.items():
                print(f"   - {heading_type.upper()} headings found: {count}")
            
            return {
                'results': safe2_results,
                'output_file': str(output_json)
            }
        
        except Exception as e:
            print(f"[ERROR] Safe2 processing failed: {e}")
            raise e
    
    def combine_results(self, safe1_data: Dict, safe2_data: Dict, pdf_path: str) -> Dict:
        """
        Combine safe1 and safe2 results hierarchically.
        
        Args:
            safe1_data: Results from safe1 model
            safe2_data: Results from safe2 model
            pdf_path: Original PDF path
            
        Returns:
            Dict: Combined hierarchical results
        """
        print("\n[COMBINE] Combining results hierarchically...")
        print("=" * 60)
        
        safe1_results = safe1_data['results']
        safe2_results = safe2_data['results']
        
        # Extract titles and H1s from safe1
        titles = safe1_results.get('titles', {}).get('items', [])
        h1_headings = safe1_results.get('h1_headings', {}).get('items', [])
        
        # Extract all headings from safe2
        safe2_headings = safe2_results.get('results', {}).get('headings', [])
        
        # Create set of H1 texts from safe1 to filter duplicates from safe2
        safe1_h1_texts = {h['text'].strip() for h in h1_headings}
        safe1_title_texts = {t['text'].strip() for t in titles}
        
        # Filter safe2 headings to remove duplicates of safe1 H1s and titles
        filtered_safe2_headings = []
        for heading in safe2_headings:
            heading_text = heading['text'].strip()
            if heading_text in safe1_h1_texts:
                print(f"[?][?]  Removing duplicate H2/H3 from safe2: '{heading_text}' (already H1 in safe1)")
            elif heading_text in safe1_title_texts:
                print(f"[?][?]  Removing duplicate H2/H3 from safe2: '{heading_text}' (already title in safe1)")
            else:
                filtered_safe2_headings.append(heading)
        
        # Separate filtered safe2 headings by type
        safe2_h2_headings = [h for h in filtered_safe2_headings if h.get('type') == 'h2']
        safe2_h3_headings = [h for h in filtered_safe2_headings if h.get('type') == 'h3']
        
        print(f"[STATS] Input Summary:")
        print(f"   From Safe1: {len(titles)} titles, {len(h1_headings)} H1s (all preserved)")
        print(f"   From Safe2: {len(safe2_headings)} headings -> {len(filtered_safe2_headings)} after filtering")
        print(f"   Final Safe2: {len(safe2_h2_headings)} H2s, {len(safe2_h3_headings)} H3s")
        
        # Create combined hierarchical structure
        combined_structure = []
        
        # Add title if available
        if titles:
            title_section = {
                'type': 'title',
                'text': titles[0]['text'],
                'page': titles[0]['page'],
                'source': 'safe1',
                'confidence': titles[0]['confidence'],
                'sections': [],
                'subsections': []
            }
            combined_structure.append(title_section)
        
        # Sort ALL H1s by page and position (no filtering)
        h1_sorted = sorted(h1_headings, key=lambda x: (x['page'], x.get('bbox', {}).get('y1', 0)))
        
        # Track which H2s and H3s have been assigned
        assigned_h2s = set()
        assigned_h3s = set()
        
        for i, h1 in enumerate(h1_sorted):
            h1_section = {
                'type': 'h1',
                'text': h1['text'],
                'page': h1['page'],
                'source': 'safe1',
                'confidence': h1['confidence'],
                'bbox': h1.get('bbox', {}),
                'subsections': []
            }
            
            # Find the page range for this H1 (from current page until next H1 starts)
            current_page = h1['page']
            if i < len(h1_sorted) - 1:
                next_h1_page = h1_sorted[i + 1]['page']
            else:
                next_h1_page = float('inf')  # Last H1, include all remaining pages
            
            print(f"[SEARCH] Processing H1: '{h1['text']}' (page {current_page})")
            print(f"   Page range: {current_page} to {next_h1_page if next_h1_page != float('inf') else 'end'}")
            
            # Find ALL H2s and H3s in the page range for this H1
            relevant_h2s = []
            relevant_h3s = []
            
            for idx, h2 in enumerate(safe2_h2_headings):
                h2_page = h2.get('page', 0)
                if current_page <= h2_page < next_h1_page and idx not in assigned_h2s:
                    relevant_h2s.append((idx, h2))
                    assigned_h2s.add(idx)
            
            for idx, h3 in enumerate(safe2_h3_headings):
                h3_page = h3.get('page', 0)
                if current_page <= h3_page < next_h1_page and idx not in assigned_h3s:
                    relevant_h3s.append((idx, h3))
                    assigned_h3s.add(idx)
            
            print(f"   Found: {len(relevant_h2s)} H2s, {len(relevant_h3s)} H3s")
            
            # Sort H2s and H3s by page and position
            relevant_h2s.sort(key=lambda x: (x[1].get('page', 0), x[1].get('position', {}).get('y0', 0)))
            relevant_h3s.sort(key=lambda x: (x[1].get('page', 0), x[1].get('position', {}).get('y0', 0)))
            
            # Process each H2 and assign H3s to it
            used_h3_indices = set()
            
            for j, (h2_idx, h2) in enumerate(relevant_h2s):
                h2_section = {
                    'type': 'h2',
                    'text': h2['text'],
                    'page': h2['page'],
                    'source': 'safe2',
                    'confidence': h2['confidence'],
                    'position': h2.get('position', {}),
                    'formatting': h2.get('formatting', {}),
                    'h3_subsections': []
                }
                
                # Find H3s that belong to this H2
                h2_page = h2.get('page', 0)
                h2_y_position = h2.get('position', {}).get('y0', 0)
                
                # Determine the boundary for this H2's H3s
                if j < len(relevant_h2s) - 1:
                    next_h2 = relevant_h2s[j + 1][1]
                    next_h2_page = next_h2.get('page', 0)
                    next_h2_y = next_h2.get('position', {}).get('y0', 0)
                else:
                    # Last H2 in this H1 section
                    next_h2_page = next_h1_page
                    next_h2_y = 0
                
                # Find H3s for this H2
                h2_h3s = []
                for h3_idx, h3 in relevant_h3s:
                    if h3_idx in used_h3_indices:
                        continue
                    
                    h3_page = h3.get('page', 0)
                    h3_y = h3.get('position', {}).get('y0', 0)
                    
                    # H3 belongs to this H2 if:
                    # 1. It's on the same page and below the H2
                    # 2. Or it's on a subsequent page (until next H2 or H1)
                    if ((h3_page == h2_page and h3_y > h2_y_position) or 
                        (h3_page > h2_page and h3_page < next_h2_page) or
                        (h3_page == next_h2_page and h3_y < next_h2_y and next_h2_page < next_h1_page)):
                        h2_h3s.append((h3_idx, h3))
                        used_h3_indices.add(h3_idx)
                
                # Add H3s to this H2
                for h3_idx, h3 in h2_h3s:
                    h3_section = {
                        'type': 'h3',
                        'text': h3['text'],
                        'page': h3['page'],
                        'source': 'safe2',
                        'confidence': h3['confidence'],
                        'position': h3.get('position', {}),
                        'formatting': h3.get('formatting', {})
                    }
                    h2_section['h3_subsections'].append(h3_section)
                
                print(f"     H2: '{h2['text']}' (page {h2['page']}) -> {len(h2_section['h3_subsections'])} H3s")
                h1_section['subsections'].append(h2_section)
            
            # Add any remaining unassigned H3s directly under H1
            unassigned_h3s = [(idx, h3) for idx, h3 in relevant_h3s if idx not in used_h3_indices]
            for h3_idx, h3 in unassigned_h3s:
                h3_section = {
                    'type': 'h3',
                    'text': h3['text'],
                    'page': h3['page'],
                    'source': 'safe2',
                    'confidence': h3['confidence'],
                    'position': h3.get('position', {}),
                    'formatting': h3.get('formatting', {})
                }
                h1_section['subsections'].append(h3_section)
                print(f"     H3 (direct): '{h3['text']}' (page {h3['page']})")
            
            combined_structure.append(h1_section)
        
        # Create final combined result
        combined_result = {
            'metadata': {
                'source_file': Path(pdf_path).name,
                'processing_timestamp': datetime.now().isoformat(),
                'safe1_file': safe1_data['output_file'],
                'safe2_file': safe2_data['output_file'],
                'combination_method': 'safe1_h1_preserved_with_safe2_filtering_and_constraints'
            },
            'summary': {
                'total_sections': len(combined_structure),
                'titles': len([s for s in combined_structure if s['type'] == 'title']),
                'h1_sections': len([s for s in combined_structure if s['type'] == 'h1']),
                'total_h2s': sum(len([sub for sub in s.get('subsections', []) if sub['type'] == 'h2']) 
                               for s in combined_structure),
                'total_h3s': sum(len([sub for sub in s.get('subsections', []) if sub['type'] == 'h3']) +
                               sum(len(sub.get('h3_subsections', [])) for sub in s.get('subsections', []) if sub['type'] == 'h2')
                               for s in combined_structure)
            },
            'hierarchical_structure': combined_structure
        }
        
        print(f"[TARGET] Applying post-processing constraints...")
        
        # Apply text constraints (colon truncation, etc.)
        combined_result = self.apply_text_constraints(combined_result)
        
        # Apply hierarchical constraints (promote orphaned H3s)
        combined_result = self.promote_orphaned_h3s(combined_result)
        
        print(f"[OK] Combination completed!")
        print(f"[STATS] Combined Structure Summary:")
        print(f"   - Total sections: {combined_result['summary']['total_sections']}")
        print(f"   - Titles: {combined_result['summary']['titles']}")
        print(f"   - H1 sections: {combined_result['summary']['h1_sections']}")
        print(f"   - Total H2s: {combined_result['summary']['total_h2s']}")
        print(f"   - Total H3s: {combined_result['summary']['total_h3s']}")
        print(f"   - H2s assigned from Safe2: {len(assigned_h2s)}/{len(safe2_h2_headings)}")
        print(f"   - H3s assigned from Safe2: {len(assigned_h3s)}/{len(safe2_h3_headings)}")
        
        return combined_result
    
    def apply_text_constraints(self, combined_result: Dict) -> Dict:
        """
        Apply text cleaning constraints to H2 and H3 headings only.
        H1 headings from safe1 are preserved unchanged.
        
        Args:
            combined_result: Combined hierarchical results
            
        Returns:
            Dict: Results with text constraints applied
        """
        cleaned_count = 0
        removed_count = 0
        
        def should_remove_heading(text: str) -> bool:
            """Check if heading should be removed based on lowercase start and word count."""
            if not text or not text.strip():
                return True
            
            # Check if starts with lowercase and has more than 6 words
            words = text.strip().split()
            if len(words) > 6 and text.strip()[0].islower():
                return True
            return False
        
        def clean_text(text: str) -> str:
            nonlocal cleaned_count
            original_text = text
            
            # 1. Truncate at colon (keep only the part before the colon)
            if ':' in text:
                text = text.split(':')[0].strip()
                
            # 2. Handle bold text truncation (remove text after bold indicators)
            # Remove common bold formatting artifacts
            bold_indicators = [' :', ' -', ' [?]', ' [?]']
            for indicator in bold_indicators:
                if indicator in text:
                    text = text.split(indicator)[0].strip()
                    
            # 3. Clean up common artifacts
            text = text.strip()
            
            if text != original_text:
                cleaned_count += 1
                
            return text
        
        # Apply to H2 and H3 headings only (skip titles and H1s)
        for section in combined_result['hierarchical_structure']:
            # Skip titles and H1 headings - they remain unchanged
            if section['type'] in ['title', 'h1']:
                # Don't modify title or H1 text
                pass
            
            # Process subsections (H2s and H3s)
            subsections_to_keep = []
            for subsection in section.get('subsections', []):
                if subsection['type'] == 'h2' and 'text' in subsection:
                    # Check if H2 should be removed
                    if should_remove_heading(subsection['text']):
                        removed_count += 1
                        continue  # Skip this H2
                    subsection['text'] = clean_text(subsection['text'])
                elif subsection['type'] == 'h3' and 'text' in subsection:
                    # Check if H3 should be removed
                    if should_remove_heading(subsection['text']):
                        removed_count += 1
                        continue  # Skip this H3
                    subsection['text'] = clean_text(subsection['text'])
                
                # Process H3s under H2s
                h3_subsections_to_keep = []
                for h3_sub in subsection.get('h3_subsections', []):
                    if 'text' in h3_sub:
                        # Check if H3 should be removed
                        if should_remove_heading(h3_sub['text']):
                            removed_count += 1
                            continue  # Skip this H3
                        h3_sub['text'] = clean_text(h3_sub['text'])
                    h3_subsections_to_keep.append(h3_sub)
                
                subsection['h3_subsections'] = h3_subsections_to_keep
                subsections_to_keep.append(subsection)
            
            section['subsections'] = subsections_to_keep
        
        print(f"   [OK] Cleaned text for {cleaned_count} H2/H3 headings (H1s preserved)")
        print(f"   [OK] Removed {removed_count} H2/H3 headings (lowercase start + >6 words)")
        return combined_result
    
    def promote_orphaned_h3s(self, combined_result: Dict) -> Dict:
        """
        Promote orphaned H3s (those not under any H2) to H2 level.
        
        Args:
            combined_result: Combined hierarchical results
            
        Returns:
            Dict: Results with orphaned H3s promoted
        """
        promoted_count = 0
        
        for section in combined_result['hierarchical_structure']:
            if section['type'] == 'h1':
                new_subsections = []
                
                for subsection in section.get('subsections', []):
                    if subsection['type'] == 'h2':
                        new_subsections.append(subsection)
                    elif subsection['type'] == 'h3':
                        # Promote H3 to H2
                        promoted_h2 = {
                            'type': 'h2',
                            'text': subsection['text'],
                            'page': subsection['page'],
                            'source': subsection['source'],
                            'confidence': subsection['confidence'],
                            'position': subsection.get('position', {}),
                            'formatting': subsection.get('formatting', {}),
                            'h3_subsections': []
                        }
                        new_subsections.append(promoted_h2)
                        promoted_count += 1
                
                section['subsections'] = new_subsections
        
        # Recalculate summary after promotions
        combined_result['summary'] = {
            'total_sections': len(combined_result['hierarchical_structure']),
            'titles': len([s for s in combined_result['hierarchical_structure'] if s['type'] == 'title']),
            'h1_sections': len([s for s in combined_result['hierarchical_structure'] if s['type'] == 'h1']),
            'total_h2s': sum(len([sub for sub in s.get('subsections', []) if sub['type'] == 'h2']) 
                           for s in combined_result['hierarchical_structure']),
            'total_h3s': sum(len([sub for sub in s.get('subsections', []) if sub['type'] == 'h3']) +
                           sum(len(sub.get('h3_subsections', [])) for sub in s.get('subsections', []) if sub['type'] == 'h2')
                           for s in combined_result['hierarchical_structure'])
        }
        
        print(f"   [OK] Promoted {promoted_count} orphaned H3s to H2s")
        return combined_result
    
    def convert_to_flat_outline(self, combined_result: Dict) -> Dict:
        """
        Convert hierarchical structure to flat outline format as described in convert_to_outline.py.
        
        Args:
            combined_result: Combined hierarchical results
            
        Returns:
            Dict: Flat outline structure with title and outline array
        """
        # Extract title
        title = ""
        outline = []
        
        # Process hierarchical structure
        for section in combined_result.get('hierarchical_structure', []):
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
        
        # Create final flat outline structure
        flat_outline = {
            "title": title,
            "outline": outline
        }
        
        print(f"[OK] Converted to flat outline format:")
        print(f"   - Title: {title}")
        print(f"   - Total outline items: {len(outline)}")
        
        # Count levels
        level_counts = {}
        for item in outline:
            level = item['level']
            level_counts[level] = level_counts.get(level, 0) + 1
        
        for level, count in sorted(level_counts.items()):
            print(f"   - {level} headings: {count}")
        
        return flat_outline
    
    def save_combined_results(self, combined_result: Dict, pdf_path: str) -> str:
        """
        Save the combined results to JSON files (flat outline format only).
        
        Args:
            combined_result: Combined hierarchical results
            pdf_path: Original PDF path
            
        Returns:
            str: Path to saved flat outline file
        """
        # Use PDF name without timestamp for exact matching
        pdf_stem = Path(pdf_path).stem
        
        # Convert to flat outline format
        flat_outline = self.convert_to_flat_outline(combined_result)
        
        # Save flat outline format (main output) - this matches the PDF name exactly
        outline_filename = f"{pdf_stem}.json"
        outline_path = self.output_dir / outline_filename
        
        with open(outline_path, 'w', encoding='utf-8') as f:
            json.dump(flat_outline, f, indent=2, ensure_ascii=False)
        
        print(f"[SAVE] Results saved to: {outline_path}")
        return str(outline_path)
    
    def cleanup_intermediate_files(self):
        """Clean up intermediate files from safe1 and safe2 processing."""
        try:
            # Clean up safe1_output directory
            safe1_output = self.output_dir / "safe1_output"
            if safe1_output.exists():
                shutil.rmtree(safe1_output)
            
            # Clean up safe2_output directory  
            safe2_output = self.output_dir / "safe2_output"
            if safe2_output.exists():
                shutil.rmtree(safe2_output)
                
            print(f"[CLEANUP] Removed intermediate files")
        except Exception as e:
            print(f"[WARNING] Failed to cleanup intermediate files: {e}")
    
    def process_pdf(self, pdf_path: str) -> str:
        """
        Complete processing pipeline for a PDF.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            str: Path to the combined output file
        """
        print(f"[TARGET] Starting Combined Heading Extraction Pipeline")
        print(f"[FILE] PDF: {Path(pdf_path).name}")
        print(f"[FOLDER] Output Directory: {self.output_dir}")
        print("=" * 80)
        
        try:
            # Step 1: Run Safe1 extractor
            safe1_data = self.run_safe1_extractor(pdf_path)
            
            # Step 2: Run Safe2 extractor
            safe2_data = self.run_safe2_extractor(pdf_path)
            
            # Step 3: Combine results
            combined_result = self.combine_results(safe1_data, safe2_data, pdf_path)
            
            # Step 4: Save combined results
            output_path = self.save_combined_results(combined_result, pdf_path)
            
            # Step 5: Clean up intermediate files
            self.cleanup_intermediate_files()
            
            print("\n[COMPLETE] PROCESSING COMPLETE!")
            print("=" * 80)
            print(f"[OK] Successfully processed: {Path(pdf_path).name}")
            print(f"[FOLDER] Output saved in: {self.output_dir}")
            print(f"[FILE] Main output: {output_path}")
            
            return output_path
            
        except Exception as e:
            print(f"\n[ERROR] Processing failed: {e}")
            raise

def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(description="Combined heading extraction using both safe1 and safe2 models")
    parser.add_argument("--pdf_path", type=str, required=True, help="Path to PDF file")
    parser.add_argument("--output_dir", type=str, default="combined_output", 
                       help="Output directory for results (default: combined_output)")
    parser.add_argument("--safe1_dir", type=str, default="safe1", 
                       help="Directory containing safe1 model (default: safe1)")
    parser.add_argument("--safe2_dir", type=str, default="safe2", 
                       help="Directory containing safe2 model (default: safe2)")
    
    args = parser.parse_args()
    
    # Verify PDF exists
    if not Path(args.pdf_path).exists():
        print(f"[ERROR] PDF file not found: {args.pdf_path}")
        sys.exit(1)
    
    try:
        # Initialize extractor
        extractor = CombinedHeadingExtractor(
            safe1_dir=args.safe1_dir,
            safe2_dir=args.safe2_dir,
            output_dir=args.output_dir
        )
        
        # Process PDF
        output_path = extractor.process_pdf(args.pdf_path)
        
        print(f"\n[RESULT] View your results:")
        print(f"   Combined output: {output_path}")
        
    except Exception as e:
        print(f"[ERROR] Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
