#!/usr/bin/env python3
"""
Test All PDFs in Test Folder
============================

This script processes all PDF files in the test folder using the combined heading extractor.
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from combined_heading_extractor import CombinedHeadingExtractor

def test_all_pdfs():
    """Process all PDFs in the test folder."""
    
    # Setup paths
    test_folder = Path("input")
    output_base = Path("output")
    
    # Verify test folder exists
    if not test_folder.exists():
        print(f"[ERROR] Test folder not found: {test_folder}")
        return
    
    # Create output directory if it doesn't exist
    output_base.mkdir(parents=True, exist_ok=True)
    
    # Find all PDF files in test folder
    pdf_files = list(test_folder.glob("*.pdf"))
    
    if not pdf_files:
        print(f"[ERROR] No PDF files found in {test_folder}")
        return
    
    print(f"[TARGET] BATCH PROCESSING ALL TEST PDFs")
    print(f"[FOLDER] Test folder: {test_folder}")
    print(f"[FOLDER] Output folder: {output_base}")
    print(f"[FILE] Found {len(pdf_files)} PDF files:")
    for pdf in pdf_files:
        print(f"   [?] {pdf.name}")
    print("=" * 80)
    
    print("=" * 80)
    
    # Process each PDF
    results = []
    successful = 0
    failed = 0
    
    for i, pdf_file in enumerate(pdf_files, 1):
        print(f"[START] PROCESSING PDF {i}/{len(pdf_files)}: {pdf_file.name}")
        print("=" * 60)
        
        try:
            # Initialize extractor with output directly to the output folder
            # The CombinedHeadingExtractor will create a JSON file with matching PDF name
            extractor = CombinedHeadingExtractor(
                safe1_dir="safe1",
                safe2_dir="safe2",
                output_dir=str(output_base)
            )
            
            # Process the PDF
            output_path = extractor.process_pdf(str(pdf_file))
            
            results.append({
                'pdf': pdf_file.name,
                'status': 'SUCCESS',
                'output_path': output_path,
                'error': None
            })
            successful += 1
            
            print(f"[OK] Successfully processed: {pdf_file.name}")
            print(f"[FILE] Output: {output_path}")
            
        except Exception as e:
            results.append({
                'pdf': pdf_file.name,
                'status': 'FAILED',
                'output_path': None,
                'error': str(e)
            })
            failed += 1
            
            print(f"[ERROR] Failed to process: {pdf_file.name}")
            print(f"   Error: {e}")
        
        print()
    
    # Print summary
    print("[COMPLETE] BATCH PROCESSING COMPLETE!")
    print("=" * 80)
    print(f"[STATS] SUMMARY:")
    print(f"   [?] Total PDFs: {len(pdf_files)}")
    print(f"   [?] Successful: {successful}")
    print(f"   [?] Failed: {failed}")
    print(f"   [?] Success rate: {(successful/len(pdf_files)*100):.1f}%")
    print()
    
    # Print detailed results
    print(f"[RESULT] DETAILED RESULTS:")
    for result in results:
        status_icon = "[OK]" if result['status'] == 'SUCCESS' else "[ERROR]"
        print(f"   {status_icon} {result['pdf']}: {result['status']}")
        if result['status'] == 'SUCCESS':
            print(f"      [FILE] Output: {result['output_path']}")
        else:
            print(f"      [ERROR] Error: {result['error']}")
    
    print()
    print(f"[FOLDER] All outputs saved in: {output_base}")
    
    return results

def view_all_results(output_dir=None):
    """View results for all processed PDFs using the viewer."""
    
    if output_dir is None:
        # Use the main output directory
        output_dir = Path("output")
        if not output_dir.exists():
            print("[ERROR] No output directory found")
            return
    
    # Find all main JSON files in the output directory (exclude hierarchical, summary, and safe outputs)
    json_files = list(output_dir.glob("*.json"))
    # Filter to include only the main result files (exclude hierarchical, batch processing, and safe outputs)
    json_files = [f for f in json_files if not f.name.startswith("combined_hierarchical_") 
                  and not f.name.startswith("batch_processing_") 
                  and not f.name.startswith("safe1_")
                  and not f.name.startswith("safe2_")]
    
    if not json_files:
        print("[ERROR] No main JSON files found in output directory")
        return
    
    print(f"[BUILDING] VIEWING ALL RESULTS")
    print(f"[FOLDER] Output directory: {output_dir}")
    print(f"[FILE] Found {len(json_files)} result files")
    print("=" * 80)
    
    # Import the viewer function
    sys.path.append('.')
    from view_final_results import display_hierarchical_structure
    
    for i, json_file in enumerate(json_files, 1):
        print(f"\n[RESULT] RESULT {i}/{len(json_files)}: {json_file.name}")
        print("=" * 60)
        try:
            display_hierarchical_structure(str(json_file))
        except Exception as e:
            print(f"[ERROR] Error viewing {json_file}: {e}")
        
        if i < len(json_files):
            input("\nPress Enter to continue to next result...")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test all PDFs in the test folder")
    parser.add_argument("--view-only", action="store_true", 
                       help="Only view existing results, don't process")
    
    args = parser.parse_args()
    
    if args.view_only:
        view_all_results()
    else:
        results = test_all_pdfs()
        
        # Ask if user wants to view results
        try:
            response = input("\n[SEARCH] Would you like to view the results? (y/n): ").lower().strip()
            if response in ['y', 'yes']:
                view_all_results()
        except KeyboardInterrupt:
            print("\n[EXIT] Exiting...")
