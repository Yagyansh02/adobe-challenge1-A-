#!/usr/bin/env python3
"""
Docker Entrypoint for PDF Heading Extraction
============================================

This script processes all PDF files in /app/input and outputs JSON results to /app/output.
Designed to work completely offline with embedded models.
"""

import os
import sys
from pathlib import Path
from combined_heading_extractor import CombinedHeadingExtractor

def main():
    """Main entrypoint for Docker container."""
    
    # Define paths
    input_dir = Path("/app/input")
    output_dir = Path("/app/output")
    
    print("[DOCKER] Starting PDF Heading Extraction Service")
    print("=" * 60)
    print(f"[INPUT] Input directory: {input_dir}")
    print(f"[OUTPUT] Output directory: {output_dir}")
    
    # Verify input directory exists
    if not input_dir.exists():
        print(f"[ERROR] Input directory not found: {input_dir}")
        print("[INFO] Please mount input directory with: -v $(pwd)/input:/app/input")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all PDF files
    pdf_files = list(input_dir.glob("*.pdf"))
    
    if not pdf_files:
        print(f"[WARNING] No PDF files found in {input_dir}")
        print("[INFO] Please add PDF files to the input directory")
        return
    
    print(f"[FOUND] {len(pdf_files)} PDF file(s) to process:")
    for pdf in pdf_files:
        print(f"   - {pdf.name}")
    print()
    
    # Initialize extractor
    try:
        extractor = CombinedHeadingExtractor(
            safe1_dir="safe1",
            safe2_dir="safe2", 
            output_dir=str(output_dir)
        )
        print("[INIT] Combined heading extractor initialized successfully")
    except Exception as e:
        print(f"[ERROR] Failed to initialize extractor: {e}")
        sys.exit(1)
    
    # Process each PDF
    successful = 0
    failed = 0
    
    for i, pdf_file in enumerate(pdf_files, 1):
        print(f"\n[PROCESS] Processing {i}/{len(pdf_files)}: {pdf_file.name}")
        print("-" * 40)
        
        try:
            # Process the PDF
            output_path = extractor.process_pdf(str(pdf_file))
            
            # Verify output was created
            if Path(output_path).exists():
                print(f"[SUCCESS] Output created: {Path(output_path).name}")
                successful += 1
            else:
                print(f"[ERROR] Output file not created: {output_path}")
                failed += 1
                
        except Exception as e:
            print(f"[ERROR] Failed to process {pdf_file.name}: {e}")
            failed += 1
    
    # Print final summary
    print("\n" + "=" * 60)
    print("[COMPLETE] Processing finished")
    print(f"[STATS] Total: {len(pdf_files)} | Success: {successful} | Failed: {failed}")
    
    if successful > 0:
        print(f"[RESULTS] Check {output_dir} for JSON output files")
    
    # Exit with error code if any failed
    if failed > 0:
        sys.exit(1)

if __name__ == "__main__":
    main()
