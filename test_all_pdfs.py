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
    test_folder = Path("test")
    output_base = Path("test_output")
    
    # Verify test folder exists
    if not test_folder.exists():
        print(f"❌ Test folder not found: {test_folder}")
        return
    
    # Find all PDF files in test folder
    pdf_files = list(test_folder.glob("*.pdf"))
    
    if not pdf_files:
        print(f"❌ No PDF files found in {test_folder}")
        return
    
    print(f"🎯 BATCH PROCESSING ALL TEST PDFs")
    print(f"📁 Test folder: {test_folder}")
    print(f"📄 Found {len(pdf_files)} PDF files:")
    for pdf in pdf_files:
        print(f"   • {pdf.name}")
    print("=" * 80)
    
    # Create base output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_output_dir = output_base / f"batch_test_{timestamp}"
    batch_output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Batch output directory: {batch_output_dir}")
    print()
    
    # Process each PDF
    results = []
    successful = 0
    failed = 0
    
    for i, pdf_file in enumerate(pdf_files, 1):
        print(f"🚀 PROCESSING PDF {i}/{len(pdf_files)}: {pdf_file.name}")
        print("=" * 60)
        
        try:
            # Create individual output directory for this PDF
            pdf_output_dir = batch_output_dir / f"{pdf_file.stem}_output"
            
            # Initialize extractor for this PDF
            extractor = CombinedHeadingExtractor(
                safe1_dir="safe1",
                safe2_dir="safe2",
                output_dir=str(pdf_output_dir)
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
            
            print(f"✅ Successfully processed: {pdf_file.name}")
            print(f"📄 Output: {output_path}")
            
        except Exception as e:
            results.append({
                'pdf': pdf_file.name,
                'status': 'FAILED',
                'output_path': None,
                'error': str(e)
            })
            failed += 1
            
            print(f"❌ Failed to process: {pdf_file.name}")
            print(f"   Error: {e}")
        
        print()
    
    # Print summary
    print("🎉 BATCH PROCESSING COMPLETE!")
    print("=" * 80)
    print(f"📊 SUMMARY:")
    print(f"   • Total PDFs: {len(pdf_files)}")
    print(f"   • Successful: {successful}")
    print(f"   • Failed: {failed}")
    print(f"   • Success rate: {(successful/len(pdf_files)*100):.1f}%")
    print()
    
    # Print detailed results
    print(f"📋 DETAILED RESULTS:")
    for result in results:
        status_icon = "✅" if result['status'] == 'SUCCESS' else "❌"
        print(f"   {status_icon} {result['pdf']}: {result['status']}")
        if result['status'] == 'SUCCESS':
            print(f"      📄 Output: {result['output_path']}")
        else:
            print(f"      ❌ Error: {result['error']}")
    
    print()
    print(f"📁 All outputs saved in: {batch_output_dir}")
    
    # Generate a summary report
    generate_summary_report(results, batch_output_dir)
    
    return results

def generate_summary_report(results, output_dir):
    """Generate a summary report of the batch processing."""
    
    report_file = output_dir / "batch_processing_summary.txt"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("BATCH PDF PROCESSING SUMMARY\n")
        f.write("=" * 50 + "\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"Total PDFs processed: {len(results)}\n")
        
        successful = sum(1 for r in results if r['status'] == 'SUCCESS')
        failed = sum(1 for r in results if r['status'] == 'FAILED')
        
        f.write(f"Successful: {successful}\n")
        f.write(f"Failed: {failed}\n")
        f.write(f"Success rate: {(successful/len(results)*100):.1f}%\n")
        f.write("\n")
        
        f.write("DETAILED RESULTS:\n")
        f.write("-" * 30 + "\n")
        
        for result in results:
            f.write(f"\nPDF: {result['pdf']}\n")
            f.write(f"Status: {result['status']}\n")
            if result['status'] == 'SUCCESS':
                f.write(f"Output: {result['output_path']}\n")
            else:
                f.write(f"Error: {result['error']}\n")
    
    print(f"📄 Summary report saved: {report_file}")

def view_all_results(batch_output_dir=None):
    """View results for all processed PDFs using the viewer."""
    
    if batch_output_dir is None:
        # Find the most recent batch output directory
        output_base = Path("test_output")
        if not output_base.exists():
            print("❌ No test output directory found")
            return
        
        batch_dirs = [d for d in output_base.iterdir() if d.is_dir() and d.name.startswith("batch_test_")]
        if not batch_dirs:
            print("❌ No batch test results found")
            return
        
        batch_output_dir = max(batch_dirs, key=lambda x: x.stat().st_mtime)
        print(f"📁 Using most recent batch: {batch_output_dir}")
    
    # Find all combined JSON files
    json_files = []
    for pdf_dir in batch_output_dir.iterdir():
        if pdf_dir.is_dir():
            combined_files = list(pdf_dir.glob("combined_*.json"))
            json_files.extend(combined_files)
    
    if not json_files:
        print("❌ No combined JSON files found")
        return
    
    print(f"🏛️  VIEWING ALL BATCH RESULTS")
    print(f"📁 Batch directory: {batch_output_dir}")
    print(f"📄 Found {len(json_files)} result files")
    print("=" * 80)
    
    # Import the viewer function
    sys.path.append('.')
    from view_final_results import display_hierarchical_structure
    
    for i, json_file in enumerate(json_files, 1):
        print(f"\n📋 RESULT {i}/{len(json_files)}: {json_file.parent.name}")
        print("=" * 60)
        try:
            display_hierarchical_structure(str(json_file))
        except Exception as e:
            print(f"❌ Error viewing {json_file}: {e}")
        
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
            response = input("\n🔍 Would you like to view the results? (y/n): ").lower().strip()
            if response in ['y', 'yes']:
                # Find the most recent batch directory
                output_base = Path("test_output")
                batch_dirs = [d for d in output_base.iterdir() if d.is_dir() and d.name.startswith("batch_test_")]
                if batch_dirs:
                    latest_batch = max(batch_dirs, key=lambda x: x.stat().st_mtime)
                    view_all_results(latest_batch)
        except KeyboardInterrupt:
            print("\n👋 Exiting...")
