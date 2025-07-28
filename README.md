# Combined Heading Extraction System

This system combines outputs from two different PDF heading extraction models to create a hierarchical document structure.

## Overview

The system uses two models:
- **Safe1**: Extracts titles and H1 headings with high accuracy
- **Safe2**: Extracts titles, H1, H2, and H3 headings

The combination process:
1. Takes titles and H1 headings from Safe1 output
2. Takes H2 and H3 headings from Safe2 output  
3. Organizes H2/H3 hierarchically under their corresponding H1 sections based on page position

## Files

### Main Scripts
- `combined_heading_extractor.py` - Main script that orchestrates both models and combines outputs
- `view_combined_results.py` - Viewer script to analyze and display the hierarchical structure

### Model Directories
- `safe1/` - Contains Safe1 model and extractor (focuses on titles and H1)
- `safe2/` - Contains Safe2 model and extractor (extracts all heading levels)

### Output Structure
```
combined_output/
├── safe1_output/           # Safe1 extraction results
├── safe2_output/           # Safe2 extraction results
└── combined_*.json         # Final combined hierarchical results
```

## Usage

### Running the Combined Extractor

```bash
# Basic usage
python combined_heading_extractor.py --pdf_path "document.pdf"

# With custom output directory
python combined_heading_extractor.py --pdf_path "document.pdf" --output_dir "my_output"

# With custom model directories
python combined_heading_extractor.py --pdf_path "document.pdf" --safe1_dir "path/to/safe1" --safe2_dir "path/to/safe2"
```

### Viewing Results

```bash
# View the hierarchical structure
python view_combined_results.py --json_path "combined_output/combined_document_20250728_180117.json"
```

## Output Format

The combined output JSON contains:

```json
{
  "metadata": {
    "source_file": "document.pdf",
    "processing_timestamp": "2025-07-28T18:01:17.589080",
    "safe1_file": "path/to/safe1/output.json",
    "safe2_file": "path/to/safe2/output.json",
    "combination_method": "hierarchical_by_page_position"
  },
  "summary": {
    "total_sections": 14,
    "titles": 1,
    "h1_sections": 13,
    "total_h2s": 30,
    "total_h3s": 80
  },
  "hierarchical_structure": [
    {
      "type": "title",
      "text": "Document Title",
      "page": 1,
      "source": "safe1",
      "confidence": 0.779,
      "sections": []
    },
    {
      "type": "h1",
      "text": "Chapter 1",
      "page": 2,
      "source": "safe1",
      "confidence": 0.876,
      "bbox": {...},
      "subsections": [
        {
          "type": "h2",
          "text": "Section 1.1",
          "page": 2,
          "source": "safe2",
          "confidence": 0.705,
          "position": {...},
          "h3_subsections": [
            {
              "type": "h3",
              "text": "Subsection 1.1.1",
              "page": 2,
              "source": "safe2",
              "confidence": 0.862,
              "position": {...}
            }
          ]
        }
      ]
    }
  ]
}
```

## Hierarchical Organization Logic

The system organizes headings hierarchically using the following logic:

1. **Title**: Extracted from Safe1, appears at the top level
2. **H1 Sections**: Extracted from Safe1, form the main document structure
3. **H2 Placement**: H2s from Safe2 are placed under the H1 that appears on the same page or the closest preceding H1
4. **H3 Placement**: H3s from Safe2 are placed under their corresponding H2 based on page position and text flow

### Page-Based Assignment

- H2s and H3s are assigned to H1 sections based on page ranges
- If an H1 is on page 3 and the next H1 is on page 7, all H2s and H3s on pages 3-6 are assigned to the first H1
- Within each H1 section, H3s are grouped under their corresponding H2s based on page position

## Example Results

For the "South of France - Cities" PDF:

- **Safe1 found**: 1 title, 13 H1 headings
- **Safe2 found**: 1 title, 11 H1s, 30 H2s, 80 H3s  
- **Combined result**: Hierarchical structure with 13 H1 sections containing 30 H2s and 80 H3s organized logically

## Statistics and Analysis

The viewer script provides:
- Summary statistics (total headings by type)
- Page distribution analysis  
- Confidence score analysis
- Full hierarchical tree visualization

## Requirements

- Python 3.7+
- Dependencies for Safe1: scikit-learn, xgboost, pandas, numpy, PyMuPDF
- Dependencies for Safe2: xgboost, scikit-learn, PyMuPDF, pdfplumber

## Error Handling

The system includes robust error handling:
- Falls back to existing outputs if model execution fails
- Provides detailed progress information
- Handles missing files and invalid paths gracefully

## Future Enhancements

Potential improvements:
- Support for more heading levels (H4, H5, H6)
- Advanced text position analysis for better hierarchy detection
- Integration of multiple model outputs with confidence weighting
- Batch processing of multiple PDFs
