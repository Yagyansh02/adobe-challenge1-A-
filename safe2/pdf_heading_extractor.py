#!/usr/bin/env python3
"""
PDF Heading Extractor
A complete solution for extracting headings from PDF files using trained XGBoost model.

This script:
1. Extracts text with metadata from PDF files
2. Uses trained XGBoost model to classify headings (title, h1, h2, h3)
3. Outputs results as JSON file with hierarchical structure

Usage:
    python pdf_heading_extractor.py --pdf_path "document.pdf" --output_json "headings.json"
    python pdf_heading_extractor.py --pdf_path "document.pdf" --confidence_threshold 0.7
"""

import os
import sys
import argparse
import json
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# PDF processing libraries
try:
    import pymupdf as fitz  # PyMuPDF
    import pdfplumber
    PDF_LIBRARIES_AVAILABLE = True
except ImportError:
    PDF_LIBRARIES_AVAILABLE = False
    print("⚠️  PDF libraries not found. Install with: pip install pymupdf pdfplumber")

# ML libraries
try:
    import joblib
    import xgboost as xgb
    from scipy import sparse
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import LabelEncoder
    ML_LIBRARIES_AVAILABLE = True
except ImportError:
    ML_LIBRARIES_AVAILABLE = False
    print("⚠️  ML libraries not found. Install with: pip install xgboost scikit-learn scipy joblib")

import re
from datetime import datetime

class PDFHeadingExtractor:
    def __init__(self, models_dir='models'):
        """
        Initialize the PDF heading extractor.
        
        Args:
            models_dir (str): Directory containing trained model files
        """
        self.models_dir = models_dir
        self.model = None
        self.tfidf = None
        self.label_encoder = None
        self.metadata_features = []
        self.model_config = {}
        
        # Load trained models
        self.load_models()
    
    def load_models(self):
        """Load all trained model components."""
        try:
            print("🔄 Loading trained models...")
            
            # Load XGBoost model
            model_path = os.path.join(self.models_dir, 'pdf_heading_classifier.json')
            if os.path.exists(model_path):
                self.model = xgb.XGBClassifier()
                self.model.load_model(model_path)
                print(f"✅ Loaded XGBoost model from {model_path}")
            else:
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            # Load TF-IDF vectorizer
            tfidf_path = os.path.join(self.models_dir, 'tfidf_vectorizer.pkl')
            if os.path.exists(tfidf_path):
                self.tfidf = joblib.load(tfidf_path)
                print(f"✅ Loaded TF-IDF vectorizer from {tfidf_path}")
            else:
                raise FileNotFoundError(f"TF-IDF vectorizer not found: {tfidf_path}")
            
            # Load label encoder
            encoder_path = os.path.join(self.models_dir, 'label_encoder.pkl')
            if os.path.exists(encoder_path):
                self.label_encoder = joblib.load(encoder_path)
                print(f"✅ Loaded label encoder from {encoder_path}")
            else:
                raise FileNotFoundError(f"Label encoder not found: {encoder_path}")
            
            # Load metadata features list
            features_path = os.path.join(self.models_dir, 'metadata_features.txt')
            if os.path.exists(features_path):
                with open(features_path, 'r') as f:
                    self.metadata_features = [line.strip() for line in f if line.strip()]
                print(f"✅ Loaded metadata features from {features_path}")
            else:
                print(f"⚠️  Metadata features file not found, using defaults")
                self.metadata_features = [
                    'font_size', 'text_length', 'word_count', 'avg_word_length',
                    'is_bold', 'is_italic', 'is_uppercase', 'is_title_case', 
                    'has_numbers', 'bbox_width', 'bbox_height', 'bbox_area'
                ]
            
            # Load model configuration
            config_path = os.path.join(self.models_dir, 'model_config.json')
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    self.model_config = json.load(f)
                print(f"✅ Loaded model configuration from {config_path}")
            
            print(f"📊 Model Info:")
            print(f"   Classes: {list(self.label_encoder.classes_)}")
            if self.model_config:
                print(f"   Training accuracy: {self.model_config.get('accuracy', 'N/A')}")
                print(f"   Weighted F1: {self.model_config.get('weighted_f1', 'N/A')}")
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            sys.exit(1)
    
    def extract_text_from_pdf(self, pdf_path, method='pymupdf'):
        """
        Extract text with metadata from PDF.
        
        Args:
            pdf_path (str): Path to PDF file
            method (str): 'pymupdf' or 'pdfplumber'
            
        Returns:
            pd.DataFrame: DataFrame with extracted text and metadata
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        print(f"📖 Extracting text from: {os.path.basename(pdf_path)}")
        
        if method == 'pymupdf':
            return self._extract_with_pymupdf(pdf_path)
        elif method == 'pdfplumber':
            return self._extract_with_pdfplumber(pdf_path)
        else:
            raise ValueError("Method must be 'pymupdf' or 'pdfplumber'")
    
    def _extract_with_pymupdf(self, pdf_path):
        """Extract text using PyMuPDF (better for font information)."""
        extracted_data = []
        
        try:
            doc = fitz.open(pdf_path)
            print(f"📄 Total pages: {len(doc)}")
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                print(f"  Processing page {page_num + 1}/{len(doc)}...", end='\r')
                
                # Get text blocks with formatting information
                blocks = page.get_text("dict")
                
                for block in blocks.get("blocks", []):
                    if "lines" in block:  # Text block
                        for line in block["lines"]:
                            line_text = ""
                            line_bbox = [float('inf'), float('inf'), 0, 0]
                            font_sizes = []
                            font_names = []
                            is_bold_flags = []
                            is_italic_flags = []
                            
                            for span in line["spans"]:
                                text = span["text"].strip()
                                if text:
                                    line_text += text + " "
                                    
                                    # Collect formatting info
                                    flags = span["flags"]
                                    is_bold_flags.append(bool(flags & 2**4))
                                    is_italic_flags.append(bool(flags & 2**1))
                                    font_sizes.append(span["size"])
                                    font_names.append(span["font"])
                                    
                                    # Update bounding box
                                    bbox = span["bbox"]
                                    line_bbox[0] = min(line_bbox[0], bbox[0])
                                    line_bbox[1] = min(line_bbox[1], bbox[1])
                                    line_bbox[2] = max(line_bbox[2], bbox[2])
                                    line_bbox[3] = max(line_bbox[3], bbox[3])
                            
                            if line_text.strip():
                                # Aggregate formatting information
                                avg_font_size = np.mean(font_sizes) if font_sizes else 12.0
                                most_common_font = max(set(font_names), key=font_names.count) if font_names else "Unknown"
                                is_bold = any(is_bold_flags) if is_bold_flags else False
                                is_italic = any(is_italic_flags) if is_italic_flags else False
                                
                                extracted_data.append({
                                    'text': line_text.strip(),
                                    'page_number': page_num + 1,
                                    'font_size': round(avg_font_size, 1),
                                    'font_name': most_common_font,
                                    'is_bold': 1 if is_bold else 0,
                                    'is_italic': 1 if is_italic else 0,
                                    'x0': round(line_bbox[0], 2),
                                    'y0': round(line_bbox[1], 2),
                                    'x1': round(line_bbox[2], 2),
                                    'y1': round(line_bbox[3], 2),
                                    'width': round(line_bbox[2] - line_bbox[0], 2),
                                    'height': round(line_bbox[3] - line_bbox[1], 2)
                                })
            
            doc.close()
            print(f"\n✅ Extracted {len(extracted_data)} text blocks using PyMuPDF")
            
        except Exception as e:
            print(f"\n❌ Error with PyMuPDF: {e}")
            return pd.DataFrame()
        
        df = pd.DataFrame(extracted_data)
        return self._clean_extracted_data(df)
    
    def _extract_with_pdfplumber(self, pdf_path):
        """Extract text using pdfplumber (alternative method)."""
        extracted_data = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                print(f"📄 Total pages: {len(pdf.pages)}")
                
                for page_num, page in enumerate(pdf.pages, 1):
                    print(f"  Processing page {page_num}/{len(pdf.pages)}...", end='\r')
                    
                    # Extract text with character-level details
                    chars = page.chars
                    
                    if chars:
                        # Group characters into text blocks
                        text_blocks = self._group_chars_into_blocks(chars)
                        
                        for block in text_blocks:
                            if len(block['text'].strip()) > 2:
                                extracted_data.append({
                                    'text': block['text'].strip(),
                                    'page_number': page_num,
                                    'font_size': block['font_size'],
                                    'font_name': block['font_name'],
                                    'is_bold': 1 if 'Bold' in block['font_name'] else 0,
                                    'is_italic': 1 if 'Italic' in block['font_name'] else 0,
                                    'x0': block['x0'],
                                    'y0': block['y0'],
                                    'x1': block['x1'],
                                    'y1': block['y1'],
                                    'width': block['x1'] - block['x0'],
                                    'height': block['y1'] - block['y0']
                                })
            
            print(f"\n✅ Extracted {len(extracted_data)} text blocks using pdfplumber")
            
        except Exception as e:
            print(f"\n❌ Error with pdfplumber: {e}")
            return pd.DataFrame()
        
        df = pd.DataFrame(extracted_data)
        return self._clean_extracted_data(df)
    
    def _group_chars_into_blocks(self, chars):
        """Group individual characters into text blocks."""
        if not chars:
            return []
        
        blocks = []
        current_block = {
            'text': '',
            'font_size': chars[0]['size'],
            'font_name': chars[0]['fontname'],
            'x0': chars[0]['x0'],
            'y0': chars[0]['y0'],
            'x1': chars[0]['x1'],
            'y1': chars[0]['y1']
        }
        
        for char in chars:
            # Check if character belongs to current block
            if (abs(char['size'] - current_block['font_size']) < 0.5 and
                char['fontname'] == current_block['font_name'] and
                abs(char['y0'] - current_block['y0']) < 5):
                
                current_block['text'] += char['text']
                current_block['x1'] = max(current_block['x1'], char['x1'])
                current_block['y1'] = max(current_block['y1'], char['y1'])
            else:
                if current_block['text'].strip():
                    blocks.append(current_block)
                
                current_block = {
                    'text': char['text'],
                    'font_size': char['size'],
                    'font_name': char['fontname'],
                    'x0': char['x0'],
                    'y0': char['y0'],
                    'x1': char['x1'],
                    'y1': char['y1']
                }
        
        if current_block['text'].strip():
            blocks.append(current_block)
        
        return blocks
    
    def _clean_extracted_data(self, df):
        """Clean and preprocess extracted data."""
        if df.empty:
            return df
        
        # Remove very short text blocks (likely artifacts)
        df = df[df['text'].str.len() >= 3].copy()
        
        # Remove purely numeric or punctuation-only blocks
        df = df[~df['text'].str.match(r'^[0-9\s\.\,\:\;\(\)\[\]\{\}]+$')].copy()
        
        # Sort by page and position
        df = df.sort_values(['page_number', 'y0', 'x0']).reset_index(drop=True)
        
        print(f"📊 After cleaning: {len(df)} text blocks")
        return df
    
    def extract_text_features(self, text_series):
        """Extract text-based features."""
        features = pd.DataFrame()
        
        features['text_length'] = text_series.str.len()
        features['word_count'] = text_series.str.split().str.len()
        
        features['uppercase_word_count'] = text_series.apply(
            lambda x: len([word for word in str(x).split() if word.isupper()]) if pd.notna(x) else 0
        )
        
        features['uppercase_ratio'] = text_series.apply(
            lambda x: sum(1 for c in str(x) if c.isupper()) / len(str(x)) if pd.notna(x) and len(str(x)) > 0 else 0
        )
        
        features['contains_numbers'] = text_series.str.contains(r'\d', regex=True, na=False).astype(int)
        features['starts_with_number'] = text_series.str.match(r'^\d', na=False).astype(int)
        
        features['special_char_count'] = text_series.apply(
            lambda x: len(re.findall(r'[^\w\s]', str(x))) if pd.notna(x) else 0
        )
        
        features['avg_word_length'] = text_series.apply(
            lambda x: np.mean([len(word) for word in str(x).split()]) if pd.notna(x) and len(str(x).split()) > 0 else 0
        )
        
        return features
    
    def prepare_features(self, df):
        """Prepare features for prediction."""
        print("🔄 Preparing features for prediction...")
        
        # Clean text data
        df['text'] = df['text'].fillna('').astype(str)
        
        # Extract TF-IDF features
        print("  - Extracting TF-IDF features...")
        tfidf_features = self.tfidf.transform(df['text'])
        
        # Extract text-based features
        print("  - Extracting text features...")
        text_features = self.extract_text_features(df['text'])
        
        # Prepare metadata features
        print("  - Preparing metadata features...")
        metadata_df = pd.DataFrame()
        
        # Add existing columns
        for feature in self.metadata_features:
            if feature in text_features.columns:
                metadata_df[feature] = text_features[feature]
            elif feature in df.columns:
                metadata_df[feature] = df[feature]
            else:
                # Create missing features with default values
                if feature == 'font_size':
                    metadata_df[feature] = df.get('font_size', 12.0)
                elif feature in ['is_bold', 'is_italic']:
                    metadata_df[feature] = df.get(feature, 0)
                elif feature in ['bbox_width', 'bbox_height', 'bbox_area']:
                    if feature == 'bbox_width':
                        metadata_df[feature] = df.get('width', 0)
                    elif feature == 'bbox_height':
                        metadata_df[feature] = df.get('height', 0)
                    else:  # bbox_area
                        metadata_df[feature] = df.get('width', 0) * df.get('height', 0)
                else:
                    metadata_df[feature] = 0
        
        # Ensure correct data types
        for feature in self.metadata_features:
            metadata_df[feature] = pd.to_numeric(metadata_df[feature], errors='coerce').fillna(0)
        
        # Combine features
        from scipy.sparse import hstack, csr_matrix
        metadata_sparse = csr_matrix(metadata_df.values)
        combined_features = hstack([tfidf_features, metadata_sparse])
        
        print(f"✅ Feature preparation complete: {combined_features.shape}")
        return combined_features
    
    def predict_headings(self, df, confidence_threshold=0.5):
        """
        Predict headings from extracted text.
        
        Args:
            df (pd.DataFrame): DataFrame with extracted text
            confidence_threshold (float): Minimum confidence for classification
            
        Returns:
            pd.DataFrame: DataFrame with predictions
        """
        if df.empty:
            print("⚠️  No text to classify")
            return df
        
        print("🔮 Predicting headings...")
        
        # Prepare features
        features = self.prepare_features(df)
        
        # Make predictions
        print("🔄 Making predictions...")
        features_dense = features.toarray()
        predictions = self.model.predict(features_dense)
        probabilities = self.model.predict_proba(features_dense)
        
        # Add predictions to dataframe
        result_df = df.copy()
        result_df['predicted_heading'] = self.label_encoder.inverse_transform(predictions)
        result_df['confidence'] = np.max(probabilities, axis=1)
        
        # Add individual class probabilities
        for i, class_name in enumerate(self.label_encoder.classes_):
            result_df[f'prob_{class_name}'] = probabilities[:, i]
        
        # Filter by confidence threshold
        low_confidence_mask = result_df['confidence'] < confidence_threshold
        result_df.loc[low_confidence_mask, 'predicted_heading'] = 'normal'
        
        print(f"✅ Predictions complete!")
        print(f"📊 Confidence threshold: {confidence_threshold}")
        print(f"📊 Low confidence predictions set to 'normal': {low_confidence_mask.sum()}")
        
        return result_df
    
    def extract_heading_hierarchy(self, result_df, confidence_threshold=0.5):
        """
        Extract hierarchical heading structure.
        
        Args:
            result_df (pd.DataFrame): DataFrame with predictions
            confidence_threshold (float): Minimum confidence for headings
            
        Returns:
            dict: Hierarchical heading structure
        """
        print("🏗️  Building heading hierarchy...")
        
        # Filter headings above confidence threshold
        headings = result_df[
            (result_df['predicted_heading'] != 'normal') & 
            (result_df['confidence'] >= confidence_threshold)
        ].copy()
        
        if headings.empty:
            print("⚠️  No headings found above confidence threshold")
            return {"headings": [], "document_structure": []}
        
        # Sort by page and position
        headings = headings.sort_values(['page_number', 'y0', 'x0']).reset_index(drop=True)
        
        # Create hierarchical structure
        hierarchy = []
        flat_headings = []
        
        for idx, row in headings.iterrows():
            heading_info = {
                'id': f"heading_{idx + 1}",
                'text': row['text'],
                'type': row['predicted_heading'],
                'confidence': float(row['confidence']),
                'page': int(row['page_number']),
                'position': {
                    'x0': float(row['x0']),
                    'y0': float(row['y0']),
                    'x1': float(row['x1']),
                    'y1': float(row['y1'])
                },
                'formatting': {
                    'font_size': float(row['font_size']),
                    'is_bold': bool(row['is_bold']),
                    'is_italic': bool(row['is_italic']),
                    'font_name': row.get('font_name', 'Unknown')
                }
            }
            
            flat_headings.append(heading_info)
            
            # Add to hierarchy based on heading type
            if row['predicted_heading'] == 'title':
                hierarchy.append({
                    'title': heading_info,
                    'sections': []
                })
            elif row['predicted_heading'] in ['h1', 'h2', 'h3'] and hierarchy:
                # Add to the last title's sections
                hierarchy[-1]['sections'].append(heading_info)
            elif row['predicted_heading'] in ['h1', 'h2', 'h3']:
                # Create new section without title
                hierarchy.append({
                    'title': None,
                    'sections': [heading_info]
                })
        
        result = {
            'headings': flat_headings,
            'document_structure': hierarchy
        }
        
        print(f"✅ Found {len(flat_headings)} headings")
        print(f"📊 Heading distribution:")
        heading_counts = headings['predicted_heading'].value_counts()
        for heading_type, count in heading_counts.items():
            print(f"   {heading_type}: {count}")
        
        return result
    
    def save_results(self, heading_structure, pdf_path, output_path):
        """Save results to JSON file."""
        print(f"💾 Saving results to: {output_path}")
        
        # Create output structure
        output_data = {
            'metadata': {
                'source_file': os.path.basename(pdf_path),
                'extraction_timestamp': datetime.now().isoformat(),
                'total_headings': len(heading_structure['headings']),
                'model_info': {
                    'classes': list(self.label_encoder.classes_),
                    'model_config': self.model_config
                }
            },
            'results': heading_structure
        }
        
        # Save to JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Results saved successfully!")
        return output_data
    
    def process_pdf(self, pdf_path, output_json=None, confidence_threshold=0.5, extraction_method='pymupdf'):
        """
        Complete PDF processing pipeline.
        
        Args:
            pdf_path (str): Path to PDF file
            output_json (str): Output JSON file path
            confidence_threshold (float): Minimum confidence for headings
            extraction_method (str): 'pymupdf' or 'pdfplumber'
            
        Returns:
            dict: Processing results
        """
        print(f"🚀 Processing PDF: {os.path.basename(pdf_path)}")
        print(f"📊 Confidence threshold: {confidence_threshold}")
        print(f"🔧 Extraction method: {extraction_method}")
        print("=" * 60)
        
        try:
            # Extract text
            df = self.extract_text_from_pdf(pdf_path, method=extraction_method)
            
            if df.empty:
                raise ValueError("No text extracted from PDF")
            
            # Predict headings
            result_df = self.predict_headings(df, confidence_threshold)
            
            # Extract hierarchy
            heading_structure = self.extract_heading_hierarchy(result_df, confidence_threshold)
            
            # Save results
            if output_json is None:
                base_name = os.path.splitext(os.path.basename(pdf_path))[0]
                output_json = f"{base_name}_headings.json"
            
            output_data = self.save_results(heading_structure, pdf_path, output_json)
            
            print("=" * 60)
            print("🎉 Processing complete!")
            print(f"📄 Source: {os.path.basename(pdf_path)}")
            print(f"📊 Total text blocks: {len(df)}")
            print(f"🎯 Headings found: {len(heading_structure['headings'])}")
            print(f"💾 Output: {output_json}")
            
            return output_data
            
        except Exception as e:
            print(f"❌ Error processing PDF: {e}")
            raise

def main():
    parser = argparse.ArgumentParser(description='Extract headings from PDF using trained XGBoost model')
    parser.add_argument('--pdf_path', type=str, required=True, help='Path to PDF file')
    parser.add_argument('--output_json', type=str, help='Output JSON file path')
    parser.add_argument('--confidence_threshold', type=float, default=0.5, 
                        help='Minimum confidence threshold for headings (default: 0.5)')
    parser.add_argument('--extraction_method', type=str, default='pymupdf', 
                        choices=['pymupdf', 'pdfplumber'], help='Text extraction method')
    parser.add_argument('--models_dir', type=str, default='models', 
                        help='Directory containing model files')
    
    args = parser.parse_args()
    
    # Check dependencies
    if not PDF_LIBRARIES_AVAILABLE:
        print("❌ PDF processing libraries not available")
        print("Install with: pip install pymupdf pdfplumber")
        sys.exit(1)
    
    if not ML_LIBRARIES_AVAILABLE:
        print("❌ ML libraries not available")
        print("Install with: pip install xgboost scikit-learn scipy joblib")
        sys.exit(1)
    
    # Initialize extractor
    try:
        extractor = PDFHeadingExtractor(models_dir=args.models_dir)
    except Exception as e:
        print(f"❌ Failed to initialize extractor: {e}")
        sys.exit(1)
    
    # Process PDF
    try:
        result = extractor.process_pdf(
            pdf_path=args.pdf_path,
            output_json=args.output_json,
            confidence_threshold=args.confidence_threshold,
            extraction_method=args.extraction_method
        )
        
        # Print summary
        headings = result['results']['headings']
        if headings:
            print(f"\n📋 Extracted Headings:")
            for heading in headings:
                print(f"   {heading['type'].upper()}: \"{heading['text']}\" (confidence: {heading['confidence']:.3f})")
        else:
            print(f"\n⚠️  No headings found above confidence threshold {args.confidence_threshold}")
        
    except Exception as e:
        print(f"❌ Processing failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
