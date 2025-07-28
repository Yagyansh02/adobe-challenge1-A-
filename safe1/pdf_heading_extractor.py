#!/usr/bin/env python3
"""
PDF Heading and Title Extraction Pipeline
==========================================

A comprehensive tool for extracting headings and titles from PDF documents using
machine learning with rule-based enhancements.

Features:
- XGBoost-based classification with rule-based spatial enhancements
- Simplified percentile-based bbox enhancement system
- Batch processing of multiple PDFs
- JSON output with comprehensive metadata

Author: Generated from Jupyter Notebook Pipeline
Date: 2025-07-28
"""

import os
import sys
import json
import argparse
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any

import pandas as pd
import numpy as np
import fitz  # PyMuPDF
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

# Suppress warnings
warnings.filterwarnings('ignore')

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

class PDFHeadingExtractor:
    """PDF Heading and Title Extraction Pipeline"""
    
    def __init__(self, model_path: str = "xgboost_pdf_classifier.pkl", 
                 preprocessor_path: str = "feature_preprocessor.pkl",
                 label_encoder_path: str = "label_encoder.pkl"):
        """
        Initialize the PDF heading extractor
        
        Args:
            model_path: Path to trained XGBoost model
            preprocessor_path: Path to feature preprocessor
            label_encoder_path: Path to label encoder
        """
        self.model_path = model_path
        self.preprocessor_path = preprocessor_path
        self.label_encoder_path = label_encoder_path
        self.model = None
        self.preprocessor = None
        self.label_encoder = None
        
    def load_models(self):
        """Load the trained models and preprocessors"""
        try:
            print(f"[LOAD] Loading models...")
            self.model = joblib.load(self.model_path)
            self.preprocessor = joblib.load(self.preprocessor_path)
            self.label_encoder = joblib.load(self.label_encoder_path)
            print(f"[OK] Models loaded successfully!")
        except Exception as e:
            print(f"[ERROR] Error loading models: {e}")
            sys.exit(1)
    
    def extract_text_with_metadata(self, pdf_path: str) -> List[Dict]:
        """Extract text blocks with metadata from PDF"""
        text_blocks = []
        
        try:
            doc = fitz.open(pdf_path)
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                blocks = page.get_text("dict")
                
                for block in blocks["blocks"]:
                    if "lines" in block:
                        for line in block["lines"]:
                            for span in line["spans"]:
                                text = span["text"].strip()
                                if text and len(text) > 1:
                                    text_blocks.append({
                                        'text': text,
                                        'page_number': page_num + 1,
                                        'font_size': span['size'],
                                        'is_bold': bool(span['flags'] & 2**4),
                                        'is_italic': bool(span['flags'] & 2**1),
                                        'x1': span['bbox'][0],
                                        'y1': span['bbox'][1],
                                        'x2': span['bbox'][2],
                                        'y2': span['bbox'][3]
                                    })
            
            doc.close()
            
        except Exception as e:
            print(f"[ERROR] Error extracting text from {pdf_path}: {e}")
            return []
            
        return text_blocks
    
    def create_features_for_prediction(self, text_blocks: List[Dict]) -> pd.DataFrame:
        """Create features for model prediction"""
        if not text_blocks:
            return pd.DataFrame()
        
        df = pd.DataFrame(text_blocks)
        
        # Text-based features
        df['text_length'] = df['text'].str.len()
        df['word_count'] = df['text'].str.split().str.len()
        df['all_caps_ratio'] = df['text'].apply(lambda x: sum(1 for c in x if c.isupper()) / len(x) if len(x) > 0 else 0)
        df['has_colon'] = df['text'].str.contains(':')
        df['is_uppercase'] = df['text'].str.isupper()
        df['starts_with_number'] = df['text'].str.match(r'^\d')
        df['ends_with_colon'] = df['text'].str.endswith(':')
        df['special_chars_count'] = df['text'].apply(lambda x: sum(1 for c in x if not c.isalnum() and not c.isspace()))
        
        # Font features
        df['font_size_normalized'] = df.groupby('page_number')['font_size'].transform(
            lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else 0
        )
        
        # Position features
        page_bounds = df.groupby('page_number').agg({'x1': ['min', 'max'], 'y1': ['min', 'max']})
        page_bounds.columns = ['_'.join(col).strip() for col in page_bounds.columns]
        
        df = df.merge(page_bounds, left_on='page_number', right_index=True, how='left')
        
        df['x1_norm'] = ((df['x1'] - df['x1_min']) / 
                        (df['x1_max'] - df['x1_min'])).fillna(0)
        df['y1_norm'] = ((df['y1'] - df['y1_min']) / 
                        (df['y1_max'] - df['y1_min'])).fillna(0)
        
        # Bbox features
        df['bbox_width'] = df['x2'] - df['x1']
        df['bbox_height'] = df['y2'] - df['y1']
        
        # Text complexity features
        df['avg_word_length'] = df['text'].apply(
            lambda x: np.mean([len(word) for word in x.split()]) if len(x.split()) > 0 else 0
        )
        df['sentence_count'] = df['text'].str.count(r'[.!?]+') + 1
        
        return df
    
    def classify_pdf_text_with_rules(self, df: pd.DataFrame) -> pd.DataFrame:
        """Classify PDF text with rule-based enhancements"""
        if df.empty:
            return df
        
        # Define feature columns
        numerical_features = [
            'font_size_normalized', 'text_length', 'word_count', 
            'all_caps_ratio', 'avg_word_length', 'sentence_count',
            'special_chars_count', 'x1_norm', 'y1_norm', 
            'bbox_width', 'bbox_height', 'page_number'
        ]
        
        boolean_features = [
            'is_bold', 'is_italic', 'has_colon', 'is_uppercase',
            'starts_with_number', 'ends_with_colon'
        ]
        
        feature_columns = ['text'] + numerical_features + boolean_features
        
        # Prepare features
        X = df[feature_columns]
        
        # Make predictions
        X_processed = self.preprocessor.transform(X)
        prediction_probabilities = self.model.predict_proba(X_processed)
        predictions_encoded = prediction_probabilities.argmax(axis=1)
        predictions = self.label_encoder.inverse_transform(predictions_encoded)
        confidence = prediction_probabilities.max(axis=1)
        
        # Create result dataframe
        df_result = df.copy()
        df_result['predicted_class'] = predictions
        df_result['confidence'] = confidence
        
        # Add probability columns
        class_names = self.label_encoder.classes_
        for i, class_name in enumerate(class_names):
            df_result[f'prob_{class_name}'] = prediction_probabilities[:, i]
        
        # Store original prob_h1 for comparison
        df_result['prob_h1_original'] = df_result['prob_h1'].copy()
        
        # APPLY RULE-BASED ENHANCEMENTS
        print("[TARGET] Applying simplified percentile-based constraints...")
        
        # Calculate page statistics
        page_stats = df_result.groupby('page_number').agg({
            'y1': ['min', 'max'],
            'bbox_width': 'mean',
            'bbox_height': 'mean'
        }).round(2)
        
        # Calculate percentiles for bbox dimensions within this PDF
        df_result['bbox_area'] = df_result['bbox_width'] * df_result['bbox_height']
        
        # Calculate percentiles for height, width (simplified to 60th and 85th)
        height_percentiles = df_result['bbox_height'].quantile([0.60, 0.85]).to_dict()
        width_percentiles = df_result['bbox_width'].quantile([0.60, 0.85]).to_dict()
        
        print(f"[STATS] PDF Bbox Percentiles:")
        print(f"   Height: 60th={height_percentiles[0.60]:.1f}, 85th={height_percentiles[0.85]:.1f}")
        print(f"   Width:  60th={width_percentiles[0.60]:.1f}, 85th={width_percentiles[0.85]:.1f}")
        
        # Apply rule-based enhancements
        enhanced_prob_h1 = df_result['prob_h1'].copy()
        enhancement_reasons = [''] * len(df_result)
        
        for idx, row in df_result.iterrows():
            page_num = row['page_number']
            
            # Get page boundaries
            if page_num in page_stats.index:
                page_min_y = page_stats.loc[page_num, ('y1', 'min')]
                page_max_y = page_stats.loc[page_num, ('y1', 'max')]
                page_height = page_max_y - page_min_y if page_max_y > page_min_y else 1
            else:
                page_height = 1
                page_min_y = row['y1']
            
            # Rule 1: Position-based enhancement (top of page)
            y_position_ratio = (row['y1'] - page_min_y) / page_height if page_height > 0 else 0
            is_top_of_page = y_position_ratio <= 0.15
            
            # Rule 2: Simplified percentile-based bbox enhancement (3 tiers)
            current_height = row['bbox_height']
            current_width = row['bbox_width']
            
            # Determine percentile tier based on BOTH height and width
            height_tier = 0
            width_tier = 0
            
            if current_height > height_percentiles[0.85]:
                height_tier = 3  # >85th percentile
            elif current_height > height_percentiles[0.60]:
                height_tier = 2  # 60th-85th percentile
            else:
                height_tier = 1  # <=60th percentile
            
            if current_width > width_percentiles[0.85]:
                width_tier = 3  # >85th percentile
            elif current_width > width_percentiles[0.60]:
                width_tier = 2  # 60th-85th percentile
            else:
                width_tier = 1  # <=60th percentile
            
            # Use the lower tier to be conservative
            bbox_tier = min(height_tier, width_tier)
            
            # Apply enhancements
            enhancement_applied = []
            current_prob_h1 = row['prob_h1']
            
            if is_top_of_page:
                current_prob_h1 = min(current_prob_h1 + 0.3, 0.99)
                enhancement_applied.append("top_position(+0.3)")
            
            # Apply simplified percentile-based enhancement with new decrease logic
            if bbox_tier == 3:  # >85th percentile
                current_prob_h1 = min(current_prob_h1 + 0.4, 0.99)
                enhancement_applied.append("bbox_85th+(+0.4)")
            elif bbox_tier == 2:  # 60th-85th percentile
                current_prob_h1 = max(current_prob_h1 - 0.075, 0.001)
                enhancement_applied.append("bbox_60th-85th(-0.075)")
            elif bbox_tier == 1:
                # Check if it's in the 60th percentile region (near but not above 60th)
                if (current_height > height_percentiles[0.60] * 0.8 or 
                    current_width > width_percentiles[0.60] * 0.8):
                    current_prob_h1 = max(current_prob_h1 - 0.1, 0.001)
                    enhancement_applied.append("bbox_near_60th(-0.15)")
                else:
                    # Check if it's in the 45-60th percentile region
                    height_45th = df_result['bbox_height'].quantile(0.45)
                    width_45th = df_result['bbox_width'].quantile(0.45)
                    
                    if ((current_height >= height_45th and current_height <= height_percentiles[0.60]) or
                        (current_width >= width_45th and current_width <= width_percentiles[0.60])):
                        current_prob_h1 = max(current_prob_h1 - 0.25, 0.001)
                        enhancement_applied.append("bbox_45th-60th(-0.25)")
                    # For anything below 45th percentile, no enhancement is applied
            
            # Update enhanced probability
            enhanced_prob_h1.iloc[idx] = current_prob_h1
            enhancement_reasons[idx] = " | ".join(enhancement_applied) if enhancement_applied else "none"
        
        # Update the dataframe with enhanced probabilities
        df_result['prob_h1_original'] = df_result['prob_h1'].copy()
        df_result['prob_h1'] = enhanced_prob_h1
        df_result['enhancement_reasons'] = enhancement_reasons
        
        # Recalculate predictions based on enhanced probabilities
        enhanced_probabilities = prediction_probabilities.copy()
        h1_idx = list(class_names).index('h1')
        normal_idx = list(class_names).index('normal')
        
        # Apply enhancements to the probability matrix
        for idx in range(len(enhanced_probabilities)):
            original_h1_prob = prediction_probabilities[idx, h1_idx]
            enhanced_h1_prob = enhanced_prob_h1.iloc[idx]
            prob_change = enhanced_h1_prob - original_h1_prob
            
            # Apply the H1 probability change and adjust normal probability accordingly
            if prob_change != 0:
                enhanced_probabilities[idx, h1_idx] = enhanced_h1_prob
                # If H1 increased, reduce normal; if H1 decreased, increase normal
                enhanced_probabilities[idx, normal_idx] = max(
                    enhanced_probabilities[idx, normal_idx] - prob_change, 
                    0.001
                )
        
        # Renormalize probabilities to sum to 1
        prob_sums = enhanced_probabilities.sum(axis=1)
        for i in range(len(enhanced_probabilities)):
            if prob_sums[i] > 0:
                enhanced_probabilities[i] = enhanced_probabilities[i] / prob_sums[i]
        
        # Update all probability columns
        for i, class_name in enumerate(class_names):
            df_result[f'prob_{class_name}'] = enhanced_probabilities[:, i]
        
        # Recalculate predicted class and confidence
        new_predictions_encoded = enhanced_probabilities.argmax(axis=1)
        new_predictions = self.label_encoder.inverse_transform(new_predictions_encoded)
        new_confidence = enhanced_probabilities.max(axis=1)
        
        df_result['predicted_class'] = new_predictions
        df_result['confidence'] = new_confidence
        
        # Add bbox information as structured data
        df_result['bbox'] = df_result.apply(lambda row: {
            'x1': float(row['x1']), 'y1': float(row['y1']), 
            'x2': float(row['x2']), 'y2': float(row['y2']),
            'width': float(row['bbox_width']), 
            'height': float(row['bbox_height']),
            'area': float(row['bbox_width'] * row['bbox_height'])
        }, axis=1)
        
        print("[OK] Rule-based enhancements applied!")
        
        # Print enhancement statistics
        enhanced_count = len([r for r in enhancement_reasons if r != "none"])
        top_enhancements = len([r for r in enhancement_reasons if "top_position" in r])
        bbox_85th_enhancements = len([r for r in enhancement_reasons if "bbox_85th+" in r])
        bbox_60th_85th_enhancements = len([r for r in enhancement_reasons if "bbox_60th-85th" in r])
        bbox_near_60th_enhancements = len([r for r in enhancement_reasons if "bbox_near_60th" in r])
        bbox_45th_60th_enhancements = len([r for r in enhancement_reasons if "bbox_45th-60th" in r])
        
        print(f"[STATS] Enhanced {enhanced_count} predictions with simplified percentile-based constraints")
        print(f"   - Top position boosts (+0.1): {top_enhancements}")
        print(f"   - >85th percentile bbox (+0.4): {bbox_85th_enhancements}")
        print(f"   - 60th-85th percentile bbox (-0.075): {bbox_60th_85th_enhancements}")
        print(f"   - Near 60th percentile bbox (-0.15): {bbox_near_60th_enhancements}")
        print(f"   - 45th-60th percentile bbox (-0.25): {bbox_45th_60th_enhancements}")
        
        return df_result
    
    def process_pdf(self, pdf_path: str, confidence_threshold: float = 0.55, 
                   normal_confidence_threshold: float = 0.85) -> Dict:
        """
        Process a single PDF file and extract headings/titles
        
        Args:
            pdf_path: Path to PDF file
            confidence_threshold: Minimum confidence for title/h1 predictions
            normal_confidence_threshold: Minimum confidence for normal text
            
        Returns:
            Dict: Processing results
        """
        print(f"[START] Processing PDF: {Path(pdf_path).name}")
        print("=" * 60)
        
        # Step 1: Extract text with metadata
        text_blocks = self.extract_text_with_metadata(pdf_path)
        
        if not text_blocks:
            return {'error': 'No text extracted from PDF'}
        
        # Step 2: Create features
        df = self.create_features_for_prediction(text_blocks)
        
        # Step 3: Make enhanced predictions with rule-based constraints
        df_with_predictions = self.classify_pdf_text_with_rules(df)
        
        # Step 4: Apply confidence filtering
        high_confidence_mask = (
            ((df_with_predictions['predicted_class'] == 'title') & 
             (df_with_predictions['confidence'] >= confidence_threshold)) |
            ((df_with_predictions['predicted_class'] == 'h1') & 
             (df_with_predictions['confidence'] >= confidence_threshold)) |
            ((df_with_predictions['predicted_class'] == 'normal') & 
             (df_with_predictions['confidence'] >= normal_confidence_threshold))
        )
        
        high_confidence = df_with_predictions[high_confidence_mask]
        
        # Extract titles and headings
        titles = high_confidence[high_confidence['predicted_class'] == 'title']
        h1_headings = high_confidence[high_confidence['predicted_class'] == 'h1']
        
        print(f"[STATS] Processing Results:")
        print(f"   [TARGET] Total text blocks: {len(df_with_predictions):,}")
        print(f"   [CHART] High confidence predictions: {len(high_confidence):,}")
        print(f"   [?] Titles: {len(titles)}")
        print(f"   [RESULT] H1 headings: {len(h1_headings)}")
        
        # Calculate enhancement statistics
        enhancement_reasons = df_with_predictions['enhancement_reasons']
        
        # Prepare results
        results = {
            'file_name': Path(pdf_path).name,
            'total_text_blocks': len(df_with_predictions),
            'high_confidence_predictions': len(high_confidence),
            'confidence_threshold': confidence_threshold,
            'normal_confidence_threshold': normal_confidence_threshold,
            'rule_enhancements': {
                'total_enhanced': len([r for r in enhancement_reasons if r != "none"]),
                'top_position_enhancements': len([r for r in enhancement_reasons if "top_position" in r]),
                'large_bbox_enhancements': 0  # Legacy field
            },
            'titles': {
                'count': len(titles),
                'items': []
            },
            'h1_headings': {
                'count': len(h1_headings),
                'items': []
            },
            'statistics': {
                'avg_confidence': high_confidence['confidence'].mean() if len(high_confidence) > 0 else 0,
                'class_distribution': df_with_predictions['predicted_class'].value_counts().to_dict(),
                'high_confidence_distribution': high_confidence['predicted_class'].value_counts().to_dict()
            }
        }
        
        # Process titles
        for _, title in titles.iterrows():
            prob_improvement = 0
            if 'prob_h1_original' in title and title['prob_h1_original'] > 0:
                prob_improvement = ((title['prob_h1'] - title['prob_h1_original']) / 
                                  title['prob_h1_original']) * 100
            
            title_data = {
                'text': title['text'],
                'page': int(title['page_number']),
                'font_size': float(title['font_size']),
                'is_bold': bool(title['is_bold']),
                'confidence': float(title['confidence']),
                'bbox': title['bbox'],
                'enhancement_applied': title.get('enhancement_reasons', 'none'),
                'prob_h1_original': float(title.get('prob_h1_original', title['confidence'])),
                'prob_h1_enhanced': float(title['prob_h1']),
                'improvement_percentage': float(prob_improvement)
            }
            results['titles']['items'].append(title_data)
        
        # Process H1 headings
        for _, heading in h1_headings.iterrows():
            prob_improvement = 0
            if 'prob_h1_original' in heading and heading['prob_h1_original'] > 0:
                prob_improvement = ((heading['prob_h1'] - heading['prob_h1_original']) / 
                                  heading['prob_h1_original']) * 100
            
            heading_data = {
                'text': heading['text'],
                'page': int(heading['page_number']),
                'font_size': float(heading['font_size']),
                'is_bold': bool(heading['is_bold']),
                'confidence': float(heading['confidence']),
                'bbox': heading['bbox'],
                'enhancement_applied': heading.get('enhancement_reasons', 'none'),
                'prob_h1_original': float(heading.get('prob_h1_original', heading['confidence'])),
                'prob_h1_enhanced': float(heading['prob_h1']),
                'improvement_percentage': float(prob_improvement)
            }
            results['h1_headings']['items'].append(heading_data)
        
        return results
    
    def process_folder(self, folder_path: str, output_folder: str = "output", 
                      confidence_threshold: float = 0.55, 
                      normal_confidence_threshold: float = 0.85):
        """
        Process all PDF files in a folder
        
        Args:
            folder_path: Path to folder containing PDF files
            output_folder: Path to output folder for JSON results
            confidence_threshold: Minimum confidence for title/h1 predictions
            normal_confidence_threshold: Minimum confidence for normal text
        """
        folder_path = Path(folder_path)
        output_folder = Path(output_folder)
        output_folder.mkdir(exist_ok=True)
        
        pdf_files = list(folder_path.glob("*.pdf"))
        
        if not pdf_files:
            print(f"[ERROR] No PDF files found in {folder_path}")
            return
        
        print(f"[START] Processing {len(pdf_files)} PDF files from {folder_path}")
        print(f"[FOLDER] Output folder: {output_folder}")
        print("=" * 70)
        
        processed_count = 0
        failed_count = 0
        
        for pdf_file in pdf_files:
            try:
                print(f"\n[FILE] Processing: {pdf_file.name}")
                
                # Process the PDF
                results = self.process_pdf(str(pdf_file), confidence_threshold, 
                                         normal_confidence_threshold)
                
                if 'error' in results:
                    print(f"[ERROR] Failed to process {pdf_file.name}: {results['error']}")
                    failed_count += 1
                    continue
                
                # Save results to JSON
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_filename = f"{pdf_file.stem}_headings_{timestamp}.json"
                output_path = output_folder / output_filename
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
                
                print(f"[OK] Saved results to: {output_filename}")
                print(f"   [RESULT] Found {results['h1_headings']['count']} H1 headings and {results['titles']['count']} titles")
                
                processed_count += 1
                
            except Exception as e:
                print(f"[ERROR] Error processing {pdf_file.name}: {e}")
                failed_count += 1
        
        print(f"\n[COMPLETE] BATCH PROCESSING COMPLETE!")
        print(f"   [OK] Successfully processed: {processed_count} files")
        print(f"   [ERROR] Failed: {failed_count} files")
        print(f"   [FOLDER] Results saved in: {output_folder}")

def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(description="Extract headings and titles from PDF documents")
    parser.add_argument("input", help="PDF file or folder containing PDF files")
    parser.add_argument("-o", "--output", default="output", 
                       help="Output folder for JSON results (default: output)")
    parser.add_argument("-c", "--confidence", type=float, default=0.55,
                       help="Confidence threshold for titles/H1 headings (default: 0.55)")
    parser.add_argument("-n", "--normal-confidence", type=float, default=0.85,
                       help="Confidence threshold for normal text (default: 0.85)")
    parser.add_argument("--model", default="xgboost_pdf_classifier.pkl",
                       help="Path to XGBoost model file")
    parser.add_argument("--preprocessor", default="feature_preprocessor.pkl",
                       help="Path to feature preprocessor file")
    parser.add_argument("--label-encoder", default="label_encoder.pkl",
                       help="Path to label encoder file")
    
    args = parser.parse_args()
    
    # Initialize extractor
    extractor = PDFHeadingExtractor(args.model, args.preprocessor, args.label_encoder)
    extractor.load_models()
    
    input_path = Path(args.input)
    
    if input_path.is_file() and input_path.suffix.lower() == '.pdf':
        # Process single PDF file
        print(f"[TARGET] Processing single PDF file: {input_path}")
        results = extractor.process_pdf(str(input_path), args.confidence, args.normal_confidence)
        
        if 'error' in results:
            print(f"[ERROR] Error: {results['error']}")
            return
        
        # Save results
        output_folder = Path(args.output)
        output_folder.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{input_path.stem}_headings_{timestamp}.json"
        output_path = output_folder / output_filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
        
        print(f"\n[OK] Results saved to: {output_path}")
        print(f"[RESULT] Found {results['h1_headings']['count']} H1 headings and {results['titles']['count']} titles")
        
    elif input_path.is_dir():
        # Process folder of PDF files
        extractor.process_folder(str(input_path), args.output, args.confidence, args.normal_confidence)
        
    else:
        print(f"[ERROR] Invalid input: {input_path} (must be PDF file or directory)")
        return

if __name__ == "__main__":
    main()
