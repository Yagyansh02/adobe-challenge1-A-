#!/usr/bin/env python3
"""
Docker Test Script
=================

Quick test to verify Docker setup works correctly.
"""

import os
import sys
from pathlib import Path

def test_docker_environment():
    """Test the Docker environment setup."""
    
    print("[TEST] Testing Docker Environment")
    print("=" * 40)
    
    # Test Python version
    print(f"[PYTHON] Version: {sys.version}")
    
    # Test required packages
    required_packages = [
        'pandas', 'numpy', 'sklearn', 'xgboost', 
        'fitz', 'pdfplumber', 'joblib', 'scipy'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"[✓] {package} - OK")
        except ImportError:
            print(f"[✗] {package} - MISSING")
            missing_packages.append(package)
    
    # Test model files
    print("\n[TEST] Checking model files...")
    
    safe1_models = [
        "safe1/feature_preprocessor.pkl",
        "safe1/label_encoder.pkl", 
        "safe1/xgboost_pdf_classifier.pkl"
    ]
    
    safe2_models = [
        "safe2/models/label_encoder.pkl",
        "safe2/models/tfidf_vectorizer.pkl",
        "safe2/models/pdf_heading_classifier.json"
    ]
    
    all_models = safe1_models + safe2_models
    missing_models = []
    
    for model_path in all_models:
        if Path(model_path).exists():
            print(f"[✓] {model_path} - Found")
        else:
            print(f"[✗] {model_path} - Missing")
            missing_models.append(model_path)
    
    # Test directories
    print("\n[TEST] Checking directories...")
    required_dirs = ["/app/input", "/app/output"]
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"[✓] {dir_path} - Exists")
        else:
            print(f"[✗] {dir_path} - Missing")
    
    # Summary
    print("\n[SUMMARY]")
    if missing_packages:
        print(f"[✗] Missing packages: {missing_packages}")
        return False
    if missing_models:
        print(f"[✗] Missing models: {missing_models}")
        return False
    
    print("[✓] All tests passed! Docker environment is ready.")
    return True

if __name__ == "__main__":
    success = test_docker_environment()
    sys.exit(0 if success else 1)
