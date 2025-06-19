#!/usr/bin/env python3
import os
import sys
import torch
import pickle
from pathlib import Path

def verify_setup():
    """Verify all components are properly set up."""
    issues = []
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 7):
        issues.append("Python version must be 3.7 or higher")
    
    # Check required directories
    required_dirs = ['out', 'data/void']
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            issues.append(f"Missing directory: {dir_path}")
    
    # Check required files
    required_files = {
        'out/model.pt': 'Model weights',
        'data/void/vocab.pkl': 'Vocabulary file',
        'data/void/meta.pkl': 'Meta configuration',
        'requirements.txt': 'Requirements file'
    }
    
    for file_path, description in required_files.items():
        if not os.path.exists(file_path):
            issues.append(f"Missing {description}: {file_path}")
    
    # Check PyTorch
    try:
        print("PyTorch version:", torch.__version__)
        print("CUDA available:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("CUDA version:", torch.version.cuda)
            print("GPU device:", torch.cuda.get_device_name(0))
    except Exception as e:
        issues.append(f"PyTorch issue: {str(e)}")
    
    # Try loading model files
    if os.path.exists('data/void/vocab.pkl'):
        try:
            with open('data/void/vocab.pkl', 'rb') as f:
                pickle.load(f)
        except Exception as e:
            issues.append(f"Error loading vocab.pkl: {str(e)}")
    
    if os.path.exists('data/void/meta.pkl'):
        try:
            with open('data/void/meta.pkl', 'rb') as f:
                pickle.load(f)
        except Exception as e:
            issues.append(f"Error loading meta.pkl: {str(e)}")
    
    # Check environment variables (warn only for PORT at build time)
    if not os.getenv('PORT'):
        print('⚠️  Warning: PORT environment variable is not set. This is expected during build and will be set by Render at runtime.')
    
    return issues

def main():
    print("=== Void AI Setup Verification ===")
    issues = verify_setup()
    
    if issues:
        print("\n❌ Found the following issues:")
        for issue in issues:
            print(f"  - {issue}")
        print("\nPlease fix these issues before running the server.")
        sys.exit(1)
    else:
        print("\n✅ All checks passed! The system is properly configured.")

if __name__ == '__main__':
    main()
