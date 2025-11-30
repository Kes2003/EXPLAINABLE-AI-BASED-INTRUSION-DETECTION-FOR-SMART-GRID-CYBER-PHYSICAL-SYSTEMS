#!/usr/bin/env python3
"""Setup script for Smart Grid IDS Project"""

from pathlib import Path
import subprocess
import sys

def create_directories():
    """Create necessary directories"""
    dirs = [
        "data/raw",
        "data/processed",
        "data/external",
        "models",
        "artifacts/visualizations",
        "config",
        "scripts",
        "tests",
        "docs",
        "logs"
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"[OK] Created directory: {dir_path}")

def check_dependencies():
    """Check if required packages are installed"""
    required = [
        "pandas",
        "numpy",
        "scikit-learn",
        "joblib",
        "matplotlib",
        "seaborn",
        "pyyaml",
        "fastapi",
        "uvicorn",
        "pydantic"
    ]
    
    missing = []
    for package in required:
        try:
            __import__(package.replace("-", "_"))
            print(f"[OK] {package} is installed")
        except ImportError:
            missing.append(package)
            print(f"[!!] {package} is missing")
    
    if missing:
        print(f"\n[WARNING] Missing packages: {', '.join(missing)}")
        print("Install them with: pip install -r requirements.txt")
        return False
    return True

def verify_structure():
    """Verify project structure"""
    required_files = [
        "requirements.txt",
        "README.md",
        "Dockerfile",
        "config/config.yaml",
        "src/api.py",
        "src/train.py",
        "src/predict.py"
    ]
    
    missing = []
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"[OK] {file_path} exists")
        else:
            missing.append(file_path)
            print(f"[!!] {file_path} is missing")
    
    if missing:
        print(f"\n[WARNING] Missing files: {', '.join(missing)}")
        return False
    return True

def main():
    print("=" * 60)
    print("Smart Grid IDS Project Setup")
    print("=" * 60)
    print()
    
    print("1. Creating directory structure...")
    create_directories()
    print()
    
    print("2. Checking dependencies...")
    deps_ok = check_dependencies()
    print()
    
    print("3. Verifying project structure...")
    structure_ok = verify_structure()
    print()
    
    if deps_ok and structure_ok:
        print("=" * 60)
        print("[SUCCESS] Setup complete! Project is ready to use.")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Generate data: python -m src.generate_data --rows 20000 --out data/raw/smartgrid_synthetic.csv")
        print("2. Train model: python -m src.train --data data/raw/smartgrid_synthetic.csv --config config/config.yaml")
        print("3. Start API: python scripts/start_api.py")
    else:
        print("=" * 60)
        print("[WARNING] Setup incomplete. Please fix the issues above.")
        print("=" * 60)
        sys.exit(1)

if __name__ == "__main__":
    main()