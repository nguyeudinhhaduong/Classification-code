"""
Quick Start Script
Demonstrates basic usage of the pipeline.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils import load_config, setup_logging, seed_everything
from src.data_loader import DataLoader
from src.pipeline import CodeClassificationPipeline
import pandas as pd


def quick_demo():
    """Quick demonstration of the pipeline."""
    
    print("=" * 60)
    print("Code Classification Pipeline - Quick Demo")
    print("=" * 60)
    
    # Setup
    setup_logging(log_dir="logs", log_file="demo.log", level="INFO")
    config = load_config("configs/config.yaml")
    seed_everything(config['project']['seed'])
    
    # Example code snippets for testing
    sample_codes = [
        """
def hello_world():
    print("Hello, World!")
    return True
        """,
        
        """
public class HelloWorld {
    public static void main(String[] args) {
        System.out.println("Hello, World!");
    }
}
        """,
        
        """
#include <iostream>
using namespace std;

int main() {
    cout << "Hello, World!" << endl;
    return 0;
}
        """
    ]
    
    print("\n1. Testing Language Detection...")
    print("-" * 60)
    
    # Initialize pipeline
    pipeline = CodeClassificationPipeline(config)
    
    # Enable only language detection stage
    config['pipeline']['stages'][0]['enabled'] = True  # Language detection
    config['pipeline']['stages'][1]['enabled'] = False  # BERT classifier
    
    pipeline.initialize_stages()
    
    # Run language detection
    try:
        detected_languages = pipeline.run_language_detection(sample_codes)
        
        print("\nResults:")
        for i, (code, lang) in enumerate(zip(sample_codes, detected_languages)):
            print(f"\nSample {i+1}:")
            print(f"Detected Language: {lang}")
            print(f"Code Preview: {code.strip()[:80]}...")
    
    except Exception as e:
        print(f"Note: Language detection requires model download.")
        print(f"Error: {e}")
    
    print("\n" + "=" * 60)
    print("Demo completed!")
    print("=" * 60)
    
    print("\n📖 Next Steps:")
    print("1. Prepare your data in data/raw/ folder")
    print("2. Run training: python main.py --mode train")
    print("3. Run evaluation: python main.py --mode eval --checkpoint path/to/checkpoint")
    print("4. Generate predictions: python main.py --mode predict --checkpoint path/to/checkpoint")
    print("\nSee README.md for detailed instructions.")


if __name__ == "__main__":
    quick_demo()
