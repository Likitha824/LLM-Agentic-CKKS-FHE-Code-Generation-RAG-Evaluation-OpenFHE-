#!/usr/bin/env python3
"""
Batch CrystalBLEU Calculator for CKKS Code Evaluation

This script calculates CrystalBLEU scores for all operations at once
by comparing generated code with reference implementations.

Usage:
    python batch_calculate_crystalbleu.py [model] [technique]
"""

import os
import sys
import re
import json
import math
from collections import Counter
from pathlib import Path

# Default parameters
MODEL = "openai"
TECHNIQUE = "self_improvement_iterative"

# Parse command line arguments
if len(sys.argv) > 1:
    MODEL = sys.argv[1]
if len(sys.argv) > 2:
    TECHNIQUE = sys.argv[2]

# Operations to process
OPERATIONS = ["addition", "multiplication", "dot_product", "matrix_multiplication", "convolution"]

# Directories
REF_DIR = "data/reference_implementations"
# Modify the RESULTS_DIR definition
RESULTS_DIR = f"results/metrics/{MODEL}/{TECHNIQUE}"

# Ensure results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

def normalize_code(code_text):
    """Normalize code for comparison."""
    # Remove comments
    code_text = re.sub(r'//.*?\n', '\n', code_text)
    code_text = re.sub(r'/\*.*?\*/', '', code_text, flags=re.DOTALL)
    
    # Remove empty lines
    code_text = re.sub(r'\n\s*\n', '\n', code_text)
    
    # Normalize whitespace
    code_text = re.sub(r'\s+', ' ', code_text)
    
    # Normalize operators
    code_text = re.sub(r'\s*([=+\-*/()<>{}[\],;])\s*', r'\1', code_text)
    
    return code_text.strip()

def get_ngrams(code_tokens, n):
    """Generate n-grams from a list of tokens."""
    return [tuple(code_tokens[i:i+n]) for i in range(len(code_tokens) - n + 1)]

def calculate_bleu(gen_code, ref_code, max_n=4):
    """Calculate BLEU score between generated and reference code."""
    # Normalize and tokenize code
    gen_tokens = normalize_code(gen_code).split()
    ref_tokens = normalize_code(ref_code).split()
    
    # Handle empty files
    if not gen_tokens:
        return 0.0
    if not ref_tokens:
        return 0.0
    
    # Calculate brevity penalty
    bp = min(1.0, len(gen_tokens) / len(ref_tokens))
    
    # Calculate n-gram precisions
    precisions = []
    for n in range(1, min(max_n + 1, 5)):  # 1 to 4-grams
        gen_ngrams = Counter(get_ngrams(gen_tokens, n))
        ref_ngrams = Counter(get_ngrams(ref_tokens, n))
        
        # Count matches
        matches = 0
        total = 0
        
        for ngram, count in gen_ngrams.items():
            matches += min(count, ref_ngrams[ngram])
            total += count
        
        # Calculate precision for this n
        if total > 0:
            precisions.append(matches / total)
        else:
            precisions.append(0.0)
    
    # Geometric mean of precisions, with smoothing
    if 0.0 in precisions:
        # Smoothing: replace 0 with a small value
        precisions = [max(p, 0.01) for p in precisions]
    
    # Log-average of precisions
    log_avg = sum(math.log(p) for p in precisions) / len(precisions)
    geo_mean = math.exp(log_avg)
    
    # Final BLEU score
    bleu = bp * geo_mean
    
    return bleu

def find_reference_file(operation):
    """Find an appropriate reference file for the operation."""
    # Try different patterns for the operation name
    patterns = [
        f"*{operation}*",
        f"*{operation.upper()}*",
        f"*{operation.lower()}*"
    ]
    
    for pattern in patterns:
        # Look in main directory
        for file in Path(REF_DIR).glob(f"{pattern}.cpp"):
            return str(file)
        
        # Look in subdirectories
        for file in Path(REF_DIR).glob(f"**/{pattern}.cpp"):
            return str(file)
    
    # If no specific match, find any .cpp file
    for file in Path(REF_DIR).glob("**/*.cpp"):
        return str(file)
    
    return None

def process_operation(operation):
    """Process a single operation."""
    print("=" * 60)
    print(f"CRYSTALBLEU CALCULATION for {MODEL} with {TECHNIQUE} on {operation}")
    print("=" * 60)
    
    # Set up directory paths
    gen_dir = f"generated_code/{MODEL}/{TECHNIQUE}/{operation}"
    
    # Check if generated code directory exists
    if not os.path.isdir(gen_dir):
        print(f"ERROR: Generated code directory {gen_dir} does not exist")
        return None
    
    # Find reference implementation
    ref_file = find_reference_file(operation)
    if not ref_file:
        print(f"ERROR: No reference implementation found for {operation}")
        return None
    
    ref_filename = os.path.basename(ref_file)
    print(f"Using reference implementation: {ref_filename}")
    
    # Read reference code
    with open(ref_file, 'r', encoding='utf-8', errors='ignore') as f:
        ref_code = f.read()
    
    # Initialize results
    results = {"results": {}}
    total_files = 0
    total_bleu = 0.0
    
    # Process each generated file
    for cpp_file in Path(gen_dir).glob("*.cpp"):
        if not cpp_file.is_file():
            continue
        
        filename = cpp_file.name
        print("-" * 60)
        print(f"Processing: {filename}")
        
        # Read generated code
        with open(cpp_file, 'r', encoding='utf-8', errors='ignore') as f:
            gen_code = f.read()
        
        # Calculate BLEU score
        bleu_score = calculate_bleu(gen_code, ref_code)
        total_bleu += bleu_score
        total_files += 1
        
        # Output details
        gen_tokens = normalize_code(gen_code).split()
        ref_tokens = normalize_code(ref_code).split()
        
        print(f"Tokens - Generated: {len(gen_tokens)}, Reference: {len(ref_tokens)}")
        print(f"CrystalBLEU score: {bleu_score:.4f}")
        
        # Add to results
        results["results"][filename] = {"bleu": bleu_score}
        print()
    
    # If no files were processed, return None
    if total_files == 0:
        print(f"No .cpp files found in {gen_dir}")
        return None
    
    # Calculate average BLEU score
    avg_bleu = total_bleu / total_files
    
    # Add summary metrics to results
    results["avg_bleu"] = avg_bleu
    results["total_samples"] = total_files
    
    # Save results to JSON file
    # Save results to JSON file
    
    output_file = os.path.join(RESULTS_DIR, f"crystalbleu_{operation}.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("*" * 72)
    print(f"SUMMARY: CrystalBLEU results for {MODEL} with {TECHNIQUE} on {operation}:")
    print("*" * 72)
    print(f"Average CrystalBLEU: {avg_bleu:.4f}")
    print(f"Total Files: {total_files}")
    print()
    
    # Print individual scores
    print("Individual scores:")
    for filename, data in results["results"].items():
        print(f"  - {filename}: {data['bleu']:.4f}")
    
    return avg_bleu

def main():
    """Main function to process all operations."""
    print("====================================================================")
    print(f"BATCH CRYSTALBLEU CALCULATION for {MODEL} with {TECHNIQUE}")
    print("====================================================================")
    
    # Process each operation
    results_summary = {}
    
    for operation in OPERATIONS:
        print(f"\nProcessing operation: {operation}")
        avg_bleu = process_operation(operation)
        
        if avg_bleu is not None:
            results_summary[operation] = avg_bleu
    
    # Print overall summary
    print("\n====================================================================")
    print(f"BATCH CRYSTALBLEU CALCULATION COMPLETE for {MODEL} with {TECHNIQUE}")
    print("====================================================================")
    
    if results_summary:
        print("\nOverall CrystalBLEU Scores:")
        for operation, score in results_summary.items():
            print(f"  - {operation}: {score:.4f}")
        
        # Calculate overall average
        overall_avg = sum(results_summary.values()) / len(results_summary)
        print(f"\nOverall Average CrystalBLEU: {overall_avg:.4f}")
    
    print(f"\nAll results saved to {RESULTS_DIR}/")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())