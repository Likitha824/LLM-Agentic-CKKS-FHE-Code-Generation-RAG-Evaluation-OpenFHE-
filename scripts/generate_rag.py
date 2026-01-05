"""
Script to generate CKKS code samples using Retrieval-Augmented Generation (RAG).
"""

import os
import sys
import argparse
import time
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import re

# Add the project root to the path so we can import our modules
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.techniques.rag import get_rag_technique
from src.utils.llm_api import get_llm_interface

def ensure_directory(directory_path):
    """Ensure the specified directory exists."""
    os.makedirs(directory_path, exist_ok=True)

def extract_code_from_response(response):
    """Extract C++ code from the LLM response."""
    # Look for code blocks with ```cpp or ``` markers
    # First try to find code blocks with explicit markers
    cpp_pattern = r"```(?:cpp)?(.*?)```"
    matches = re.findall(cpp_pattern, response, re.DOTALL)
    
    if matches:
        # Get the longest code block (assuming that's the most complete one)
        longest_match = max(matches, key=len).strip()
        
        # Check if it contains the essential components of a complete C++ program
        if "#include" in longest_match and "int main" in longest_match:
            return longest_match
        
        # If the longest match doesn't look like a complete program, 
        # search for a match that contains both include and main
        for match in matches:
            if "#include" in match and "int main" in match:
                return match.strip()
        
        # If still no suitable match, return the longest one
        return longest_match
    
    # If no code blocks found with markdown formatting, try other patterns
    
    # Look for a complete C++ program structure
    complete_pattern = r"(#include.*?int\s+main.*?return\s+0;.*?\})"
    complete_match = re.search(complete_pattern, response, re.DOTALL)
    if complete_match:
        return complete_match.group(1).strip()
    
    # 1. Look for #include lines as the start of code
    include_pattern = r"(#include.*?(?:\n|$).*)"
    include_match = re.search(include_pattern, response, re.DOTALL)
    if include_match:
        return include_match.group(1).strip()
    
    # 2. Look for blocks that have typical C++ patterns and keywords
    cpp_keywords = [
        r"int\s+main\s*\(", 
        r"using\s+namespace", 
        r"#define", 
        r"auto\s+[\w_]+\s*=", 
        r"std::", 
        r"class\s+\w+",
        r"void\s+\w+\s*\("
    ]
    
    for keyword in cpp_keywords:
        keyword_match = re.search(f"({keyword}.*)", response, re.DOTALL)
        if keyword_match:
            return keyword_match.group(1).strip()
    
    # 3. If all else fails, check if the entire response looks like code
    # by checking if most lines start with typical C++ syntax
    lines = response.strip().split('\n')
    code_line_indicators = [
        r'^\s*#', r'^\s*\w+\s*\(', r'^\s*\w+\s*=', r'^\s*if\s*\(', 
        r'^\s*for\s*\(', r'^\s*while', r'^\s*return', r'^\s*\{', r'^\s*\}'
    ]
    
    code_line_count = 0
    for line in lines:
        for indicator in code_line_indicators:
            if re.match(indicator, line):
                code_line_count += 1
                break
    
    # If more than 30% of lines look like code, assume it's all code
    if len(lines) > 0 and code_line_count / len(lines) > 0.3:
        return response.strip()
    
    # If we couldn't find any code patterns, return the original response
    return response

def validate_code(code, operation):
    """
    Validate the generated code for common errors.
    
    Args:
        code: The generated code
        operation: The operation type (addition, multiplication, etc.)
        
    Returns:
        tuple: (is_valid, issues_list)
    """
    issues = []
    
    # Check for essential includes
    if "#include \"openfhe.h\"" not in code and "#include <openfhe.h>" not in code:
        issues.append("Missing OpenFHE header include")
    
    # Check for namespace
    if "using namespace lbcrypto" not in code:
        issues.append("Missing 'using namespace lbcrypto'")
    
    # Check for main function
    if "int main" not in code:
        issues.append("Missing main function")
    
    # Check for proper crypto context setup
    if "CryptoContext<DCRTPoly>" not in code:
        issues.append("Missing CryptoContext setup")
    
    if "Enable(PKE)" not in code or "Enable(KEYSWITCH)" not in code or "Enable(LEVELEDSHE)" not in code:
        issues.append("Missing one or more Enable calls (PKE, KEYSWITCH, LEVELEDSHE)")
    
    # Check for ADVANCEDSHE in operations that require rotations
    if operation in ["dot_product", "matrix_multiplication", "convolution"]:
        if "Enable(ADVANCEDSHE)" not in code:
            issues.append("Missing Enable(ADVANCEDSHE) for operations requiring rotations")
    
    # Check for key generation
    if "KeyGen" not in code:
        issues.append("Missing key generation")
    
    # Check for known API usage errors based on compilation failures
    if "SetLength(vector" in code or "SetLength(ciphertext" in code:
        issues.append("Invalid SetLength usage - should be called on plaintext objects, not vectors or crypto context")
    
    # Check for correct rotation keys generation
    if operation in ["dot_product", "matrix_multiplication", "convolution"]:
        if "EvalRotateKeyGen" in code:
            # Check if rotation keys are passed with integers directly
            if re.search(r"EvalRotateKeyGen\s*\(\s*[0-9]", code):
                issues.append("EvalRotateKeyGen should take secretKey as first parameter, not integers")
            # Check if indices are missing when using integers
            if "EvalRotateKeyGen" in code and not (re.search(r"EvalRotateKeyGen.*vector", code) or "EvalSumKeyGen" in code):
                issues.append("EvalRotateKeyGen should either use vector of indices or use EvalSumKeyGen")
    
    # Check for incorrect inner product usage
    if operation == "dot_product" and "EvalInnerProduct" in code:
        if not re.search(r"EvalInnerProduct\([^,]*,[^,]*,[^,]*\)", code):
            issues.append("EvalInnerProduct requires three parameters (ciphertext1, ciphertext2, batchSize)")
    
    # Check for EvalSum with incorrect parameters
    if "EvalSum" in code:
        if re.search(r"EvalSum\s*\([^,]+,\s*[a-zA-Z]", code) and not re.search(r"EvalSum\s*\([^,]+,\s*[0-9]", code):
            issues.append("EvalSum's second parameter should be a size_t/int value (batchSize), not a ciphertext")
    
    # Check for EvalMultConstant which doesn't exist in OpenFHE
    if "EvalMultConstant" in code:
        issues.append("EvalMultConstant doesn't exist in OpenFHE - use EvalMult with scalar parameter instead")
    
    # Check for EvalRelinearize which doesn't exist in some OpenFHE versions
    if "EvalRelinearize" in code:
        issues.append("EvalRelinearize might not exist in your OpenFHE version - use Relinearize instead")
    
    # Check for incorrect EvalMult initialization with non-ciphertext
    if re.search(r"EvalMult\s*\(\s*[0-9\.]+", code):
        issues.append("EvalMult's first parameter must be a ciphertext, not a scalar")
    
    # Operation-specific checks
    if operation in ["multiplication", "dot_product", "matrix_multiplication", "convolution"]:
        if "EvalMultKeyGen" not in code:
            issues.append("Missing EvalMultKeyGen for multiplication operation")
    
    if operation in ["dot_product", "matrix_multiplication", "convolution"]:
        if "EvalRotateKeyGen" not in code and "EvalSumKeyGen" not in code:
            issues.append("Missing rotation key generation for operation requiring rotations")
    
    # Check for proper operation implementation
    operation_api_map = {
        "addition": "EvalAdd",
        "multiplication": "EvalMult",
        "dot_product": ["EvalInnerProduct", "EvalSum"],
        "matrix_multiplication": ["EvalMult", "EvalRotate"],
        "convolution": ["EvalMult", "EvalRotate"]
    }
    
    api_functions = operation_api_map.get(operation, [])
    if isinstance(api_functions, str):
        api_functions = [api_functions]
    
    missing_apis = [api for api in api_functions if api not in code]
    if missing_apis:
        issues.append(f"Missing required API functions for {operation}: {', '.join(missing_apis)}")
    
    # Check for decryption
    if "Decrypt" not in code:
        issues.append("Missing decryption step")
    
    # Check for verification
    if "Expected" not in code and "expected" not in code:
        issues.append("Missing result verification")
    
    return len(issues) == 0, issues

def generate_samples(
    llm_name,
    operation,
    rag_technique="basic",
    num_samples=5,
    knowledge_base_dir=None,
    vector_index_path=None
):
    llm = get_llm_interface(llm_name)
    # pass vector_index_path into kwargs if vector
    rav_kwargs = {}
    if rag_technique == "vector" and vector_index_path:
        rav_kwargs["index_path"] = vector_index_path

    rag = get_rag_technique(
        operation, rag_technique, knowledge_base_dir, **rav_kwargs
    )

    output_dir = Path("generated_code") / llm_name.replace("/", "_") / rag_technique / operation
    ensure_directory(output_dir)

    for i in range(num_samples):
        print(f"Sample {i+1}/{num_samples}: {operation} with {rag_technique} RAG")
        prompt = rag.generate_rag_prompt()
        response = llm.generate(prompt, temperature=0.7, max_tokens=4000)
        code = extract_code_from_response(response)

        # validation + retry logic remains unchanged
        is_valid, issues = validate_code(code, operation)
        retries = 0
        while not is_valid and retries < 2:
            guidance = "Please fix:\n" + "\n".join(f"- {issue}" for issue in issues)
            retry_prompt = prompt + "\n\n" + guidance
            response = llm.generate(retry_prompt, temperature=0.5, max_tokens=4000)
            code = extract_code_from_response(response)
            is_valid, issues = validate_code(code, operation)
            retries += 1

        # save code + metadata
        sample_path = output_dir / f"sample_{i+1}.cpp"
        with open(sample_path, "w") as f:
            f.write(code)

        meta = {
            "prompt": prompt,
            "response_preview": response[:500],
            "issues": issues if not is_valid else [],
            "llm": llm_name,
            "operation": operation,
            "technique": rag_technique,
            "timestamp": time.time()
        }
        with open(output_dir / f"sample_{i+1}_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        print(f"Saved {sample_path}")
        if not is_valid:
            print(f"WARNING: code issues: {issues}")
        if i < num_samples - 1:
            time.sleep(1)


def generate_all_operations(
    llm_name,
    rag_technique="basic",
    num_samples=5,
    knowledge_base_dir=None
):
    """Generate samples for all operations using a specific LLM and RAG technique."""
    operations = ["addition", "multiplication", "dot_product", "matrix_multiplication", "convolution"]
    
    for operation in operations:
        print(f"\n{'='*80}")
        print(f"Generating {num_samples} samples for {operation} using {llm_name} with {rag_technique} RAG")
        print(f"{'='*80}\n")
        
        generate_samples(
            llm_name,
            operation,
            rag_technique,
            num_samples,
            knowledge_base_dir
        )

def main():
    parser = argparse.ArgumentParser(description="Generate CKKS code using RAG")
    parser.add_argument("--llm", default="deepseek", choices=["deepseek","llama","gemini","openai","both"])
    parser.add_argument("--operation", choices=[
        "addition","multiplication","dot_product","matrix_multiplication","convolution","all"
    ], default="all")
    parser.add_argument("--technique", default="basic", choices=["basic","graph","vector"],
                        help="RAG technique to use")
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--knowledge-base-dir", type=str, default=None,
                        help="Where your JSON docs live")
    parser.add_argument("--vector-index-path", type=str, default=None,
                        help="Optional path to read/write FAISS index for vector RAG")

    args = parser.parse_args()

    llms = ["deepseek","llama"] if args.llm=="both" else [args.llm]
    operations = ["addition","multiplication","dot_product","matrix_multiplication","convolution"] \
                 if args.operation=="all" else [args.operation]

    for llm in llms:
        for op in operations:
            generate_samples(
                llm,
                op,
                rag_technique=args.technique,
                num_samples=args.num_samples,
                knowledge_base_dir=args.knowledge_base_dir,
                vector_index_path=args.vector_index_path
            )

if __name__ == "__main__":
    main()