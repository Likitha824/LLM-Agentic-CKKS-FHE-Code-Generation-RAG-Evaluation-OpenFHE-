"""
Script to generate CKKS code samples using enhanced prompting techniques.
Supports natural language requests that are mapped to specific operations.
"""

import os
import sys
import argparse
import time
import re
import json
from pathlib import Path

# Add the project root to the path so we can import our modules
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.utils.llm_api import get_llm_interface
from src.techniques.prompting import get_prompt_generator  # Import the prompt generator

def ensure_directory(directory_path):
    """Ensure the specified directory exists."""
    os.makedirs(directory_path, exist_ok=True)

def extract_code_from_response(response):
    """Extract C++ code from the LLM response."""
    # Look for code blocks with ```cpp or ``` markers
    import re
    
    # First try to find code blocks with explicit markers
    cpp_pattern = r"```(?:cpp)?(.*?)```"
    matches = re.findall(cpp_pattern, response, re.DOTALL)
    
    if matches:
        # Get the longest code block (assuming that's the most complete one)
        return max(matches, key=len).strip()
    
    # If no code blocks found with markdown formatting, try other patterns
    
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
    # This will be handled as an error in the evaluation phase
    return response

def determine_operation_from_request(request):
    """Try to determine the operation from the request text."""
    request_lower = request.lower()
    
    # Use more specific matching patterns
    if re.search(r'\b(matrix\s*multiplication|matmul|matrix)\b', request_lower):
        return "matrix_multiplication"
    elif re.search(r'\b(dot\s*product|inner\s*product|scalar\s*product)\b', request_lower):
        return "dot_product"
    elif re.search(r'\b(add|adder|addition|sum)\b', request_lower) and not re.search(r'\b(multipl|product)\b', request_lower):
        return "addition"
    elif re.search(r'\b(multipl|multiplier|multiplication)\b', request_lower):
        return "multiplication"
    elif re.search(r'\b(convolution|convolve|conv)\b', request_lower):
        return "convolution"
    else:
        # Check for more general terms as a fallback
        if "add" in request_lower or "sum" in request_lower:
            return "addition"
        elif "mult" in request_lower or "product" in request_lower:
            return "multiplication"
        
        return "unknown"

def generate_with_enhanced_prompt(
    llm_name,
    request=None, 
    operation_type=None,
    technique="chain_of_thought",
    temperature=0.9,
):
    """
    Generate code samples using enhanced prompting techniques.
    
    Args:
        llm_name: Name of the LLM to use
        request: Natural language request (if provided, will be used to determine operation)
        operation_type: Type of CKKS operation (addition, multiplication, etc.)
        technique: Prompting technique to use
        temperature: Temperature for generation (higher = more diverse)
    
    Returns:
        Generated code sample and path where it's saved
    """
    # Determine operation type from request if not explicitly provided
    if operation_type is None and request is not None:
        operation_type = determine_operation_from_request(request)
        if operation_type == "unknown":
            raise ValueError(f"Could not determine operation type from request: {request}")
    elif operation_type is None:
        raise ValueError("Either request or operation_type must be provided")
        
    # Initialize the LLM interface
    llm = get_llm_interface(llm_name)
    
    # Get the enhanced prompt
    prompt_generator = get_prompt_generator(operation_type, technique)
    prompt = prompt_generator.generate_prompt(temperature=temperature)
    
    # Create the output directory structure
    output_dir = project_root / "generated_code" / llm_name.replace("/", "_") / technique / operation_type
    ensure_directory(output_dir)
    
    # Generate the sample
    print(f"Generating {operation_type} code using {llm_name} with {technique} prompting...")
    if request:
        print(f"Based on request: {request}")
    
    # Generate the code
    response = llm.generate(prompt, temperature=temperature, max_tokens=4000)
    
    # Extract the code from the response
    code = extract_code_from_response(response)
    
    # Generate a timestamp to create a unique filename
    timestamp = int(time.time())
    
    # Save the sample
    sample_path = output_dir / f"sample_{timestamp}.cpp"
    with open(sample_path, "w") as f:
        f.write(code)
    
    # Also save the full response and prompt for reference
    metadata_path = output_dir / f"sample_{timestamp}_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump({
            "original_request": request,
            "determined_operation": operation_type,
            "prompt": prompt,
            "full_response": response,
            "llm": llm_name,
            "technique": technique,
            "timestamp": timestamp,
            "temperature": temperature
        }, f, indent=2)
    
    print(f"Saved sample to {sample_path}")
    
    return code, sample_path

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate CKKS code samples using enhanced prompting")
    parser.add_argument("--llm", default="deepseek", 
                        help="Name of the LLM to use (e.g., deepseek, llama, mistral)")
    
    # Make both request and operation optional, but require at least one
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--request", 
                      help="Natural language request (e.g., 'Generate an adder for CKKS')")
    group.add_argument("--operation", 
                      choices=["addition", "multiplication", "dot_product", "matrix_multiplication", "convolution"],
                      help="Type of CKKS operation to implement")
    
    parser.add_argument("--technique", default="chain_of_thought", 
                        help="Prompting technique to use")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Temperature for generation (higher = more diverse)")
    parser.add_argument("--count", type=int, default=1,
                        help="Number of samples to generate")
    
    args = parser.parse_args()
    
    # Generate multiple samples if requested
    for i in range(args.count):
        if args.count > 1:
            print(f"\nGenerating sample {i+1} of {args.count}")
        
        code, path = generate_with_enhanced_prompt(
            llm_name=args.llm,
            request=args.request,
            operation_type=args.operation,
            technique=args.technique,
            temperature=args.temperature
        )
        
        # Print a preview of the generated code
        print("\nGenerated Code:")
        print("=" * 80)
        print(code[:1000] + "..." if len(code) > 1000 else code)
        print("=" * 80)
        print(f"Full code saved to: {path}")

if __name__ == "__main__":
    main()