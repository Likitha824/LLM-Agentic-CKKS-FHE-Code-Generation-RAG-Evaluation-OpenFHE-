"""
Script to generate CKKS code samples using .
"""

import os
import sys
import argparse
import time
import json
import re
from pathlib import Path

# Add the project root to the path so we can import our modules
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.techniques.decoding import get_decoder
from src.utils.llm_api import get_llm_interface

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

def determine_operation_from_request(request):
    """Try to determine the operation from the request text."""
    request_lower = request.lower()
    
    # Use regex for more precise matching of operations
    if re.search(r'\b(matrix\s*multiplication|matmul|matrix)\b', request_lower):
        return "matrix_multiplication"
    elif re.search(r'\b(dot\s*product|inner\s*product|scalar\s*product)\b', request_lower):
        return "dot_product"
    elif re.search(r'\b(add|adder|addition|sum)\b', request_lower) and not re.search(r'\b(multipl|matrix|dot|product)\b', request_lower):
        return "addition"
    elif re.search(r'\b(multipl|multiplier|multiplication)\b', request_lower) and not re.search(r'\b(matrix|dot)\b', request_lower):
        return "multiplication"
    elif re.search(r'\b(convolution|convolve|conv)\b', request_lower):
        return "convolution"
    else:
        # Fallback checks for more general terms
        if "add" in request_lower or "sum" in request_lower:
            return "addition"
        elif "mult" in request_lower or "product" in request_lower:
            if "matrix" in request_lower:
                return "matrix_multiplication"
            elif "dot" in request_lower:
                return "dot_product"
            else:
                return "multiplication"
        
        return "unknown"

def generate_samples(
    llm_name, 
    operation, 
    technique="self_consistency",
    num_samples=5,  # Number of final samples you want
    candidates_per_sample=3,  # Number of candidates to generate for each final sample
    original_request=None
):
    """
    Generate multiple samples using self-consistency for each sample.
    
    Args:
        llm_name: Name of the LLM to use
        operation: CKKS operation to implement
        technique: Decoding technique to use
        num_samples: Number of final samples to generate
        candidates_per_sample: Number of candidates to generate for each final sample
        original_request: Original natural language request (if any)
    
    Returns:
        List of the final samples
    """
    # Initialize the LLM interface
    llm = get_llm_interface(llm_name)
    
    # Initialize the decoder
    decoder = get_decoder(operation, technique, num_samples=candidates_per_sample)
    
    # Create the output directory structure
    output_dir = project_root / "generated_code" / llm_name.replace("/", "_") / technique / operation
    ensure_directory(output_dir)
    
    # List to store final samples
    final_samples = []
    
    print(f"Generating {num_samples} samples for {operation} using {llm_name} with {technique}...")
    if original_request:
        print(f"Based on request: {original_request}")
    
    # Get the base prompt
    if hasattr(decoder, 'generate_prompt'):
        # For SelfConsistencyBeamSearchDecoder
        prompt = decoder.generate_prompt()
        
        # For each final sample we want
        for sample_idx in range(num_samples):
            print(f"\nGenerating final sample {sample_idx+1}/{num_samples}...")
            print(f"  Creating {candidates_per_sample} candidates for self-consistency analysis...")
            
            # Generate multiple candidates for this sample
            candidates = []
            
            # Create variations for diversity
            temperatures = [0.3, 0.4, 0.35, 0.45, 0.4]
            variations = [
                "code clarity and readability",
                "computational efficiency",
                "minimizing noise growth",
                "detailed documentation",
                "both memory efficiency and readability"
            ]
            
            # Generate candidate implementations
            for i in range(candidates_per_sample):
                print(f"    Generating candidate {i+1}/{candidates_per_sample}...")
                
                # Create a varied prompt
                temp = temperatures[i % len(temperatures)]
                focus = variations[i % len(variations)]
                candidate_prompt = prompt + f"\n\nImplement this with a focus on {focus}."
                
                # Generate the candidate
                response = llm.generate(
                    candidate_prompt, 
                    temperature=temp,
                    max_tokens=2048
                )
                
                # Extract the code
                code = extract_code_from_response(response)
                
                if code:
                    candidates.append(code)
                else:
                    print("    Failed to extract code")
                
                # Small delay between generations
                if i < candidates_per_sample - 1:
                    time.sleep(1)
            
            # If we have candidates, apply self-consistency
            if candidates:
                # For self-consistency without compilation check
                compilable_status = [True] * len(candidates)
                
                # Perform consistency analysis and selection
                most_consistent = decoder.select_most_consistent(candidates, compilable_status)
                
                # Add to final samples
                final_samples.append(most_consistent)
                
                # Save this final sample
                timestamp = int(time.time())
                sample_path = output_dir / f"sample_{sample_idx+1}_{timestamp}.cpp"
                with open(sample_path, "w") as f:
                    f.write(most_consistent)
                
                # Save metadata
                metadata_path = output_dir / f"sample_{sample_idx+1}_{timestamp}_metadata.json"
                with open(metadata_path, "w") as f:
                    json.dump({
                        "original_request": original_request,
                        "operation": operation,
                        "final_sample_index": sample_idx + 1,
                        "candidates_generated": len(candidates),
                        "llm": llm_name,
                        "technique": technique,
                        "timestamp": timestamp
                    }, f, indent=2)
                
                print(f"    Final sample {sample_idx+1} saved to: {sample_path}")
            else:
                print(f"    Could not generate candidates for sample {sample_idx+1}")
            
            # Add a delay between final samples
            if sample_idx < num_samples - 1:
                time.sleep(2)
                
    elif hasattr(decoder, 'generate_prompts'):
        # For CandidateSelectionDecoder
        for sample_idx in range(num_samples):
            print(f"\nGenerating final sample {sample_idx+1}/{num_samples}...")
            print(f"  Creating {candidates_per_sample} candidates for candidate selection...")
            
            # Get varied prompts for diversity
            prompts = decoder.generate_prompts()
            
            # Generate multiple candidates for this sample
            candidates = []
            
            # Generate candidate implementations
            temperatures = [0.3, 0.4, 0.35, 0.45, 0.4]  # Different temperatures for diversity
            
            for i, prompt in enumerate(prompts[:candidates_per_sample]):
                print(f"    Generating candidate {i+1}/{len(prompts[:candidates_per_sample])}...")
                
                # Get temperature for this generation
                temp = temperatures[i % len(temperatures)]
                
                # Generate the candidate
                response = llm.generate(
                    prompt, 
                    temperature=temp,
                    max_tokens=2048
                )
                
                # Extract the code
                code = extract_code_from_response(response)
                
                if code:
                    candidates.append(code)
                else:
                    print("    Failed to extract code")
                
                # Small delay between generations
                if i < len(prompts[:candidates_per_sample]) - 1:
                    time.sleep(1)
            
            # If we have candidates, apply candidate selection
            if candidates:
                # Select the best candidate
                best_candidate = decoder.select_best_candidate(candidates)
                
                # Add to final samples
                final_samples.append(best_candidate)
                
                # Save this final sample
                timestamp = int(time.time())
                sample_path = output_dir / f"sample_{sample_idx+1}_{timestamp}.cpp"
                with open(sample_path, "w") as f:
                    f.write(best_candidate)
                
                # Save metadata
                metadata_path = output_dir / f"sample_{sample_idx+1}_{timestamp}_metadata.json"
                with open(metadata_path, "w") as f:
                    json.dump({
                        "original_request": original_request,
                        "operation": operation,
                        "final_sample_index": sample_idx + 1,
                        "candidates_generated": len(candidates),
                        "llm": llm_name,
                        "technique": technique,
                        "timestamp": timestamp
                    }, f, indent=2)
                
                print(f"    Final sample {sample_idx+1} saved to: {sample_path}")
            else:
                print(f"    Could not generate candidates for sample {sample_idx+1}")
            
            # Add a delay between final samples
            if sample_idx < num_samples - 1:
                time.sleep(2)
    else:
        raise AttributeError(f"Decoder of type {type(decoder).__name__} does not have required prompt generation methods")
    
    return final_samples
    
def generate_all_operations(llm_name, technique="self_consistency_beam", num_samples=5, beam_width=5, candidates_per_sample=3):
    """Generate samples for all operations using a specific LLM and decoding technique."""
    operations = ["addition", "multiplication", "dot_product", "matrix_multiplication", "convolution"]
    
    for operation in operations:
        print(f"\n{'='*80}")
        print(f"Generating {num_samples} samples for {operation} using {llm_name} with {technique}")
        print(f"{'='*80}\n")
        
        generate_samples(
            llm_name,
            operation,
            technique,
            num_samples,
            candidates_per_sample
        )

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate CKKS code using decoding techniques")
    parser.add_argument("--llm", default="deepseek", choices=["deepseek", "llama", "openai", "both"], 
                        help="Name of the LLM to use (deepseek, llama, mistral, or both)")
    
    # Make request and operation mutually exclusive
    operation_group = parser.add_mutually_exclusive_group(required=True)
    operation_group.add_argument("--request", 
                      help="Natural language request (e.g., 'Generate an adder for CKKS')")
    operation_group.add_argument("--operation", choices=[
        "addition", "multiplication", "dot_product", "matrix_multiplication", "convolution", "all"
    ], help="CKKS operation to implement")
    
    parser.add_argument("--technique", default="self_consistency_beam", 
                        help="Decoding technique to use")
    parser.add_argument("--num-samples", type=int, default=5, 
                        help="Number of samples to generate")
    parser.add_argument("--beam-width", type=int, default=5, 
                        help="Width for beam search")
    parser.add_argument("--candidates-per-sample", type=int, default=3,
                        help="Number of candidates to generate for each final sample")
    
    args = parser.parse_args()
    
    # Determine operation from request if provided
    operation = args.operation
    original_request = None
    
    if args.request:
        original_request = args.request
        detected_operation = determine_operation_from_request(args.request)
        if detected_operation == "unknown":
            print(f"Could not determine operation from request: {args.request}")
            return
        operation = detected_operation
        print(f"Detected operation '{operation}' from request: {args.request}")
    
    llms_to_use = []
    if args.llm == "both":
        llms_to_use = ["deepseek", "llama","openai"]
    else:
        llms_to_use = [args.llm]
    
    for llm in llms_to_use:
        if operation == "all":
            generate_all_operations(
                llm,
                args.technique,
                args.num_samples,
                args.beam_width,
                args.candidates_per_sample
            )
        else:
            generate_samples(
                llm,
                operation,
                args.technique,
                args.num_samples,
                args.candidates_per_sample,
                original_request
            )

if __name__ == "__main__":
    main()