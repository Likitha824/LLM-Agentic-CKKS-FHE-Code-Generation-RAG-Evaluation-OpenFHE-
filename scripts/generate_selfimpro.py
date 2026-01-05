"""
Script to generate CKKS code samples using the iterative self-improvement technique.
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

from src.techniques.self_improvement import get_self_improvement
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

def determine_operation_from_request(request):
    """Try to determine the operation from the request text."""
    request_lower = request.lower()
    
    # Use regex for more precise matching of operations
    if re.search(r'\b(matrix\s*multipl|matmul)\b', request_lower):
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

def generate_with_self_improvement(
    llm_name, 
    operation, 
    technique="iterative", 
    max_iterations=3,
    original_request=None,
    count=1  # New parameter to generate multiple code samples
):
    """
    Generate code using the iterative self-improvement technique.
    
    Args:
        llm_name: Name of the LLM to use
        operation: CKKS operation to implement
        technique: Self-improvement technique to use
        max_iterations: Maximum number of improvement iterations
        original_request: Original natural language request (if any)
        count: Number of code samples to generate
        
    Returns:
        List of tuples (final code, iterations, score, output_path)
    """
    # Initialize the LLM interface
    llm = get_llm_interface(llm_name)
    
    # Initialize the self-improvement technique
    self_improvement = get_self_improvement(operation, technique, max_iterations)
    
    # Create the output directory structure
    output_dir = project_root / "generated_code" / llm_name.replace("/", "_") / f"self_improvement_{technique}" / operation
    ensure_directory(output_dir)
    
    # Results list to store multiple generated codes
    results = []
    
    # Generate multiple code samples
    for sample_index in range(count):
        # Generate timestamp for this sample
        timestamp = int(time.time()) + sample_index
        
        print(f"\nGenerating code sample {sample_index + 1} of {count} for {operation} using {llm_name}...")
        if original_request:
            print(f"Based on request: {original_request}")
            
        initial_prompt = self_improvement.generate_initial_prompt()
        response = llm.generate(initial_prompt, temperature=0.7, max_tokens=4000)  # Slightly higher temperature for diversity
        current_code = extract_code_from_response(response)
        
        # Save the initial code
        initial_path = output_dir / f"sample_{timestamp}_iteration_0.cpp"
        with open(initial_path, "w") as f:
            f.write(current_code)
        
        # Track iterations
        iterations = [
            {
                "iteration": 0,
                "code": current_code,
                "prompt": initial_prompt,
                "score": 0.0,
                "suggestions": []
            }
        ]
        
        # Evaluate and improve iteratively
        current_iteration = 0
        max_iteration = max_iterations
        
        while current_iteration < max_iteration:
            # Evaluate current code
            print(f"Evaluating iteration {current_iteration}...")
            score, suggestions = self_improvement.evaluate_code(current_code)
            
            # Update the last iteration with evaluation results
            iterations[-1]["score"] = score
            iterations[-1]["suggestions"] = suggestions
            
            print(f"  Score: {score:.2f}/10")
            print(f"  Suggestions: {len(suggestions)}")
            
            # Check if improvement is needed
            if not self_improvement.is_improvement_needed(score, suggestions):
                print("No further improvements needed!")
                break
                
            # Generate improvement prompt
            improvement_prompt = self_improvement.generate_improvement_prompt(
                current_code, score, suggestions
            )
            
            # Generate improved code
            current_iteration += 1
            print(f"Generating improvement iteration {current_iteration}...")
            
            response = llm.generate(improvement_prompt, temperature=0.7, max_tokens=4000)
            improved_code = extract_code_from_response(response)
            
            # Save the improved code
            improved_path = output_dir / f"sample_{timestamp}_iteration_{current_iteration}.cpp"
            with open(improved_path, "w") as f:
                f.write(improved_code)
            
            # Track this iteration
            iterations.append({
                "iteration": current_iteration,
                "code": improved_code,
                "prompt": improvement_prompt,
                "score": 0.0,
                "suggestions": []
            })
            
            # Update current code for next iteration
            current_code = improved_code
            
            # Short delay to avoid rate limits
            time.sleep(2)
        
        # Perform final evaluation
        final_score, final_suggestions = self_improvement.evaluate_code(current_code)
        iterations[-1]["score"] = final_score
        iterations[-1]["suggestions"] = final_suggestions
        
        print(f"Final score after {current_iteration} iterations: {final_score:.2f}/10")
        
        # Save metadata about the iterations
        metadata_path = output_dir / f"sample_{timestamp}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump({
                "original_request": original_request,
                "llm": llm_name,
                "operation": operation,
                "technique": f"self_improvement_{technique}",
                "max_iterations": max_iterations,
                "actual_iterations": current_iteration,
                "final_score": final_score,
                "iterations": [
                    {
                        "iteration": it["iteration"],
                        "score": it["score"],
                        "suggestions": it["suggestions"],
                        # Don't include the full code in metadata to keep it manageable
                        "code_length": len(it["code"])
                    }
                    for it in iterations
                ],
                "timestamp": timestamp
            }, f, indent=2)
        
        # Final output path is the last iteration
        final_path = output_dir / f"sample_{timestamp}_final.cpp"
        with open(final_path, "w") as f:
            f.write(current_code)
            
        print(f"Generated code saved to {final_path}")
        
        # Add to results
        results.append((current_code, current_iteration, final_score, final_path))
        
        # Add a delay between generating multiple samples to ensure unique timestamps
        time.sleep(2)
    
    # Print summary of generated samples
    print("\nGenerated Samples Summary:")
    print(f"{'Sample':<10} {'Iterations':<12} {'Final Score':<12}")
    print("-" * 40)
    for i, (_, iterations, score, _) in enumerate(results, 1):
        print(f"{f'Sample {i}':<10} {iterations:<12} {score:<12.2f}")
    
    return results

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate CKKS code using self-improvement")
    parser.add_argument("--llm", default="deepseek", choices=["deepseek", "llama", "openai", "mistral", "both"], 
                        help="Name of the LLM to use (deepseek, llama, mistral, or both)")
    
    # Make request and operation mutually exclusive
    operation_group = parser.add_mutually_exclusive_group(required=True)
    operation_group.add_argument("--request", 
                      help="Natural language request (e.g., 'Generate an adder for CKKS')")
    operation_group.add_argument("--operation", choices=[
        "addition", "multiplication", "dot_product", "matrix_multiplication", "convolution", "all"
    ], help="CKKS operation to implement")
    
    parser.add_argument("--technique", default="iterative", 
                        help="Self-improvement technique to use")
    parser.add_argument("--max-iterations", type=int, default=3, 
                        help="Maximum number of improvement iterations")
    parser.add_argument("--count", type=int, default=5, 
                        help="Number of code samples to generate")
    
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
        llms_to_use = ["deepseek", "llama"]
    else:
        llms_to_use = [args.llm]
    
    all_results = []
    for llm in llms_to_use:
        if operation == "all":
            results = generate_all_operations(
                llm,
                args.technique,
                args.max_iterations,
                args.count
            )
        else:
            results = generate_with_self_improvement(
                llm,
                operation,
                args.technique,
                args.max_iterations,
                original_request,
                args.count
            )
        all_results.extend(results)
    
    return all_results

# You'll need to modify generate_all_operations to handle the count parameter
def generate_all_operations(llm_name, technique="iterative", max_iterations=3, count=1):
    """Generate samples for all operations using self-improvement."""
    operations = ["addition", "multiplication", "dot_product", "matrix_multiplication", "convolution"]
    all_results = []
    
    for operation in operations:
        print(f"\n{'='*80}")
        print(f"Generating {count} code sample(s) for {operation} using {llm_name} with self-improvement")
        print(f"{'='*80}\n")
        
        results = generate_with_self_improvement(
            llm_name,
            operation,
            technique,
            max_iterations,
            count=count
        )
        
        all_results.extend(results)
        
    return all_results


if __name__ == "__main__":
    main()
