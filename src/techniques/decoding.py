"""
Implementation of decoding techniques for CKKS code generation.
"""

from typing import Dict, List, Any, Optional
import re

class CandidateSelectionDecoder:
    """
    Implements a decoding technique that generates multiple candidates and selects the best one.
    This uses a search-and-selection strategy: generate multiple samples and select based on predefined criteria.
    """
    
    def __init__(self, operation_type: str, num_candidates: int = 3):
        """
        Initialize the candidate selection decoder.
        
        Args:
            operation_type: Type of CKKS operation
            num_candidates: Number of candidates to generate before selection
        """
        self.operation_type = operation_type
        self.num_candidates = num_candidates
        self.allowed_operations = [
            "addition", 
            "multiplication", 
            "dot_product", 
            "matrix_multiplication", 
            "convolution"
        ]
        
        if operation_type not in self.allowed_operations:
            raise ValueError(f"Unsupported operation: {operation_type}. "
                            f"Supported operations: {', '.join(self.allowed_operations)}")
    
    def _generate_base_prompt(self) -> str:
        """Generate a base prompt for the operation."""
        base_prompts = {
            "addition": "Implement CKKS addition operation using OpenFHE library. Write a complete C++ program that adds two encrypted vectors element-wise.",
            "multiplication": "Implement CKKS multiplication operation using OpenFHE library. Write a complete C++ program that multiplies two encrypted vectors element-wise.",
            "dot_product": "Implement CKKS dot product operation using OpenFHE library. Write a complete C++ program that computes the inner product of two encrypted vectors.",
            "matrix_multiplication": "Implement CKKS matrix multiplication operation using OpenFHE library. Write a complete C++ program that multiplies two encrypted matrices.",
            "convolution": "Implement CKKS convolution operation using OpenFHE library. Write a complete C++ program that performs convolution between an encrypted signal and a kernel."
        }
        return base_prompts.get(self.operation_type, "")
    
    def _add_constraints(self, base_prompt: str, constraint_level: int) -> str:
        """Add varying constraints to the prompt to generate diverse candidates."""
        constraints = []
        
        if constraint_level == 0:
            # Basic constraints focusing on correctness
            constraints = [
                "Use the correct modern OpenFHE API with CCParams<CryptoContextCKKSRNS>",
                "Ensure all necessary library features are enabled",
                "Make sure to generate appropriate keys for the operation",
                "Include proper error handling"
            ]
        elif constraint_level == 1:
            # Medium constraints focusing on performance
            constraints = [
                "Use the correct modern OpenFHE API with CCParams<CryptoContextCKKSRNS>",
                "Optimize parameter selection for the specific operation",
                "Use descriptive variable names",
                "Include detailed comments for each step",
                "Implement robust error handling"
            ]
        else:
            # Advanced constraints focusing on best practices
            constraints = [
                "Use the correct modern OpenFHE API with CCParams<CryptoContextCKKSRNS>",
                "Optimize the implementation for both performance and memory usage",
                "Follow best practices for homomorphic encryption",
                "Use a consistent naming convention throughout",
                "Include comprehensive error handling and validation",
                "Minimize the noise growth in the ciphertexts"
            ]
        
        # Add operation-specific API constraints to ensure correct generation
        api_constraints = []
        if self.operation_type == "addition":
            api_constraints = [
                "Use cryptoContext->EvalAdd() for addition, NOT the + operator",
                "Set appropriate multiplicative depth (at least 1)",
                "Verify that vectors have compatible dimensions"
            ]
        elif self.operation_type == "multiplication":
            api_constraints = [
                "Generate multiplication evaluation keys with cryptoContext->EvalMultKeyGen()",
                "Use cryptoContext->EvalMult() for multiplication, NOT the * operator",
                "Set appropriate multiplicative depth (at least 2)",
                "Consider the noise growth in multiplication operations"
            ]
        elif self.operation_type == "dot_product":
            api_constraints = [
                "Generate both multiplication keys AND rotation keys",
                "Enable ADVANCEDSHE for rotation operations",
                "Perform element-wise multiplication followed by summation",
                "Use a tree-based approach for efficient summation",
                "Consider that the final result will be in the first slot"
            ]
        elif self.operation_type == "matrix_multiplication":
            api_constraints = [
                "Generate both multiplication keys AND rotation keys",
                "Enable ADVANCEDSHE for rotation operations",
                "Choose an appropriate matrix encoding strategy",
                "Consider the computational complexity of different approaches",
                "Verify matrix dimension compatibility"
            ]
        elif self.operation_type == "convolution":
            api_constraints = [
                "Generate both multiplication keys AND rotation keys for all kernel positions",
                "Enable ADVANCEDSHE for rotation operations",
                "Handle boundary conditions appropriately",
                "Consider the most efficient approach for kernel application"
            ]
        
        # Combine general and API-specific constraints
        constraints.extend(api_constraints)
        example_guidance = """
Below is a minimal skeleton illustrating the correct OpenFHE API pattern for CKKS:

```cpp
// Parameter setup
CCParams<CryptoContextCKKSRNS> parameters;
parameters.SetMultiplicativeDepth(/* e.g., 2 or more */);
parameters.SetScalingModSize(/* e.g., 40 or 50 */);
parameters.SetBatchSize(/* e.g., 8 */);
parameters.SetSecurityLevel(SecurityLevel::HEStd_128_classic);

auto cryptoContext = GenCryptoContext(parameters);
cryptoContext->Enable(PKE);
cryptoContext->Enable(KEYSWITCH);
cryptoContext->Enable(LEVELEDSHE);

// Key generation
auto keyPair = cryptoContext->KeyGen();

// If multiplication is required
cryptoContext->EvalMultKeyGen(keyPair.secretKey);

// If rotations are required (e.g., for dot product, matrix multiplication, convolution)
cryptoContext->Enable(ADVANCEDSHE);
std::vector<int> rotations = { /* e.g., 1, 2, 4 */ };
cryptoContext->EvalRotateKeyGen(keyPair.secretKey, rotations);

// Encoding and encryption
std::vector<double> inputVector = { /* e.g., 1.0, 2.0, 3.0, 4.0 */ };
Plaintext plaintext = cryptoContext->MakeCKKSPackedPlaintext(inputVector);
auto ciphertext = cryptoContext->Encrypt(keyPair.publicKey, plaintext);

DO NOT use these incorrect patterns:
❌ CryptoContextFactory<DCRTPoly>::GenerateCKKSContext
❌ FHEcontext, FHESecretKey, FHEPubKey
❌ CKKSParameters, CKKSContext
❌ Using operators like + or * directly on ciphertexts
❌ Encrypting without encoding first
"""
        
        # Add constraints to prompt
        constraint_text = "\n".join([f"- {constraint}" for constraint in constraints])
        if constraint_level == 0:
            return f"{base_prompt}\n\nPlease follow these API requirements:\n{constraint_text}\n{example_guidance}"
        elif constraint_level == 1:
            return f"{base_prompt}\n\nPlease optimize the implementation and follow these requirements:\n{constraint_text}\n{example_guidance}"
        else:
            return f"{base_prompt}\n\nPlease implement an optimized solution following these advanced requirements:\n{constraint_text}\n{example_guidance}"
    

        
    def generate_prompts(self) -> List[str]:
        """
        Generate multiple diverse prompts to create candidate solutions.
        
        Returns:
            List of prompts to generate different candidate solutions
        """
        base_prompt = self._generate_base_prompt()
        prompts = []
        
        # Generate diverse prompts with different constraints
        for i in range(self.num_candidates):
            constraint_level = i % 3  # Cycle through constraint levels
            prompt = self._add_constraints(base_prompt, constraint_level)
            prompts.append(prompt)
        
        return prompts
    
    def select_best_candidate(self, candidates: List[str]) -> str:
        """
        Select the best candidate solution based on predefined criteria.
        
        Args:
            candidates: List of candidate code solutions
            
        Returns:
            The selected best candidate
        """
        # If we have fewer than expected candidates, return the first one
        if not candidates:
            return ""
        if len(candidates) == 1:
            return candidates[0]
        
        # Score each candidate based on various criteria
        candidate_scores = []
        
        for code in candidates:
            score = 0
            
            # 1. Check for presence of key components
            if "#include" in code and "openfhe.h" in code:
                score += 1
            if "namespace lbcrypto" in code:
                score += 1
            if "int main()" in code:
                score += 1
                
            # 2. Check for correct CKKS-specific code
            if "CCParams<CryptoContextCKKSRNS>" in code:
                score += 2
            elif "CryptoContextFactory" in code:
                score -= 2  # Penalize old API
                
            if "GenCryptoContext" in code:
                score += 2
            if "KeyGen()" in code:
                score += 1
            if "MakeCKKSPackedPlaintext" in code:
                score += 2
                
            # 3. Check for operation-specific code and correct API usage
            if self.operation_type == "addition":
                if "EvalAdd" in code:
                    score += 3
                if "+" in code and "ciphertext" in code.lower():
                    score -= 2  # Penalize using + operator
            
            elif self.operation_type == "multiplication":
                if "EvalMultKeyGen" in code:
                    score += 2
                if "EvalMult" in code:
                    score += 3
                if "*" in code and "ciphertext" in code.lower():
                    score -= 2  # Penalize using * operator
            
            elif self.operation_type == "dot_product":
                if "EvalMultKeyGen" in code:
                    score += 2
                if "EvalRotateKeyGen" in code:
                    score += 2
                if "EvalMult" in code:
                    score += 2
                if "EvalRotate" in code or "EvalSum" in code:
                    score += 3
            
            elif self.operation_type == "matrix_multiplication":
                if "EvalMultKeyGen" in code:
                    score += 2
                if "EvalRotateKeyGen" in code:
                    score += 2
                if "EvalMult" in code and "EvalRotate" in code:
                    score += 3
            
            elif self.operation_type == "convolution":
                if "EvalMultKeyGen" in code:
                    score += 2
                if "EvalRotateKeyGen" in code:
                    score += 2
                if "EvalRotate" in code and "EvalMult" in code:
                    score += 3
                
            # 4. Check for proper decryption and verification
            if "Decrypt" in code:
                score += 1
            if "SetLength" in code:
                score += 1
                
            # 5. Check for error handling
            if "try" in code and "catch" in code:
                score += 1
                
            # 6. Check for comments and documentation
            comment_count = len(re.findall(r'//.*|/\*[\s\S]*?\*/', code))
            score += min(comment_count / 5, 2)  # Cap at 2 points for comments
            
            # 7. Penalize for obvious errors or placeholders
            if "TODO" in code or "FIXME" in code:
                score -= 2
            if "Your code here" in code or "Insert code" in code:
                score -= 2
            
            # 8. Penalize for overly short code (likely incomplete)
            if len(code.split('\n')) < 20:
                score -= 3
                
            candidate_scores.append(score)
        
        # Find the candidate with the highest score
        if not candidate_scores:
            return candidates[0] if candidates else ""
            
        best_index = candidate_scores.index(max(candidate_scores))
        return candidates[best_index]

class SelfConsistencyBeamSearchDecoder:
    """
    Implements true self-consistency decoding with beam search.
    Generates multiple samples using beam search, then selects based on consistency across samples.
    """
    
    def __init__(self, operation_type: str, num_samples: int = 5, beam_width: int = 5):
        """
        Initialize the self-consistency beam search decoder.
        
        Args:
            operation_type: Type of CKKS operation
            num_samples: Number of samples to generate for self-consistency
            beam_width: Width of beam for beam search
        """
        self.operation_type = operation_type
        self.num_samples = num_samples
        self.beam_width = beam_width
        self.allowed_operations = [
            "addition", 
            "multiplication", 
            "dot_product", 
            "matrix_multiplication", 
            "convolution"
        ]
        
        if operation_type not in self.allowed_operations:
            raise ValueError(f"Unsupported operation: {operation_type}. "
                           f"Supported operations: {', '.join(self.allowed_operations)}")
    
    def generate_prompt(self) -> str:
        """
        Generate a prompt optimized for self-consistency.
        
        Returns:
            Complete prompt string
        """
        # Reuse the CandidateSelectionDecoder's base prompt generator
        candidate_decoder = CandidateSelectionDecoder(self.operation_type)
        base_prompt = candidate_decoder._generate_base_prompt()
        return candidate_decoder._add_constraints(base_prompt, 1)
    
    def extract_code(self, text: str) -> Optional[str]:
        """Extract C++ code from generated text."""
        # Look for code between triple backticks
        cpp_pattern = r"```cpp\s*(.*?)\s*```"
        cpp_matches = re.findall(cpp_pattern, text, re.DOTALL)
        
        if cpp_matches:
            return cpp_matches[0]
        
        # Also try without language specifier
        code_pattern = r"```\s*(.*?)\s*```"
        code_matches = re.findall(code_pattern, text, re.DOTALL)
        
        if code_matches:
            # Check if it looks like C++ code
            if "#include" in code_matches[0] and "int main" in code_matches[0]:
                return code_matches[0]
        
        # If no code blocks found, try to extract based on common patterns
        if "#include" in text and "int main" in text:
            lines = text.split('\n')
            start_idx = -1
            end_idx = -1
            
            for i, line in enumerate(lines):
                if "#include" in line and start_idx == -1:
                    start_idx = i
                if start_idx != -1 and "return 0;" in line:
                    # Look for closing brace after return 0
                    for j in range(i, min(i+5, len(lines))):
                        if "}" in lines[j]:
                            end_idx = j
                            break
            
            if start_idx != -1 and end_idx != -1:
                return "\n".join(lines[start_idx:end_idx+1])
        
        return None
    
    def get_generation_params(self) -> Dict[str, Any]:
        """
        Get generation parameters for the LLM.
        This includes beam search parameters.
        
        Returns:
            Dictionary of generation parameters
        """
        return {
            "do_sample": False,     # Use beam search, not sampling
            "num_beams": self.beam_width,  # Use provided beam width
            "num_return_sequences": 1,  # Return best sequence from each beam search
            "max_tokens": 2048      # Adjust as needed
        }
    
    def analyze_consistency(self, samples: List[str]) -> Dict[str, Any]:
        """
        Analyze consistency patterns across multiple code samples.
        
        Args:
            samples: List of code samples to analyze
            
        Returns:
            Dictionary with consistency analysis results
        """
        # Count occurrences of key API patterns
        api_patterns = {
            "parameter_setup": r"CCParams\s*<\s*CryptoContextCKKSRNS\s*>\s*\w+",
            "context_creation": r"GenCryptoContext\s*\(\s*\w+\s*\)",
            "security_level": r"SecurityLevel::(\w+)",
            "multiplicative_depth": r"SetMultiplicativeDepth\s*\(\s*(\d+)\s*\)",
            "batch_size": r"SetBatchSize\s*\(\s*(\d+)\s*\)",
            "scaling_factor": r"SetScalingModSize\s*\(\s*(\d+)\s*\)",
            "enable_features": r"Enable\s*\(\s*(\w+)\s*\)",
            "key_generation": r"KeyGen\s*\(\s*\)",
            "mult_key_gen": r"EvalMultKeyGen\s*\(\s*\w+\.secretKey\s*\)",
            "rotate_key_gen": r"EvalRotateKeyGen\s*\(\s*\w+\.secretKey\s*,\s*\{([^}]*)\}\s*\)",
            "encryption": r"Encrypt\s*\(\s*\w+\.publicKey\s*,\s*\w+\s*\)"
        }
        
        # Operation-specific patterns
        if self.operation_type == "addition":
            api_patterns["operation"] = r"EvalAdd\s*\(\s*\w+\s*,\s*\w+\s*\)"
        elif self.operation_type == "multiplication":
            api_patterns["operation"] = r"EvalMult\s*\(\s*\w+\s*,\s*\w+\s*\)"
        elif self.operation_type == "dot_product":
            api_patterns["operation"] = r"EvalRotate|EvalSum|EvalMult"
        elif self.operation_type == "matrix_multiplication":
            api_patterns["operation"] = r"EvalRotate.*EvalMult"
        elif self.operation_type == "convolution":
            api_patterns["operation"] = r"EvalRotate.*EvalMult"
            
        # Count pattern occurrences and extract values
        pattern_counts = {pattern: 0 for pattern in api_patterns}
        pattern_values = {pattern: [] for pattern in api_patterns}
        
        for sample in samples:
            for pattern_name, pattern in api_patterns.items():
                matches = re.findall(pattern, sample)
                if matches:
                    pattern_counts[pattern_name] += 1
                    if isinstance(matches[0], tuple):
                        pattern_values[pattern_name].extend(matches[0])
                    else:
                        pattern_values[pattern_name].extend(matches)
        
        # Find consensus patterns (most common implementation approaches)
        consensus = {}
        
        # For numeric parameters, find the most common value
        for param in ["multiplicative_depth", "batch_size", "scaling_factor"]:
            if pattern_values[param]:
                # Convert to integers and find most common
                try:
                    values = [int(v) for v in pattern_values[param]]
                    if values:
                        consensus[param] = max(set(values), key=values.count)
                except (ValueError, TypeError):
                    pass
        
        # For security level, find most common
        if pattern_values["security_level"]:
            consensus["security_level"] = max(set(pattern_values["security_level"]), key=pattern_values["security_level"].count)
        
        # Calculate consistency score for each sample
        consistency_scores = []
        
        for sample in samples:
            score = 0
            
            # Check if sample uses the consensus parameter values
            for param, value in consensus.items():
                pattern = f"{param}.*{value}"
                if re.search(pattern, sample):
                    score += 1
            
            # Check if sample follows the most common API patterns
            for pattern, count in pattern_counts.items():
                if count > len(samples) / 2:  # More than half the samples use this pattern
                    if re.search(api_patterns[pattern], sample):
                        score += 1
            
            consistency_scores.append(score)
        
        return {
            "pattern_counts": pattern_counts,
            "consensus": consensus,
            "consistency_scores": consistency_scores
        }
    
    def select_most_consistent(self, samples: List[str], compilable_samples: List[bool]) -> str:
        """
        Select the most consistent sample that is also compilable.
        
        Args:
            samples: List of code samples
            compilable_samples: List of booleans indicating if each sample compiles
            
        Returns:
            The selected most consistent sample
        """
        if not samples:
            return ""
            
        # Get only the compilable samples
        valid_samples = [samples[i] for i in range(len(samples)) if compilable_samples[i]]
        
        # If no samples compile, return the first sample (best effort)
        if not valid_samples:
            return samples[0]
            
        # If only one sample compiles, return it
        if len(valid_samples) == 1:
            return valid_samples[0]
            
        # Analyze consistency across compilable samples
        consistency_analysis = self.analyze_consistency(valid_samples)
        
        # Get the consistency scores
        scores = consistency_analysis["consistency_scores"]
        
        # Return the sample with the highest consistency score
        best_index = scores.index(max(scores))
        return valid_samples[best_index]

def get_decoder(operation_type: str, technique: str = "candidate_selection", **kwargs):
    """Factory function to get the appropriate decoder."""
    if technique == "candidate_selection":
        num_candidates = kwargs.get("num_candidates", 3)
        return CandidateSelectionDecoder(operation_type, num_candidates)
    elif technique == "self_consistency_beam":
        num_samples = kwargs.get("num_samples", 5)
        beam_width = kwargs.get("beam_width", 5)
        return SelfConsistencyBeamSearchDecoder(operation_type, num_samples, beam_width)
    else:
        raise ValueError(f"Unsupported decoding technique: {technique}")