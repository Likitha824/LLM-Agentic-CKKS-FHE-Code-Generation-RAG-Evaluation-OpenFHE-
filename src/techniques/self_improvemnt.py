"""
Implementation of self-improvement techniques for CKKS code generation.
"""

from typing import Dict, List, Any, Optional, Tuple
import re

class IterativeSelfImprovement:
    """
    Implements iterative self-improvement for CKKS code generation.
    
    This technique involves:
    1. Generating an initial solution
    2. Self-evaluating the solution against criteria
    3. Identifying areas for improvement
    4. Refining the solution iteratively
    """
    
    def __init__(self, operation_type: str, max_iterations: int = 3):
        """
        Initialize the iterative self-improvement technique.
        
        Args:
            operation_type: Type of CKKS operation
            max_iterations: Maximum number of improvement iterations
        """
        self.operation_type = operation_type
        self.max_iterations = max_iterations
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
    
    def generate_initial_prompt(self) -> str:
        """Generate the initial prompt for code generation."""
        operation_descriptions = {
            "addition": "element-wise addition of two encrypted vectors",
            "multiplication": "element-wise multiplication of two encrypted vectors",
            "dot_product": "dot product (inner product) of two encrypted vectors",
            "matrix_multiplication": "matrix multiplication of two encrypted matrices",
            "convolution": "1D convolution of an encrypted signal with a kernel"
        }
        
        description = operation_descriptions.get(self.operation_type, "")
        
        # Add specific constraints to avoid hallucinations and common errors
        wrong_api_warnings = """
IMPORTANT: DO NOT use any of these incorrect/outdated API calls:
- DO NOT use CryptoContextFactory<DCRTPoly>::GenerateCKKSContext
- DO NOT use CKKSParameters, CKKSContext, CryptoContextCKKS
- DO NOT use context(8192, 80, 100) constructor pattern
- DO NOT use FHEcontext, FHESecretKey, FHEPubKey classes
- DO NOT use plaintext.at<double>(i) pattern
- DO NOT use operator+ or operator* directly on ciphertexts
- DO NOT use genCryptoContextCKKS function
"""

        correct_api_example = """
CORRECT API PATTERN:
```cpp
// Parameter setup - Use this EXACT pattern
CCParams<CryptoContextCKKSRNS> parameters;
parameters.SetMultiplicativeDepth(2);
parameters.SetScalingModSize(40);
parameters.SetBatchSize(8);
parameters.SetSecurityLevel(SecurityLevel::HEStd_128_classic);

CryptoContext<DCRTPoly> cryptoContext = GenCryptoContext(parameters);
cryptoContext->Enable(PKE);
cryptoContext->Enable(KEYSWITCH);
cryptoContext->Enable(LEVELEDSHE);

// Key generation
KeyPair<DCRTPoly> keyPair = cryptoContext->KeyGen();

// Data encoding and encryption
std::vector<double> vector1 = {1.0, 2.0, 3.0, 4.0};
Plaintext plaintext1 = cryptoContext->MakeCKKSPackedPlaintext(vector1);
auto ciphertext1 = cryptoContext->Encrypt(keyPair.publicKey, plaintext1);
```
"""
        
        prompt = f"""
You are an expert in Fully Homomorphic Encryption (FHE) and the Cheon-Kim-Kim-Song (CKKS) scheme.

TASK: Implement CKKS {self.operation_type} operation using OpenFHE.

OPERATION DESCRIPTION:
This operation performs {description}.

{wrong_api_warnings}

{correct_api_example}

Please generate a complete C++ program that implements this operation. The program should:
1. Include necessary headers (#include "openfhe.h")
2. Use namespace lbcrypto
3. Set up appropriate parameters for CKKS using CCParams<CryptoContextCKKSRNS>
4. Generate encryption keys
5. Perform {self.operation_type} on encrypted data using the proper EvalXXX methods
6. Decrypt and verify the results
7. Include clear comments explaining each step

Your solution should be a single, compilable C++ file with all the necessary code.
"""
        return prompt
    
    def evaluate_code(self, code: str) -> Tuple[float, List[str]]:
        """
        Evaluate the generated code against a set of criteria.
        
        Args:
            code: The code to evaluate
            
        Returns:
            Tuple of (score, list of improvement suggestions)
        """
        score = 0.0
        max_score = 10.0
        suggestions = []
        
        # Check for outdated/incorrect API usage
        incorrect_patterns = [
            (r"CryptoContextFactory<.*>::[gG]en", "Do not use CryptoContextFactory::GenCryptoContextCKKS or similar outdated API calls"),
            (r"CKKSParameters", "Do not use CKKSParameters, which is not part of the modern OpenFHE API"),
            (r"CKKSContext", "Do not use CKKSContext, which is not part of the modern OpenFHE API"),
            (r"FHEcontext|FHESecKey|FHEPubKey", "Do not use FHEcontext, FHESecKey, or FHEPubKey which are not OpenFHE classes"),
            (r"at<double>", "Do not use .at<double>() which is not a valid method for OpenFHE plaintexts"),
            (r"cipher\d+\s*\+\s*cipher", "Do not use operator+ directly on ciphertexts, use EvalAdd instead"),
            (r"cipher\d+\s*\*\s*cipher", "Do not use operator* directly on ciphertexts, use EvalMult instead")
        ]
        
        for pattern, message in incorrect_patterns:
            if re.search(pattern, code):
                score -= 1.0  # Heavy penalty for incorrect API usage
                suggestions.append(message)
        
        # Check for basic structure
        if "#include \"openfhe.h\"" not in code:
            suggestions.append("Add '#include \"openfhe.h\"' for OpenFHE.")
        else:
            score += 0.5
            
        if "namespace lbcrypto" not in code:
            suggestions.append("Add 'using namespace lbcrypto;' for OpenFHE.")
        else:
            score += 0.5
            
        if "int main" not in code:
            suggestions.append("Add a main function to demonstrate the operation.")
        else:
            score += 0.5
            
        # Check for CKKS setup
        if "CCParams<CryptoContextCKKSRNS>" not in code:
            suggestions.append("Use CCParams<CryptoContextCKKSRNS> for parameter setup.")
        else:
            score += 1.0
            
        if "GenCryptoContext" not in code:
            suggestions.append("Use GenCryptoContext to initialize the CKKS context.")
        else:
            score += 0.5
            
        if "Enable(PKE)" not in code and "Enable" not in code:
            suggestions.append("Enable the necessary features using cc->Enable().")
        else:
            score += 0.5
            
        # Check for key generation
        if "KeyGen" not in code:
            suggestions.append("Generate encryption keys using KeyGen().")
        else:
            score += 1.0
            
        if "EvalMultKeyGen" not in code and self.operation_type in ["multiplication", "dot_product", "matrix_multiplication", "convolution"]:
            suggestions.append("Generate multiplication evaluation keys using EvalMultKeyGen().")
        elif self.operation_type in ["multiplication", "dot_product", "matrix_multiplication", "convolution"]:
            score += 1.0
            
        if "EvalRotateKeyGen" not in code and self.operation_type in ["dot_product", "matrix_multiplication", "convolution"]:
            suggestions.append("Generate rotation keys for advanced operations using EvalRotateKeyGen().")
        elif self.operation_type in ["dot_product", "matrix_multiplication", "convolution"]:
            score += 1.0
            
        # Check for data preparation
        if "MakeCKKSPackedPlaintext" not in code:
            suggestions.append("Use MakeCKKSPackedPlaintext to encode data for CKKS.")
        else:
            score += 1.0
            
        if "Encrypt" not in code:
            suggestions.append("Encrypt the plaintexts using the Encrypt() method.")
        else:
            score += 0.5
            
        # Check for operation-specific code
        operation_functions = {
            "addition": "EvalAdd",
            "multiplication": "EvalMult",
            "dot_product": ["EvalInnerProduct", "EvalMult.*EvalSum"],
            "matrix_multiplication": ["EvalMatMult", "EvalMult.*EvalRotate"],
            "convolution": ["EvalConv", "EvalMult.*EvalRotate"]
        }
        
        op_func = operation_functions.get(self.operation_type)
        if isinstance(op_func, str):
            if op_func not in code:
                suggestions.append(f"Use {op_func}() to perform the {self.operation_type} operation.")
            else:
                score += 1.5
        elif isinstance(op_func, list):
            found = False
            for func in op_func:
                if re.search(func, code):
                    found = True
                    break
            if not found:
                suggestions.append(f"Implement the {self.operation_type} operation using appropriate functions.")
            else:
                score += 1.5
                
        # Check for decryption and verification
        if "Decrypt" not in code:
            suggestions.append("Decrypt the result using the Decrypt() method.")
        else:
            score += 0.5
            
        # Normalize score (ensure it's not negative due to penalties)
        normalized_score = max(0, min(10, score))
        
        return normalized_score, suggestions
    
    def generate_improvement_prompt(self, code: str, score: float, suggestions: List[str]) -> str:
        """
        Generate a prompt for improving the code based on evaluation.
        
        Args:
            code: The current code
            score: The evaluation score
            suggestions: List of improvement suggestions
            
        Returns:
            A prompt for code improvement
        """
        suggestion_text = "\n".join([f"- {suggestion}" for suggestion in suggestions])
        
        # Provide a reference for the specific operation
        reference_snippets = {
            "addition": """
// Correct pattern for addition:
auto ciphertextAdd = cryptoContext->EvalAdd(ciphertext1, ciphertext2);

// Decrypt and decode
Plaintext plaintextResult;
cryptoContext->Decrypt(keyPair.secretKey, ciphertextAdd, &plaintextResult);
plaintextResult->SetLength(vector1.size());
""",
            "multiplication": """
// Generate multiplication keys first:
cryptoContext->EvalMultKeyGen(keyPair.secretKey);

// Then multiply:
auto ciphertextMult = cryptoContext->EvalMult(ciphertext1, ciphertext2);

// Decrypt and decode
Plaintext plaintextResult;
cryptoContext->Decrypt(keyPair.secretKey, ciphertextMult, &plaintextResult);
plaintextResult->SetLength(vector1.size());
""",
            "dot_product": """
// Required setup for dot product:
cryptoContext->Enable(ADVANCEDSHE);
cryptoContext->EvalMultKeyGen(keyPair.secretKey);

// Generate rotation keys
std::vector<int> rotations;
for (int i = 1; i < 8; i *= 2) {
    rotations.push_back(i);
}
cryptoContext->EvalRotateKeyGen(keyPair.secretKey, rotations);

// Perform element-wise multiplication
auto ciphertextMult = cryptoContext->EvalMult(ciphertext1, ciphertext2);

// Sum the components using rotations
auto ciphertextSum = ciphertextMult;
for (int i = 1; i < 8; i *= 2) {
    auto rotated = cryptoContext->EvalRotate(ciphertextSum, i);
    ciphertextSum = cryptoContext->EvalAdd(ciphertextSum, rotated);
}
"""
        }
        
        reference_snippet = reference_snippets.get(self.operation_type, "")
        
        prompt = f"""
You previously generated the following CKKS {self.operation_type} implementation:

```cpp
{code}
```

I've evaluated this code and found some areas for improvement (current score: {score:.1f}/10):

{suggestion_text}

IMPORTANT: Make sure to use the modern OpenFHE API. Here is the correct pattern for {self.operation_type}:

```cpp
{reference_snippet}
```

Please provide an improved version of this code that addresses these suggestions.
Make sure the improved code is a complete, compilable C++ program implementing
CKKS {self.operation_type} using OpenFHE.
"""
        return prompt
    
    def is_improvement_needed(self, score: float, suggestions: List[str]) -> bool:
        """
        Determine if code improvement is needed.
        
        Args:
            score: Current evaluation score
            suggestions: List of improvement suggestions
            
        Returns:
            True if improvement is needed, False otherwise
        """
        # If score is very high or there are no suggestions, no improvement needed
        if score >= 9.0 or not suggestions:
            return False
        return True

def get_self_improvement(operation_type: str, technique: str = "iterative", max_iterations: int = 3):
    """Factory function to get the appropriate self-improvement technique."""
    if technique == "iterative":
        return IterativeSelfImprovement(operation_type, max_iterations)
    else:
        raise ValueError(f"Unsupported self-improvement technique: {technique}")
