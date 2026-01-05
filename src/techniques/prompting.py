"""
Implementation of Chain-of-Thought (CoT) prompting technique for CKKS code generation
with explicit API guidance to ensure proper usage.
"""

from typing import Dict, List, Any, Optional
import os
import re
import random

class ChainOfThoughtPrompt:
    """Chain-of-Thought prompting for CKKS operations with explicit API guidance."""
    
    def __init__(self, operation_type: str):
        """
        Initialize the CoT prompt generator.
        
        Args:
            operation_type: Type of CKKS operation (addition, multiplication, dot_product, 
                           matrix_multiplication, convolution)
        """
        self.operation_type = operation_type
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
    
    def _get_operation_description(self) -> str:
        """Get detailed description for the specific operation."""
        descriptions = {
            "addition": "element-wise addition of two encrypted vectors, preserving homomorphic properties",
            "multiplication": "element-wise multiplication of two encrypted vectors, which requires multiplication keys",
            "dot_product": "dot product (inner product) of two encrypted vectors, combining element-wise multiplication with summation",
            "matrix_multiplication": "matrix multiplication of two encrypted matrices, requiring complex encoding and rotation operations",
            "convolution": "1D convolution of an encrypted signal with a plaintext or encrypted kernel"
        }
        return descriptions.get(self.operation_type, "")
    
    def _get_critical_api_details(self) -> List[str]:
        """Get critical API details to avoid common mistakes."""
        common_details = [
            "Use CCParams<CryptoContextCKKSRNS> for parameter setup, NOT older APIs",
            "Security level must be specified as an enum (e.g., SecurityLevel::HEStd_128_classic), NOT as an integer",
            "Use GenCryptoContext(parameters) to create the context, NOT CryptoContextFactory::Create()",
            "After decryption, use GetRealPackedValue() to access results, NOT GetRealPackedVector()",
            "Set the length of the result before displaying: result->SetLength(batchSize)"
        ]
        
        operation_specific = {
            "addition": [
                "Use cc->EvalAdd() for addition, NEVER use the + operator on ciphertexts"
            ],
            "multiplication": [
                "MUST generate multiplication keys with cc->EvalMultKeyGen(keys.secretKey) before performing multiplications",
                "Use cc->EvalMult() for multiplication, NEVER use the * operator on ciphertexts"
            ],
            "dot_product": [
                "MUST generate both multiplication keys (EvalMultKeyGen) AND rotation keys (EvalRotateKeyGen)",
                "Use a proper summation approach with rotations to compute the final dot product",
                "The result will be in the first slot of the resulting ciphertext"
            ],
            "matrix_multiplication": [
                "MUST generate both multiplication keys (EvalMultKeyGen) AND rotation keys (EvalRotateKeyGen)",
                "Properly flatten matrices in row-major or column-major order",
                "Use the correct rotation indices based on matrix dimensions"
            ],
            "convolution": [
                "MUST generate both multiplication keys (EvalMultKeyGen) AND rotation keys (EvalRotateKeyGen)",
                "Generate rotation keys for ALL required rotation indices",
                "Use cc->EvalRotate() followed by cc->EvalMult() and cc->EvalAdd() for each kernel position"
            ]
        }
        
        return common_details + operation_specific.get(self.operation_type, [])
    
    def _get_valid_function_calls(self) -> str:
        """Get a list of valid OpenFHE function calls to help prevent hallucinated functions."""
        return """
VALID FUNCTION CALLS - Reference these instead of inventing functions:

namespace
using namespace lbcrypto;

libraries
#include "openfhe.h"


Context/Parameter Setup:
- CCParams<CryptoContextCKKSRNS> parameters;
- parameters.SetMultiplicativeDepth(int depth);
- parameters.SetScalingModSize(int bitSize);
- parameters.SetBatchSize(int batchSize);
- parameters.SetSecurityLevel(SecurityLevel::HEStd_128_classic);
- auto cc = GenCryptoContext(parameters);

Features:
- cc->Enable(PKE);
- cc->Enable(KEYSWITCH);
- cc->Enable(LEVELEDSHE);

Key Generation:
- auto keys = cc->KeyGen();
- cc->EvalMultKeyGen(keys.secretKey);
- cc->EvalRotateKeyGen(keys.secretKey, {indices});

Encoding/Encryption:
- auto plaintext = cc->MakeCKKSPackedPlaintext(vector, ...);
- auto ciphertext = cc->Encrypt(keys.publicKey, plaintext);

Operations:
- auto result = cc->EvalAdd(cipher1, cipher2);
- auto result = cc->EvalSub(cipher1, cipher2);
- auto result = cc->EvalMult(cipher1, cipher2);
- auto result = cc->EvalMult(cipher, scalar);
- auto result = cc->EvalRotate(cipher, index);
- auto result = cc->EvalSum(cipher, batchSize);

Decryption:
- Plaintext result;
- cc->Decrypt(keys.secretKey, cipher, &result);
- result->SetLength(batchSize);
- std::vector<double> values = result->GetRealPackedValue();
"""
    
    def _get_correct_api_examples(self) -> str:
        """Get explicit examples of correct vs incorrect API usage."""
        return """
Here are examples of correct vs. incorrect API usage:

CORRECT:
- parameters.SetSecurityLevel(SecurityLevel::HEStd_128_classic);  // Uses enum
- auto cc = GenCryptoContext(parameters);  // Correct context creation
- auto result = cc->EvalAdd(cipher1, cipher2);  // Correct addition
- std::vector<double> decrypted = result->GetRealPackedValue();  // Correct method name

INCORRECT:
- parameters.SetSecurityLevel(128);  // ERROR: Using integer instead of enum
- auto cc = CryptoContextFactory<CryptoContextCKKSRNS>::Create(parameters);  // ERROR: Old API
- auto result = cipher1 + cipher2;  // ERROR: Cannot use operators directly
- std::vector<double> decrypted = result->GetRealPackedVector();  // ERROR: Wrong method name
"""
    
    def _get_implementation_steps(self) -> List[str]:
        """Get detailed implementation steps for the operation."""
        common_steps = [
            "Set up the CKKS crypto context with appropriate parameters:",
            "   - Set multiplicative depth (at least 1 for addition, 2+ for other operations)",
            "   - Set scaling factor bit-length (e.g., 50)",
            "   - Set batch size (e.g., 8)",
            "   - Set security level using the ENUM value: SecurityLevel::HEStd_128_classic",
            
            "Generate the crypto context using GenCryptoContext(parameters)",
            
            "Enable the required features:",
            "   - Enable PKE",
            "   - Enable KEYSWITCH",
            "   - Enable LEVELEDSHE",
            
            "Generate key pair using cc->KeyGen()"
        ]
        
        operation_steps = {
            "addition": [
                "Define two vectors of doubles for input",
                "Encode vectors as plaintexts using cc->MakeCKKSPackedPlaintext()",
                "Encrypt plaintexts using cc->Encrypt()",
                "Perform homomorphic addition using cc->EvalAdd()",
                "Decrypt the result using cc->Decrypt()",
                "Extract and display the result using result->SetLength() and result->GetRealPackedValue()"
            ],
            "multiplication": [
                "Generate multiplication evaluation keys with cc->EvalMultKeyGen(keys.secretKey)",
                "Define two vectors of doubles for input",
                "Encode vectors as plaintexts using cc->MakeCKKSPackedPlaintext()",
                "Encrypt plaintexts using cc->Encrypt()",
                "Perform homomorphic multiplication using cc->EvalMult()",
                "Decrypt the result using cc->Decrypt()",
                "Extract and display the result using result->SetLength() and result->GetRealPackedValue()"
            ],
            "dot_product": [
                "Generate multiplication evaluation keys with cc->EvalMultKeyGen(keys.secretKey)",
                "Generate rotation keys for all needed rotations with cc->EvalRotateKeyGen(keys.secretKey, {...})",
                "Define two vectors of doubles for input",
                "Encode vectors as plaintexts using cc->MakeCKKSPackedPlaintext()",
                "Encrypt plaintexts using cc->Encrypt()",
                "Perform element-wise multiplication using cc->EvalMult()",
                "Implement a summation algorithm using cc->EvalRotate() and cc->EvalAdd()",
                "Decrypt the result using cc->Decrypt()",
                "Extract the dot product from the first slot using result->SetLength(1) and result->GetRealPackedValue()[0]"
            ],
            "matrix_multiplication": [
                "Generate multiplication evaluation keys with cc->EvalMultKeyGen(keys.secretKey)",
                "Generate rotation keys for all needed indices with cc->EvalRotateKeyGen(keys.secretKey, {...})",
                "Define two matrices (as vectors of vectors) for input",
                "Flatten matrices into 1D vectors for CKKS encoding",
                "Encode vectors as plaintexts using cc->MakeCKKSPackedPlaintext()",
                "Encrypt plaintexts using cc->Encrypt()",
                "Implement matrix multiplication using cc->EvalMult(), cc->EvalRotate(), and cc->EvalAdd()",
                "Decrypt the result using cc->Decrypt()",
                "Reshape the 1D result back into the result matrix",
                "Display the result matrix"
            ],
            "convolution": [
                "Generate multiplication evaluation keys with cc->EvalMultKeyGen(keys.secretKey)",
                "Generate rotation keys for the required shifts with cc->EvalRotateKeyGen(keys.secretKey, {...})",
                "Define signal vector and kernel vector for input",
                "Encode the signal as plaintext using cc->MakeCKKSPackedPlaintext()",
                "Encrypt the signal using cc->Encrypt()",
                "For each position in kernel:",
                "   - Rotate the encrypted signal using cc->EvalRotate()",
                "   - Multiply by kernel coefficient using cc->EvalMult() with a scalar or plaintext",
                "   - Add to the result using cc->EvalAdd()",
                "Decrypt the final result using cc->Decrypt()",
                "Extract and display the result using result->SetLength() and result->GetRealPackedValue()"
            ]
        }
        
        return common_steps + operation_steps.get(self.operation_type, [])
    
    def _get_debugging_guidance(self) -> str:
        """Get guidance on debugging and error handling for CKKS implementations."""
        return """
DEBUGGING AND ERROR HANDLING:

Common compilation errors and solutions:
1. "invalid conversion from 'int' to 'lbcrypto::SecurityLevel'":
   Solution: Use SecurityLevel::HEStd_128_classic instead of 128

2. "'Create' is not a member of 'lbcrypto::CryptoContextFactory<...>'":
   Solution: Use GenCryptoContext(parameters) instead

3. "has no member named 'GetRealPackedVector'":
   Solution: Use GetRealPackedValue() instead

4. "undefined reference to `lbcrypto::CryptoContextImpl<...>::EvalMultKeyGen(...)":
   Solution: Make sure you've enabled KEYSWITCH with cc->Enable(KEYSWITCH)

Runtime error checking:
1. Check if vectors lengths match the batch size
2. Verify the rotation indices are within valid range
3. Set appropriate multiplicative depth (higher for complex operations)
4. Add print statements to verify intermediate results
5.5."All multi-line comments must be formatted using the block comment style for example 
/*
 * This is a well-formatted comment.
 * Each line should begin with an asterisk.
 */ 
where each line inside the comment starts with an asterisk"
"""
    
    
    def _get_efficient_algorithms(self) -> str:
        """Get guidance on efficient algorithms for complex CKKS operations."""
        efficient_algorithms = {
            "dot_product": """
EFFICIENT DOT PRODUCT IMPLEMENTATION:
The most efficient way to compute a dot product in CKKS is:

1. First perform element-wise multiplication:
   auto cMult = cc->EvalMult(c1, c2);

2. Then use a tree-based approach for summation:
   // For a vector of size 8, we need log_2(8) = 3 steps
   auto cSum = cMult;
   for (int i = 1; i <= batchSize/2; i *= 2) {
       auto cRot = cc->EvalRotate(cSum, i);
       cSum = cc->EvalAdd(cSum, cRot);
   }
   // Result is in the first slot
""",
            "matrix_multiplication": """
EFFICIENT MATRIX MULTIPLICATION IMPLEMENTATION:
For matrices A (m×n) and B (n×p), implemented as:

For matrices A (m×n) and B (n×p), consider the following strategy:
1. Encode matrix A in row-major order and arrange B appropriately.
2. Pre-compute rotation keys for all indices 0...n-1.
3. For each row of A and column of B, compute the dot product:
   - Multiply corresponding elements using cc->EvalMult().
   - Use iterative rotations with cc->EvalRotate() and additions with cc->EvalAdd() (e.g., baby-step giant-step) to sum products.
4. Store the dot product result and reshape the final output into the result matrix.

Example pseudocode for m×n * n×p matrix multiplication:
// Pre-compute rotation keys for all indices 0...n-1
std::vector<int> rotations;
for (int i = 0; i < n; i++) rotations.push_back(i);
cc->EvalRotateKeyGen(keys.secretKey, rotations);

// For each row of result matrix
for (int i = 0; i < m; i++) {
    // Extract row i of A
    auto a_i = extract_row(A_encrypted, i);
    
    // For each column j of B
    for (int j = 0; j < p; j++) {
        // Extract column j of B
        auto b_j = extract_column(B_encrypted, j);
        
        // Compute dot product of row i of A with column j of B
        auto c_ij = compute_dot_product(a_i, b_j);
        
        // Store in result matrix
        result[i][j] = c_ij;
    }
}
""",
            "convolution": """
EFFICIENT CONVOLUTION IMPLEMENTATION:
For a signal x and a kernel h of sizes N and M:
1. Encode the signal and encrypt it.
2. Pre-compute rotation keys for all kernel positions.
3. For each kernel position:
   - Rotate the encrypted signal using cc->EvalRotate().
   - Multiply by the kernel coefficient using cc->EvalMult().
   - Sum all contributions using cc->EvalAdd().

Example pseudocode for 1D convolution:
// Initialize result to zero
auto result = cc->EvalMult(signal, 0.0);

// For each kernel position
for (int i = 0; i < kernelSize; i++) {
    // Rotate signal to align with kernel position
    auto rotated = cc->EvalRotate(signal, i);
    
    // Multiply by kernel coefficient
    auto weighted = cc->EvalMult(rotated, kernelCoeff[i]);
    
    // Add to result
    result = cc->EvalAdd(result, weighted);
}
"""
        }
        
        return efficient_algorithms.get(self.operation_type, "")
    
    def generate_prompt(self, temperature: float = 0.7) -> str:
        """
        Generate the Chain-of-Thought prompt for the specified operation.
        
        Args:
            temperature: Controls randomness in generation (higher = more diverse outputs)
        
        Returns:
            Complete prompt string
        """
        description = self._get_operation_description()
        critical_api = self._get_critical_api_details()
        valid_functions = self._get_valid_function_calls()
        api_examples = self._get_correct_api_examples()
        steps = self._get_implementation_steps()
        debugging = self._get_debugging_guidance()
        efficient_algo = self._get_efficient_algorithms()
        
        # Introduce some variation in prompts at higher temperatures
        if temperature > 0.5:
            variations = [
                f"Implement CKKS {self.operation_type} for OpenFHE with a focus on correctness and efficiency",
                f"Create a fully functional OpenFHE implementation of CKKS {self.operation_type}",
                f"Develop a robust implementation of {self.operation_type} using CKKS in OpenFHE",
                f"Write a correct version of CKKS {self.operation_type} with OpenFHE",
                f"Implement a properly functioning CKKS {self.operation_type} operation using OpenFHE"
            ]
            task_description = random.choice(variations)
        else:
            task_description = f"Implement CKKS {self.operation_type} operation using OpenFHE"

        # Construct the prompt
        prompt = f"""
You are an expert in Fully Homomorphic Encryption (FHE) and the Cheon-Kim-Kim-Song (CKKS) scheme.

TASK: {task_description}

OPERATION DESCRIPTION:
This operation performs {description}.

CRITICAL API DETAILS - READ CAREFULLY:
{chr(10).join("- " + detail for detail in critical_api)}



{valid_functions}

{api_examples}

{efficient_algo}

{debugging}

IMPLEMENTATION STEPS:
{chr(10).join(f"{i+1}. {step}" for i, step in enumerate(steps))}


Now, implement a complete C++ program for CKKS {self.operation_type} using OpenFHE.
The program should:
1. Include necessary and appropriate headers necessary for the program
2. Set up appropriate parameters for CKKS
3. Follow all the implementation steps above
4. Include clear comments explaining each step
5. Avoid ALL the common API mistakes highlighted above

Your solution must be a single, compilable C++ file with all the necessary code.
"""
        return prompt

def get_prompt_generator(operation_type: str, technique: str = "chain_of_thought"):
    """Factory function to get the appropriate prompt generator."""
    if technique == "chain_of_thought":
        return ChainOfThoughtPrompt(operation_type)
    else:
        raise ValueError(f"Unsupported prompting technique: {technique}")