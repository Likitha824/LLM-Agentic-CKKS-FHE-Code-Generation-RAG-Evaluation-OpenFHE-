"""
Repository of CKKS code examples for different operations and frameworks.
This file contains example implementations that can be imported by the prompting module.
"""

# Dictionary of code examples organized by framework and operation
OPENFHE_EXAMPLES = {
    "addition": """// This is a simplified example of CKKS addition in OpenFHE
#include "openfhe.h"
#include <iostream>
#include <vector>

using namespace lbcrypto;

int main() {
    // Set up crypto parameters
    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetMultiplicativeDepth(1);
    parameters.SetScalingModSize(30);
    parameters.SetBatchSize(8);
    parameters.SetSecurityLevel(SecurityLevel::HEStd_128_classic);
    
    CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);
    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    
    // Generate keys
    KeyPair<DCRTPoly> keyPair = cc->KeyGen();
    
    // Define input vectors
    std::vector<double> vec1 = {1.0, 2.0, 3.0, 4.0};
    std::vector<double> vec2 = {5.0, 6.0, 7.0, 8.0};
    
    // Encode and encrypt
    Plaintext plaintext1 = cc->MakeCKKSPackedPlaintext(vec1);
    Plaintext plaintext2 = cc->MakeCKKSPackedPlaintext(vec2);
    
    auto ciphertext1 = cc->Encrypt(keyPair.publicKey, plaintext1);
    auto ciphertext2 = cc->Encrypt(keyPair.publicKey, plaintext2);
    
    // Perform addition
    auto ciphertextAdd = cc->EvalAdd(ciphertext1, ciphertext2);
    
    // Decrypt result
    Plaintext plaintextResult;
    cc->Decrypt(keyPair.secretKey, ciphertextAdd, &plaintextResult);
    
    // Note: Call SetLength on the plaintext, not on the crypto context
    plaintextResult->SetLength(vec1.size());
    
    // Get vector result
    std::vector<double> result = plaintextResult->GetRealPackedValue();
    
    // Calculate expected result
    std::vector<double> expectedResult;
    for (size_t i = 0; i < vec1.size(); i++) {
        expectedResult.push_back(vec1[i] + vec2[i]);
    }
    
    // Print and verify results
    std::cout << "Input 1: ";
    for (const auto& val : vec1) std::cout << val << " ";
    std::cout << std::endl;
    
    std::cout << "Input 2: ";
    for (const auto& val : vec2) std::cout << val << " ";
    std::cout << std::endl;
    
    std::cout << "Result: ";
    for (const auto& val : result) std::cout << val << " ";
    std::cout << std::endl;
    
    std::cout << "Expected: ";
    for (const auto& val : expectedResult) std::cout << val << " ";
    std::cout << std::endl;
    
    return 0;
}""",
    "multiplication": """// This is a simplified example of CKKS multiplication in OpenFHE
#include "openfhe.h"
#include <iostream>
#include <vector>

using namespace lbcrypto;

int main() {
    // Set up crypto parameters
    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetMultiplicativeDepth(2);  // Need higher depth for multiplication
    parameters.SetScalingModSize(40);
    parameters.SetBatchSize(8);
    parameters.SetSecurityLevel(SecurityLevel::HEStd_128_classic);
    
    CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);
    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    
    // Generate keys
    KeyPair<DCRTPoly> keyPair = cc->KeyGen();
    
    // We need multiplication keys - IMPORTANT!
    cc->EvalMultKeyGen(keyPair.secretKey);
    
    // Define input vectors
    std::vector<double> vec1 = {1.0, 2.0, 3.0, 4.0};
    std::vector<double> vec2 = {5.0, 6.0, 7.0, 8.0};
    
    // Encode and encrypt
    Plaintext plaintext1 = cc->MakeCKKSPackedPlaintext(vec1);
    Plaintext plaintext2 = cc->MakeCKKSPackedPlaintext(vec2);
    
    auto ciphertext1 = cc->Encrypt(keyPair.publicKey, plaintext1);
    auto ciphertext2 = cc->Encrypt(keyPair.publicKey, plaintext2);
    
    // Perform multiplication
    auto ciphertextMult = cc->EvalMult(ciphertext1, ciphertext2);
    
    // Scalar multiplication uses the same function with different parameter types
    double scalar = 2.5;
    auto ciphertextScalarMult = cc->EvalMult(ciphertext1, scalar);  // Note this is NOT EvalMultConstant!
    
    // Decrypt result
    Plaintext plaintextResult;
    cc->Decrypt(keyPair.secretKey, ciphertextMult, &plaintextResult);
    plaintextResult->SetLength(vec1.size());
    
    // Get vector result
    std::vector<double> result = plaintextResult->GetRealPackedValue();
    
    // Calculate expected result
    std::vector<double> expectedResult;
    for (size_t i = 0; i < vec1.size(); i++) {
        expectedResult.push_back(vec1[i] * vec2[i]);
    }
    
    // Print and verify results
    std::cout << "Result: ";
    for (const auto& val : result) std::cout << val << " ";
    std::cout << std::endl;
    
    std::cout << "Expected: ";
    for (const auto& val : expectedResult) std::cout << val << " ";
    std::cout << std::endl;
    
    return 0;
}""",
    "dot_product": """// This is a simplified example of CKKS dot product in OpenFHE
#include "openfhe.h"
#include <iostream>
#include <vector>

using namespace lbcrypto;

int main() {
    // Set up crypto parameters
    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetMultiplicativeDepth(2);
    parameters.SetScalingModSize(40);
    parameters.SetBatchSize(8);  // Must be at least as large as vector size
    parameters.SetSecurityLevel(SecurityLevel::HEStd_128_classic);
    
    CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);
    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);  // For rotation operations
    
    // Generate keys
    KeyPair<DCRTPoly> keyPair = cc->KeyGen();
    
    // Need both multiplication and rotation keys
    cc->EvalMultKeyGen(keyPair.secretKey);
    
    // Generate rotation keys needed for EvalSum
    // Either use specific indices or use EvalSumKeyGen
    cc->EvalSumKeyGen(keyPair.secretKey);
    
    // Define input vectors
    std::vector<double> vec1 = {1.0, 2.0, 3.0, 4.0};
    std::vector<double> vec2 = {5.0, 6.0, 7.0, 8.0};
    size_t vectorSize = vec1.size();
    
    // Encode and encrypt
    Plaintext plaintext1 = cc->MakeCKKSPackedPlaintext(vec1);
    Plaintext plaintext2 = cc->MakeCKKSPackedPlaintext(vec2);
    
    auto ciphertext1 = cc->Encrypt(keyPair.publicKey, plaintext1);
    auto ciphertext2 = cc->Encrypt(keyPair.publicKey, plaintext2);
    
    // Perform dot product
    // Method 1: Using EvalInnerProduct (requires 3 parameters including batchSize)
    auto ciphertextDotProduct = cc->EvalInnerProduct(ciphertext1, ciphertext2, vectorSize);
    
    // Method 2: Manual dot product
    auto ciphertextMult = cc->EvalMult(ciphertext1, ciphertext2);  // Element-wise multiplication
    auto ciphertextSum = cc->EvalSum(ciphertextMult, vectorSize);  // Sum all elements
    
    // Decrypt result
    Plaintext plaintextResult;
    cc->Decrypt(keyPair.secretKey, ciphertextDotProduct, &plaintextResult);
    plaintextResult->SetLength(1);  // Dot product is a single value
    
    // Get result
    std::vector<double> result = plaintextResult->GetRealPackedValue();
    
    // Calculate expected result
    double expectedResult = 0;
    for (size_t i = 0; i < vec1.size(); i++) {
        expectedResult += vec1[i] * vec2[i];
    }
    
    // Print and verify results
    std::cout << "Dot product result: " << result[0] << std::endl;
    std::cout << "Expected result: " << expectedResult << std::endl;
    
    return 0;
}""",
    "matrix_multiplication": """// This is a simplified example of CKKS matrix multiplication in OpenFHE
#include "openfhe.h"
#include <iostream>
#include <vector>

using namespace lbcrypto;

// Helper function to print a matrix
void printMatrix(const std::vector<std::vector<double>>& mat, const std::string& name) {
    std::cout << name << ":" << std::endl;
    for (const auto& row : mat) {
        for (const auto& val : row) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }
}

// Helper function to flatten a matrix into a vector
std::vector<double> flattenMatrix(const std::vector<std::vector<double>>& mat) {
    std::vector<double> result;
    for (const auto& row : mat) {
        result.insert(result.end(), row.begin(), row.end());
    }
    return result;
}

int main() {
    // Set up crypto parameters
    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetMultiplicativeDepth(3);
    parameters.SetScalingModSize(40);
    parameters.SetBatchSize(16);  // Must be large enough for the flattened matrices
    parameters.SetSecurityLevel(SecurityLevel::HEStd_128_classic);
    
    CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);
    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);  // For rotation operations
    
    // Generate keys
    KeyPair<DCRTPoly> keyPair = cc->KeyGen();
    
    // Need both multiplication and rotation keys
    cc->EvalMultKeyGen(keyPair.secretKey);
    
    // Generate rotation keys
    // Here we generate keys for all possible rotations we might need
    std::vector<int> rotationIndices;
    for (int i = 1; i < 8; i++) {
        rotationIndices.push_back(i);
        rotationIndices.push_back(-i);
    }
    cc->EvalRotateKeyGen(keyPair.secretKey, rotationIndices);
    
    // Define input matrices
    std::vector<std::vector<double>> matrixA = {
        {1.0, 2.0},
        {3.0, 4.0}
    };
    std::vector<std::vector<double>> matrixB = {
        {5.0, 6.0},
        {7.0, 8.0}
    };
    
    // Get matrix dimensions
    size_t rowsA = matrixA.size();
    size_t colsA = matrixA[0].size();
    size_t rowsB = matrixB.size();
    size_t colsB = matrixB[0].size();
    
    // Flatten matrices for encoding
    std::vector<double> flatA = flattenMatrix(matrixA);
    std::vector<double> flatB = flattenMatrix(matrixB);
    
    // Encode and encrypt
    Plaintext plaintextA = cc->MakeCKKSPackedPlaintext(flatA);
    auto ciphertextA = cc->Encrypt(keyPair.publicKey, plaintextA);
    
    // For efficiency, we'll use matrix B as plaintext
    Plaintext plaintextB = cc->MakeCKKSPackedPlaintext(flatB);
    
    // Matrix multiplication implementation
    // Here's a simplified approach that works for small matrices
    
    // Initialize result matrix (C = A * B)
    std::vector<std::vector<double>> matrixC(rowsA, std::vector<double>(colsB, 0.0));
    
    // Compute matrix multiplication (plaintext for verification)
    for (size_t i = 0; i < rowsA; i++) {
        for (size_t j = 0; j < colsB; j++) {
            for (size_t k = 0; k < colsA; k++) {
                matrixC[i][j] += matrixA[i][k] * matrixB[k][j];
            }
        }
    }
    
    // Print input and expected output
    printMatrix(matrixA, "Matrix A");
    printMatrix(matrixB, "Matrix B");
    printMatrix(matrixC, "Expected Result (A * B)");
    
    // Here's the homomorphic matrix multiplication
    // This is a simplified approach for pedagogical purposes
    
    // Create masks for the dot products
    std::vector<Ciphertext<DCRTPoly>> resultElements;
    
    for (size_t i = 0; i < rowsA; i++) {
        for (size_t j = 0; j < colsB; j++) {
            // Create a mask for the dot product of row i of A with column j of B
            std::vector<double> mask(flatA.size(), 0.0);
            
            for (size_t k = 0; k < colsA; k++) {
                // Position to extract from matrix A (row i, column k)
                mask[i * colsA + k] = matrixB[k][j];
            }
            
            // Create plaintext mask
            Plaintext maskPlaintext = cc->MakeCKKSPackedPlaintext(mask);
            
            // Multiply ciphertext with mask and sum elements
            auto elementProduct = cc->EvalMult(ciphertextA, maskPlaintext);
            auto elementSum = cc->EvalSum(elementProduct, colsA);
            
            resultElements.push_back(elementSum);
        }
    }
    
    // Decrypt and verify results
    std::vector<std::vector<double>> resultMatrix(rowsA, std::vector<double>(colsB));
    for (size_t i = 0; i < rowsA; i++) {
        for (size_t j = 0; j < colsB; j++) {
            Plaintext decrypted;
            cc->Decrypt(keyPair.secretKey, resultElements[i * colsB + j], &decrypted);
            decrypted->SetLength(1);
            
            std::vector<double> value = decrypted->GetRealPackedValue();
            resultMatrix[i][j] = value[0];
        }
    }
    
    // Print results
    printMatrix(resultMatrix, "Homomorphic Result (A * B)");
    
    return 0;
}""",
    "convolution": """// This is a simplified example of CKKS convolution in OpenFHE
#include "openfhe.h"
#include <iostream>
#include <vector>

using namespace lbcrypto;

// Helper function to compute plaintext convolution
std::vector<double> computeConvolution(const std::vector<double>& signal, 
                                       const std::vector<double>& kernel) {
    size_t signalSize = signal.size();
    size_t kernelSize = kernel.size();
    size_t resultSize = signalSize + kernelSize - 1;
    
    std::vector<double> result(resultSize, 0.0);
    
    for (size_t i = 0; i < resultSize; i++) {
        for (size_t j = 0; j < kernelSize; j++) {
            if (i >= j && i - j < signalSize) {
                result[i] += signal[i - j] * kernel[j];
            }
        }
    }
    
    return result;
}

int main() {
    // Set up crypto parameters
    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetMultiplicativeDepth(2);
    parameters.SetScalingModSize(40);
    parameters.SetBatchSize(16);  // Must be large enough for the convolution result
    parameters.SetSecurityLevel(SecurityLevel::HEStd_128_classic);
    
    CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);
    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);  // For rotation operations
    
    // Generate keys
    KeyPair<DCRTPoly> keyPair = cc->KeyGen();
    
    // Need both multiplication and rotation keys
    cc->EvalMultKeyGen(keyPair.secretKey);
    
    // Generate rotation keys
    std::vector<int> rotationIndices;
    for (int i = -4; i <= 4; i++) {
        if (i != 0) rotationIndices.push_back(i);
    }
    cc->EvalRotateKeyGen(keyPair.secretKey, rotationIndices);
    
    // Define input signal and kernel
    std::vector<double> signal = {1.0, 2.0, 3.0, 4.0};
    std::vector<double> kernel = {0.5, 0.5};
    
    size_t signalSize = signal.size();
    size_t kernelSize = kernel.size();
    size_t resultSize = signalSize + kernelSize - 1;
    
    // Compute expected result
    std::vector<double> expectedResult = computeConvolution(signal, kernel);
    
    // Print inputs and expected output
    std::cout << "Signal: ";
    for (const auto& val : signal) std::cout << val << " ";
    std::cout << std::endl;
    
    std::cout << "Kernel: ";
    for (const auto& val : kernel) std::cout << val << " ";
    std::cout << std::endl;
    
    std::cout << "Expected Convolution: ";
    for (const auto& val : expectedResult) std::cout << val << " ";
    std::cout << std::endl;
    
    // Encode and encrypt the signal
    Plaintext plaintextSignal = cc->MakeCKKSPackedPlaintext(signal);
    auto ciphertextSignal = cc->Encrypt(keyPair.publicKey, plaintextSignal);
    
    // Initialize result vector (for storing results)
    std::vector<Ciphertext<DCRTPoly>> convolutionResult(resultSize);
    
    // Zero out the initial result
    // IMPORTANT: We do NOT use EvalMultConstant or initialize with a non-ciphertext
    // We need a ciphertext initialized to zero, so we multiply by zero
    auto zeroPlaintext = cc->MakeCKKSPackedPlaintext({0.0});
    auto ciphertextZero = cc->Encrypt(keyPair.publicKey, zeroPlaintext);
    
    // Initialize all elements to zero
    for (size_t i = 0; i < resultSize; i++) {
        convolutionResult[i] = ciphertextZero;
    }
    
    // Perform convolution
    for (size_t i = 0; i < kernelSize; i++) {
        // Create a plaintext for the kernel coefficient
        std::vector<double> kernelCoeff(signalSize, kernel[i]);
        Plaintext kernelPlaintext = cc->MakeCKKSPackedPlaintext(kernelCoeff);
        
        // Rotate the signal
        auto rotatedSignal = i == 0 ? ciphertextSignal : cc->EvalRotate(ciphertextSignal, -i);
        
        // Multiply by kernel coefficient
        auto product = cc->EvalMult(rotatedSignal, kernelPlaintext);
        
        // Add to the appropriate positions in the result
        for (size_t j = 0; j < signalSize; j++) {
            if (i + j < resultSize) {
                convolutionResult[i + j] = cc->EvalAdd(convolutionResult[i + j], product);
            }
        }
    }
    
    // Decrypt and verify results
    std::vector<double> result(resultSize);
    for (size_t i = 0; i < resultSize; i++) {
        Plaintext decrypted;
        cc->Decrypt(keyPair.secretKey, convolutionResult[i], &decrypted);
        decrypted->SetLength(1);
        
        std::vector<double> value = decrypted->GetRealPackedValue();
        result[i] = value[0];
    }
    
    // Print results
    std::cout << "Homomorphic Convolution: ";
    for (const auto& val : result) std::cout << val << " ";
    std::cout << std::endl;
    
    return 0;
}"""
}

# Add examples for other frameworks (SEAL, HELib) as needed
SEAL_EXAMPLES = {
    # SEAL examples will go here
}

HELIB_EXAMPLES = {
    # HELib examples will go here
}

def get_code_example(framework, operation):
    """
    Retrieve a specific code example based on framework and operation.
    
    Args:
        framework: The FHE framework ('openfhe', 'seal', or 'helib')
        operation: The operation type ('addition', 'multiplication', etc.)
        
    Returns:
        str: The code example or empty string if not found
    """
    framework = framework.lower()
    
    if framework == 'openfhe':
        return OPENFHE_EXAMPLES.get(operation, "")
    elif framework == 'seal':
        return SEAL_EXAMPLES.get(operation, "")
    elif framework == 'helib':
        return HELIB_EXAMPLES.get(operation, "")
    else:
        return ""