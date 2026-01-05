#include "openfhe.h"
#include <vector>
#include <iostream>

using namespace lbcrypto;

// Helper function to print vectors
void printVector(const std::vector<double>& vec, const std::string& name) {
    std::cout << name << " = [";
    for (size_t i = 0; i < vec.size(); ++i) {
        std::cout << vec[i];
        if (i < vec.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]" << std::endl;
}

// Helper function to calculate normal convolution (for verification)
std::vector<double> calculateConvolution(const std::vector<double>& signal, const std::vector<double>& kernel) {
    int resultSize = signal.size() + kernel.size() - 1;
    std::vector<double> result(resultSize, 0.0);
    
    for (size_t i = 0; i < resultSize; i++) {
        for (size_t j = 0; j < kernel.size(); j++) {
            if (i >= j && i - j < signal.size()) {
                result[i] += signal[i - j] * kernel[j];
            }
        }
    }
    
    return result;
}

int main() {
    // Step 1: Setup CryptoContext
    uint32_t multDepth = 2;  // Increased for rotation operations
    uint32_t scaleModSize = 50;
    uint32_t batchSize = 8;  // Must be large enough for input + kernel size

    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetMultiplicativeDepth(multDepth);
    parameters.SetScalingModSize(scaleModSize);
    parameters.SetBatchSize(batchSize);

    CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);

    // Enable the features that you wish to use
    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    std::cout << "CKKS scheme is using ring dimension " << cc->GetRingDimension() << std::endl << std::endl;

    // Step 2: Key Generation
    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);
    
    // Generate rotation keys for all possible rotations in the convolution
    std::vector<int> rotations;
    for (int i = 1; i < batchSize; i++) {
        rotations.push_back(i);
        rotations.push_back(-i);
    }
    cc->EvalRotateKeyGen(keys.secretKey, rotations);

    // Step 3: Define input signal and convolution kernel
    // Input signal of length 5
    std::vector<double> signal = {1.0, 2.0, 3.0, 4.0, 5.0};
    
    // Convolution kernel of length 3
    std::vector<double> kernel = {0.5, 1.0, 0.5};
    
    printVector(signal, "Input Signal");
    printVector(kernel, "Convolution Kernel");
    
    // Pad signal to batchSize
    signal.resize(batchSize, 0.0);
    
    // Encode and encrypt the signal
    Plaintext ptxtSignal = cc->MakeCKKSPackedPlaintext(signal);
    auto cSignal = cc->Encrypt(keys.publicKey, ptxtSignal);
    
    // Step 4: Implement 1D convolution
    // For each position in the result, we compute a dot product of the kernel with a window of the signal
    auto cResult = cc->EvalMult(cSignal, 0.0); // Initialize with zeros
    
    for (size_t i = 0; i < kernel.size(); i++) {
        // Rotate the signal to align with the current kernel position
        auto rotated = (i == 0) ? cSignal : cc->EvalRotate(cSignal, i);
        
        // Multiply by the kernel value and add to the result
        auto term = cc->EvalMult(rotated, kernel[i]);
        cResult = cc->EvalAdd(cResult, term);
    }
    
    // Step 5: Decryption and output
    Plaintext result;
    std::cout.precision(8);
    
    cc->Decrypt(keys.secretKey, cResult, &result);
    result->SetLength(batchSize);
    
    std::cout << std::endl << "Results of homomorphic 1D convolution: " << std::endl;
    std::cout << "Encrypted result: " << result << std::endl;
    
    // Calculate expected convolution result
    std::vector<double> expectedResult = calculateConvolution(
        std::vector<double>(signal.begin(), signal.begin() + 5), // Original signal without padding
        kernel
    );
    
    // Pad expected result to match the CKKS output size
    expectedResult.resize(batchSize, 0.0);
    
    printVector(expectedResult, "Expected result");
    
    // Verify result
    std::cout << "Verification: " << std::endl;
    for (size_t i = 0; i < expectedResult.size(); i++) {
        std::complex<double> decryptedValue = result->GetCKKSPackedValue()[i];
        double expectedValue = expectedResult[i];
        double relativeError = (expectedValue == 0) ? 0 : std::abs((decryptedValue - expectedValue) / expectedValue);
        
        std::cout << "Position " << i << ": " 
                  << "Expected = " << expectedValue << ", "
                  << "Actual = " << decryptedValue << ", "
                  << "Relative Error = " << relativeError << std::endl;
    }
    
    std::cout << "Estimated precision in bits: " << result->GetLogPrecision() << std::endl;
    
    return 0;
}
