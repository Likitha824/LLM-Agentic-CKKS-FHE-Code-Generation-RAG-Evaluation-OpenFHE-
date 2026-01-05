#include "openfhe.h"
#include <vector>
#include <iostream>

using namespace lbcrypto;

// Helper function to print matrices
void printMatrix(const std::vector<std::vector<double>>& matrix, const std::string& name) {
    std::cout << name << " = [" << std::endl;
    for (const auto& row : matrix) {
        std::cout << "  [";
        for (size_t i = 0; i < row.size(); ++i) {
            std::cout << row[i];
            if (i < row.size() - 1) {
                std::cout << ", ";
            }
        }
        std::cout << "]" << std::endl;
    }
    std::cout << "]" << std::endl;
}

int main() {
    // Step 1: Setup CryptoContext
    uint32_t multDepth = 2;  // Increased for rotation and matrix operations
    uint32_t scaleModSize = 50;
    uint32_t batchSize = 8;  // Must be at least 4 for 2x2 matrices

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
    
    // Generate rotation keys for all possible rotations in the matrix multiplication
    std::vector<int> rotations;
    for (int i = 1; i < batchSize; i++) {
        rotations.push_back(i);
    }
    cc->EvalRotateKeyGen(keys.secretKey, rotations);

    // Step 3: Define input matrices (2x2)
    std::vector<std::vector<double>> matrixA = {
        {1.0, 2.0},
        {3.0, 4.0}
    };
    
    std::vector<std::vector<double>> matrixB = {
        {5.0, 6.0},
        {7.0, 8.0}
    };
    
    printMatrix(matrixA, "Matrix A");
    printMatrix(matrixB, "Matrix B");
    
    // Flatten matrices into vectors (row-major order)
    std::vector<double> flatA = {matrixA[0][0], matrixA[0][1], matrixA[1][0], matrixA[1][1]};
    std::vector<double> flatB = {matrixB[0][0], matrixB[0][1], matrixB[1][0], matrixB[1][1]};
    
    // Pad vectors to batchSize
    flatA.resize(batchSize, 0.0);
    flatB.resize(batchSize, 0.0);
    
    // Encode and encrypt the matrices
    Plaintext ptxtA = cc->MakeCKKSPackedPlaintext(flatA);
    Plaintext ptxtB = cc->MakeCKKSPackedPlaintext(flatB);
    
    auto cA = cc->Encrypt(keys.publicKey, ptxtA);
    auto cB = cc->Encrypt(keys.publicKey, ptxtB);
    
    // Step 4: Implement matrix multiplication C = A * B
    // For 2x2 matrices:
    // C[0][0] = A[0][0] * B[0][0] + A[0][1] * B[1][0]
    // C[0][1] = A[0][0] * B[0][1] + A[0][1] * B[1][1]
    // C[1][0] = A[1][0] * B[0][0] + A[1][1] * B[1][0]
    // C[1][1] = A[1][0] * B[0][1] + A[1][1] * B[1][1]
    
    // Rotate B to align for multiplication
    auto BCol0 = cc->EvalRotate(cB, 0);  // No rotation needed for first column
    auto BCol1 = cc->EvalRotate(cB, 1);  // Rotate to bring second column elements to the right positions
    
    // Create masks for each row of A
    std::vector<double> maskA_row0 = {1.0, 1.0, 0.0, 0.0};
    std::vector<double> maskA_row1 = {0.0, 0.0, 1.0, 1.0};
    maskA_row0.resize(batchSize, 0.0);
    maskA_row1.resize(batchSize, 0.0);
    
    auto ptxtMaskA_row0 = cc->MakeCKKSPackedPlaintext(maskA_row0);
    auto ptxtMaskA_row1 = cc->MakeCKKSPackedPlaintext(maskA_row1);
    
    // Extract rows from A
    auto A_row0 = cc->EvalMult(cA, ptxtMaskA_row0);
    auto A_row1 = cc->EvalMult(cA, ptxtMaskA_row1);
    
    // Multiply rows of A with columns of B
    auto C00_01 = cc->EvalMult(A_row0, BCol0);  // First row of A * first column of B
    auto C01_11 = cc->EvalMult(A_row0, BCol1);  // First row of A * second column of B
    auto C10_11 = cc->EvalMult(A_row1, BCol0);  // Second row of A * first column of B
    auto C11_11 = cc->EvalMult(A_row1, BCol1);  // Second row of A * second column of B
    
    // Sum the products to get the final result
    auto C00 = cc->EvalAdd(cc->EvalRotate(C00_01, 1), C00_01);
    auto C01 = cc->EvalAdd(cc->EvalRotate(C01_11, 1), C01_11);
    auto C10 = cc->EvalAdd(cc->EvalRotate(C10_11, 1), C10_11);
    auto C11 = cc->EvalAdd(cc->EvalRotate(C11_11, 1), C11_11);
    
    // Combine the results
    std::vector<double> maskC00 = {1.0, 0.0, 0.0, 0.0};
    std::vector<double> maskC01 = {0.0, 1.0, 0.0, 0.0};
    std::vector<double> maskC10 = {0.0, 0.0, 1.0, 0.0};
    std::vector<double> maskC11 = {0.0, 0.0, 0.0, 1.0};
    maskC00.resize(batchSize, 0.0);
    maskC01.resize(batchSize, 0.0);
    maskC10.resize(batchSize, 0.0);
    maskC11.resize(batchSize, 0.0);
    
    auto ptxtMaskC00 = cc->MakeCKKSPackedPlaintext(maskC00);
    auto ptxtMaskC01 = cc->MakeCKKSPackedPlaintext(maskC01);
    auto ptxtMaskC10 = cc->MakeCKKSPackedPlaintext(maskC10);
    auto ptxtMaskC11 = cc->MakeCKKSPackedPlaintext(maskC11);
    
    auto cC00 = cc->EvalMult(C00, ptxtMaskC00);
    auto cC01 = cc->EvalMult(C01, ptxtMaskC01);
    auto cC10 = cc->EvalMult(C10, ptxtMaskC10);
    auto cC11 = cc->EvalMult(C11, ptxtMaskC11);
    
    auto cResult = cc->EvalAdd(cc->EvalAdd(cC00, cC01), cc->EvalAdd(cC10, cC11));
    
    // Step 5: Decryption and output
    Plaintext result;
    std::cout.precision(8);
    
    cc->Decrypt(keys.secretKey, cResult, &result);
    result->SetLength(batchSize);
    
    std::cout << std::endl << "Results of homomorphic matrix multiplication: " << std::endl;
    std::cout << "Encrypted result: " << result << std::endl;
    
    // Extract the 2x2 result matrix from the decrypted vector
    std::vector<std::vector<double>> matrixC = {
        {result->GetCKKSPackedValue()[0], result->GetCKKSPackedValue()[1]},
        {result->GetCKKSPackedValue()[2], result->GetCKKSPackedValue()[3]}
    };
    
    // Expected result calculation
    std::vector<std::vector<double>> expectedC = {
        {matrixA[0][0] * matrixB[0][0] + matrixA[0][1] * matrixB[1][0], 
         matrixA[0][0] * matrixB[0][1] + matrixA[0][1] * matrixB[1][1]},
        {matrixA[1][0] * matrixB[0][0] + matrixA[1][1] * matrixB[1][0], 
         matrixA[1][0] * matrixB[0][1] + matrixA[1][1] * matrixB[1][1]}
    };
    
    printMatrix(matrixC, "Result Matrix C");
    printMatrix(expectedC, "Expected Matrix C");
    
    std::cout << "Estimated precision in bits: " << result->GetLogPrecision() << std::endl;
    
    return 0;
}
