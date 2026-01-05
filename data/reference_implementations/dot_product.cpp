#include "openfhe.h"
#include <iostream>
#include <vector>

using namespace std;
using namespace lbcrypto;

int main() {
    // Step 1: Set CKKS parameters
    uint32_t batchSize = 10;
    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetMultiplicativeDepth(5);
    parameters.SetScalingModSize(50);
    parameters.SetScalingTechnique(FLEXIBLEAUTO);
    parameters.SetBatchSize(batchSize);

    CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);
    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);

    // Step 2: Key generation
    auto keys = cc->KeyGen();
    cc->EvalSumKeyGen(keys.secretKey);
    cc->EvalMultKeyGen(keys.secretKey);

    // Step 3: Define two input vectors to encrypt
    vector<double> input1 = vector<double>(10);
    vector<double> input2 = vector<double>(10);

    for (int i=0; i < 10; i++) {
        cin >> input1[i];
    }

    for (int i=0; i < 10; i++) {
        cin >> input2[i];
    }

    // Padding with zeros to fit batch size if needed
    input1.resize(batchSize, 0.0);
    input2.resize(batchSize, 0.0);

    // Step 4: Encode and encrypt the inputs
    Plaintext plaintext1 = cc->MakeCKKSPackedPlaintext(input1);
    Plaintext plaintext2 = cc->MakeCKKSPackedPlaintext(input2);

    auto ciphertext1 = cc->Encrypt(keys.publicKey, plaintext1);
    auto ciphertext2 = cc->Encrypt(keys.publicKey, plaintext2);

    // Step 5: Perform encrypted vector multiplication
    auto ciphertextMult = cc->EvalMult(ciphertext1, ciphertext2);

    // Step 6: Sum up the numbers in all slots
    auto ciphertextSum = cc->EvalSum(ciphertextMult, batchSize);

    // Step 7: Decrypt and decode the result
    Plaintext decryptedResult;
    cc->Decrypt(keys.secretKey, ciphertextMult, &decryptedResult);
    decryptedResult->SetLength(input1.size());

    // Step 8: Print the decrypted result
    cout << "Result of homomorphic dot product: " << decryptedResult << endl;

    return 0;
}

