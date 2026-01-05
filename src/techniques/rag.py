"""
Implementation of Retrieval-Augmented Generation (RAG) techniques for CKKS code generation.
"""

import pickle
from typing import Dict, List, Any, Optional
import os
import re
import json
from pathlib import Path
from src.techniques.codeExamples import OPENFHE_EXAMPLES
import faiss
import numpy as np
import openai

class BasicRAG:
    """
    Basic Retrieval-Augmented Generation for CKKS code.
    
    This technique:
    1. Maintains a knowledge base of OpenFHE documentation chunks
    2. Retrieves relevant documentation based on the operation
    3. Augments prompts with the retrieved documentation
    """
    
    def __init__(self, operation_type: str, knowledge_base_dir: Optional[str] = None):
        """
        Initialize the RAG technique.
        
        Args:
            operation_type: Type of CKKS operation
            knowledge_base_dir: Directory containing knowledge base files
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
        
        # Set up knowledge base directory
        if knowledge_base_dir is None:
            # Default to 'data/knowledge_base' relative to this file
            self.knowledge_base_dir = Path(__file__).parent.parent.parent / "data" / "knowledge_base"
        else:
            self.knowledge_base_dir = Path(knowledge_base_dir)
            
        # Ensure the knowledge base directory exists
        os.makedirs(self.knowledge_base_dir, exist_ok=True)
        
        # Load or create the knowledge base
        self.knowledge_base = self._load_or_create_knowledge_base()
    
    def _load_or_create_knowledge_base(self) -> Dict[str, List[str]]:
        """
        Load the knowledge base from files or create it if it doesn't exist.
        
        Returns:
            Dictionary mapping topics to lists of documentation chunks
        """
        # Default knowledge base structure
        knowledge_base = {
            "general": [],
            "parameters": [],
            "key_generation": [],
            "encoding": [],
            "encryption": [],
            "operations": {
                "addition": [],
                "multiplication": [],
                "dot_product": [],
                "matrix_multiplication": [],
                "convolution": []
            },
            "decryption": [],
            "examples": {
                "addition": [],
                "multiplication": [],
                "dot_product": [],
                "matrix_multiplication": [],
                "convolution": []
            }
        }
        
        # Check if we have knowledge base files
        kb_file = self.knowledge_base_dir / "openfhe_ckks_knowledge_base.json"
        
        if kb_file.exists():
            # Load existing knowledge base
            try:
                with open(kb_file, "r") as f:
                    loaded_kb = json.load(f)
                    knowledge_base.update(loaded_kb)
            except Exception as e:
                print(f"Error loading knowledge base: {e}")
                # Fall back to default knowledge base with added documentation
                knowledge_base = self._create_default_knowledge_base()
        else:
            # Create default knowledge base with documentation
            knowledge_base = self._create_default_knowledge_base()
            
            # Save it for future use
            with open(kb_file, "w") as f:
                json.dump(knowledge_base, f, indent=2)
        
        return knowledge_base
    
    def _create_default_knowledge_base(self) -> Dict[str, Any]:
        """
        Create a default knowledge base with OpenFHE CKKS documentation.
        
        Returns:
            Knowledge base dictionary
        """
        # Basic knowledge base structure
        knowledge_base = {
            "general": [
                "OpenFHE is an open-source FHE library that implements the CKKS scheme.",
                "CKKS (Cheon-Kim-Kim-Song) is a homomorphic encryption scheme for approximate arithmetic.",
                "CKKS allows computation on encrypted real numbers through scaling and rounding.",
            ],
            "parameters": [
                "For CKKS, important parameters include:\n- multiplicativeDepth: Maximum depth of nested multiplications\n- scalingModSize: Bit-length of the scaling factor\n- batchSize: Number of slots in a packed ciphertext\n- securityLevel: Security level in bits",
                "CCParams<CryptoContextCKKSRNS> parameters;\nparameters.SetMultiplicativeDepth(multiplicativeDepth);\nparameters.SetScalingModSize(scalingModSize);\nparameters.SetBatchSize(batchSize);",
                "CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);",
            ],
            "key_generation": [
                "Key generation in OpenFHE CKKS involves creating public/private keys and special evaluation keys.",
                "KeyPair<DCRTPoly> keys = cc->KeyGen();",
                "For multiplication operations, multiplication evaluation keys are needed:\ncc->EvalMultKeyGen(keys.secretKey);",
                "For rotation operations, rotation evaluation keys are needed:\ncc->EvalRotateKeyGen(keys.secretKey, {1, 2, -1, -2});",
            ],
            "encoding": [
                "Data is encoded into plaintexts before encryption in CKKS.",
                "Plaintext ptxt = cc->MakeCKKSPackedPlaintext(data);",
                "The data is a vector of double values that will be encoded.",
                "CKKS allows packing multiple values into a single ciphertext for efficient batch processing.",
            ],
            "encryption": [
                "Encryption converts plaintexts to ciphertexts that can be computed on.",
                "Ciphertext<DCRTPoly> cipher = cc->Encrypt(keys.publicKey, ptxt);",
                "After encryption, computations are performed on ciphertexts without decryption.",
            ],
            "operations": {
                "addition": [
                    "Addition in CKKS performs element-wise addition of encrypted vectors.",
                    "Ciphertext<DCRTPoly> cResult = cc->EvalAdd(c1, c2);",
                    "Addition is a relatively simple operation in CKKS that doesn't increase noise significantly.",
                ],
                "multiplication": [
                    "Multiplication in CKKS performs element-wise multiplication of encrypted vectors.",
                    "Ciphertext<DCRTPoly> cResult = cc->EvalMult(c1, c2);",
                    "Multiplication increases noise and requires appropriate parameter selection.",
                    "EvalMultKeyGen must be called before performing multiplications.",
                ],
                "dot_product": [
                    "Dot product in CKKS computes the inner product of two encrypted vectors.",
                    "You can use EvalInnerProduct directly: Ciphertext<DCRTPoly> cResult = cc->EvalInnerProduct(c1, c2);",
                    "Alternatively, compute using element-wise multiplication followed by summation.",
                    "For manual implementation, first multiply element-wise, then sum using rotations.",
                    "EvalRotateKeyGen must be called before performing rotations for summation.",
                ],
                "matrix_multiplication": [
                    "Matrix multiplication in CKKS requires careful data encoding and manipulation.",
                    "Matrices need to be encoded in a row or column-major format.",
                    "A combination of EvalMult and EvalRotate operations is needed.",
                    "For m×n and n×p matrices, use n multiplications and (n-1) additions.",
                    "Matrix multiplication requires rotation keys: cc->EvalRotateKeyGen(keys.secretKey, rotations);",
                ],
                "convolution": [
                    "Convolution in CKKS involves a sliding window operation on encrypted data.",
                    "For 1D convolution, use a combination of rotations and multiplications.",
                    "For each kernel position, rotate the input, multiply by the kernel coefficient, and add to the result.",
                    "Rotation keys are required: cc->EvalRotateKeyGen(keys.secretKey, rotationIndices);",
                    "Boundary conditions need special handling in convolution operations.",
                ],
            },
            "decryption": [
                "Decryption converts ciphertexts back to plaintexts.",
                "Plaintext result;\ncc->Decrypt(keys.secretKey, ciphertext, &result);",
                "After decryption, the result contains approximate real values.",
                "result->GetRealPackedValue() returns the vector of real values.",
            ],
            "examples": {
                "addition": [
                    "// Addition example\nCiphertext<DCRTPoly> cAdd = cc->EvalAdd(c1, c2);\nPlaintext ptxtAdd;\ncc->Decrypt(keys.secretKey, cAdd, &ptxtAdd);",
                ],
                "multiplication": [
                    "// Multiplication example\ncc->EvalMultKeyGen(keys.secretKey);\nCiphertext<DCRTPoly> cMult = cc->EvalMult(c1, c2);\nPlaintext ptxtMult;\ncc->Decrypt(keys.secretKey, cMult, &ptxtMult);",
                ],
                "dot_product": [
                    "// Dot product example\ncc->EvalMultKeyGen(keys.secretKey);\ncc->EvalRotateKeyGen(keys.secretKey, {1, 2, 4});\nCiphertext<DCRTPoly> cDot = cc->EvalInnerProduct(c1, c2, batchSize);\nPlaintext ptxtDot;\ncc->Decrypt(keys.secretKey, cDot, &ptxtDot);",
                ],
                "matrix_multiplication": [
                    "// Matrix multiplication example\ncc->EvalMultKeyGen(keys.secretKey);\nfor (int i = 1; i < matrixDim; i++) {\n    cc->EvalRotateKeyGen(keys.secretKey, {i});\n}\n// Perform operations for each row and column\nCiphertext<DCRTPoly> cMatMult = /* complex operations */;\nPlaintext ptxtMatMult;\ncc->Decrypt(keys.secretKey, cMatMult, &ptxtMatMult);",
                ],
                "convolution": [
                    "// Convolution example\ncc->EvalMultKeyGen(keys.secretKey);\nfor (int i = 1; i < kernelSize; i++) {\n    cc->EvalRotateKeyGen(keys.secretKey, {i});\n}\n// Initialize result ciphertext\nCiphertext<DCRTPoly> cResult = cc->EvalMult(c1, 0);\n// For each kernel position\nfor (int i = 0; i < kernelSize; i++) {\n    auto rotated = cc->EvalRotate(c1, i);\n    auto term = cc->EvalMult(rotated, kernel[i]);\n    cResult = cc->EvalAdd(cResult, term);\n}\nPlaintext ptxtConv;\ncc->Decrypt(keys.secretKey, cResult, &ptxtConv);",
                ],
            }
        }
        
        return knowledge_base
    
    def _retrieve_relevant_documentation(self) -> List[str]:
        """
        Retrieve documentation relevant to the current operation.
        
        Returns:
            List of relevant documentation chunks
        """
        # Start with general information
        relevant_docs = []
        
        # Add general FHE and CKKS information
        relevant_docs.extend(self.knowledge_base["general"])
        
        # Add parameter setup information
        relevant_docs.extend(self.knowledge_base["parameters"])
        
        # Add key generation information
        relevant_docs.extend(self.knowledge_base["key_generation"])
        
        # Add encoding and encryption information
        relevant_docs.extend(self.knowledge_base["encoding"])
        relevant_docs.extend(self.knowledge_base["encryption"])
        
        # Add operation-specific information
        if self.operation_type in self.knowledge_base["operations"]:
            relevant_docs.extend(self.knowledge_base["operations"][self.operation_type])
        
        # Add example for the specific operation
        if self.operation_type in self.knowledge_base["examples"]:
            relevant_docs.extend(self.knowledge_base["examples"][self.operation_type])
        
        # Add decryption information
        relevant_docs.extend(self.knowledge_base["decryption"])
        
        return relevant_docs
    
    def generate_rag_prompt(self) -> str:
        docs = self._retrieve_relevant_documentation()
        context = "\n\n".join(docs)
        descriptions = {
            "addition": "element-wise addition of two encrypted vectors",
            "multiplication": "element-wise multiplication of two encrypted vectors",
            "dot_product": "dot product (inner product) of two encrypted vectors",
            "matrix_multiplication": "matrix multiplication of two encrypted matrices",
            "convolution": "1D convolution of an encrypted signal with a kernel"
        }
        desc = descriptions[self.operation_type]
        prompt = f"""
You are an expert in Fully Homomorphic Encryption (FHE) using the CKKS scheme in OpenFHE.

RELEVANT DOCUMENTATION:
{context}

TASK: Implement CKKS {self.operation_type} operation using OpenFHE.

OPERATION DESCRIPTION:
This operation performs {desc}.

Your program MUST:
1. Include headers: #include "openfhe.h", <iostream>, <vector>
2. using namespace lbcrypto;
3. Set up parameters with CCParams<CryptoContextCKKSRNS>:
   parameters.SetMultiplicativeDepth(...);
   parameters.SetScalingModSize(...);
   parameters.SetBatchSize(...);
   parameters.SetSecurityLevel(SecurityLevel::HEStd_128_classic);
4. Create context:
   auto cc = GenCryptoContext(parameters);
   cc->Enable(PKE); cc->Enable(KEYSWITCH); cc->Enable(LEVELEDSHE);
5. If your operation uses rotations (dot_product, matrix_multiplication, convolution):
   cc->Enable(ADVANCEDSHE);
   // Then generate rotation keys:
   // Option A: cc->EvalRotateKeyGen(keys.secretKey, {indices});
   // Option B: cc->EvalSumKeyGen(keys.secretKey);
6. Generate keys:
   auto keys = cc->KeyGen();
   cc->EvalMultKeyGen(keys.secretKey);
7. Encode and encrypt test data using MakeCKKSPackedPlaintext and Encrypt
8. Perform {self.operation_type} with EvalAdd/EvalMult/etc
9. Decrypt, SetLength(...), GetRealPackedValue(), and verify
10. Include clear comments for each step

Return a single, compilable C++ file.
"""
        return prompt


class GraphRAG(BasicRAG):
    """
    Graph-based Retrieval-Augmented Generation for CKKS code.
    
    This technique builds on BasicRAG by:
    1. Organizing documentation in a knowledge graph structure with weighted nodes and edges
    2. Retrieving documentation based on graph relationships and relevance scores
    3. Following multi-hop paths in the graph for more comprehensive retrieval
    4. Using bidirectional graph traversal for more complete context gathering
    """
    
    def __init__(self, operation_type: str, knowledge_base_dir: Optional[str] = None):
        """Initialize the Graph RAG technique."""
        super().__init__(operation_type, knowledge_base_dir)
        
        # Load or create the knowledge graph
        self.knowledge_graph = self._load_or_create_knowledge_graph()
        
        # Operation-specific relevance weights
        self.relevance_weights = self._initialize_relevance_weights()
    
    def _initialize_relevance_weights(self) -> Dict[str, float]:
        """
        Initialize node relevance weights based on operation type.
        
        Returns:
            Dictionary mapping node IDs to relevance weights
        """
        # Base weights for all operations
        weights = {
            "ckks": 0.8,
            "parameters": 0.7,
            "crypto_context": 0.6,
            "key_generation": 0.7,
            "keypair": 0.6,
            "encoding": 0.7,
            "plaintext": 0.6,
            "encryption": 0.7,
            "encrypt": 0.6,
            "decryption": 0.5,
            "decrypt": 0.5,
        }
        
        # Operation-specific weights
        operation_weights = {
            "addition": {
                "addition": 1.0,
                "eval_add": 1.0,
                "multiplication": 0.2,
                "eval_mult": 0.2,
                "eval_mult_key": 0.1,
                "dot_product": 0.1,
                "matrix_multiplication": 0.1,
                "convolution": 0.1,
            },
            "multiplication": {
                "multiplication": 1.0,
                "eval_mult": 1.0,
                "eval_mult_key": 0.9,
                "addition": 0.3,
                "eval_add": 0.3,
                "dot_product": 0.2,
                "matrix_multiplication": 0.2,
                "convolution": 0.2,
            },
            "dot_product": {
                "dot_product": 1.0,
                "eval_inner_product": 1.0,
                "multiplication": 0.8,
                "eval_mult": 0.8,
                "eval_mult_key": 0.7,
                "rotation": 0.8,
                "eval_rotate": 0.8,
                "eval_rotate_key": 0.7,
                "addition": 0.5,
                "eval_add": 0.5,
                "matrix_multiplication": 0.3,
                "convolution": 0.2,
            },
            "matrix_multiplication": {
                "matrix_multiplication": 1.0,
                "matrix_encoding": 0.9,
                "dot_product": 0.7,
                "eval_inner_product": 0.6,
                "multiplication": 0.8,
                "eval_mult": 0.8,
                "eval_mult_key": 0.7,
                "rotation": 0.9,
                "eval_rotate": 0.9,
                "eval_rotate_key": 0.9,
                "addition": 0.6,
                "eval_add": 0.6,
                "convolution": 0.3,
            },
            "convolution": {
                "convolution": 1.0,
                "convolution_implementation": 1.0,
                "rotation": 0.9,
                "eval_rotate": 0.9,
                "eval_rotate_key": 0.9,
                "multiplication": 0.8,
                "eval_mult": 0.8,
                "eval_mult_key": 0.7,
                "addition": 0.6,
                "eval_add": 0.6,
                "matrix_multiplication": 0.3,
                "dot_product": 0.3,
            },
        }
        
        # Apply operation-specific weights
        if self.operation_type in operation_weights:
            for node, weight in operation_weights[self.operation_type].items():
                weights[node] = weight
                
        return weights
    
    def _load_or_create_knowledge_graph(self) -> Dict[str, Any]:
        """
        Load or create a knowledge graph for OpenFHE CKKS.
        
        Returns:
            Knowledge graph dictionary
        """
        # Check if we have a knowledge graph file
        kg_file = self.knowledge_base_dir / "openfhe_ckks_knowledge_graph.json"
        
        if kg_file.exists():
            # Load existing knowledge graph
            try:
                with open(kg_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading knowledge graph: {e}")
                # Fall back to creating a new one
                graph = self._create_default_knowledge_graph()
        else:
            # Create default knowledge graph
            graph = self._create_default_knowledge_graph()
            
            # Save it for future use
            with open(kg_file, "w") as f:
                json.dump(graph, f, indent=2)
        
        return graph
    
    def _create_default_knowledge_graph(self) -> Dict[str, Any]:
        """
        Create a default knowledge graph for OpenFHE CKKS.
        
        Returns:
            Knowledge graph with nodes and edges
        """
        # Create graph nodes
        nodes = {
            "ckks": {
                "type": "scheme",
                "content": "CKKS (Cheon-Kim-Kim-Song) is a homomorphic encryption scheme for encrypted approximate arithmetic on real numbers."
            },
            "parameters": {
                "type": "concept",
                "content": "CKKS parameters include multiplicativeDepth, scalingModSize, batchSize, and securityLevel."
            },
            "parameter_selection": {
                "type": "concept",
                "content": "Proper parameter selection is crucial for CKKS operations. Higher multiplicativeDepth allows more nested multiplications but increases computational cost."
            },
            "scaling_factor": {
                "type": "concept",
                "content": "The scaling factor in CKKS controls precision. Typically set via scalingModSize parameter (bits), often 30-60 bits."
            },
            "batch_size": {
                "type": "concept",
                "content": "The batch size determines how many values can be packed into a single ciphertext. It should be a power of 2."
            },
            "crypto_context": {
                "type": "api",
                "content": "CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);"
            },
            "key_generation": {
                "type": "concept",
                "content": "Key generation creates encryption keys and special evaluation keys."
            },
            "keypair": {
                "type": "api",
                "content": "KeyPair<DCRTPoly> keys = cc->KeyGen();"
            },
            "eval_mult_key": {
                "type": "api",
                "content": "cc->EvalMultKeyGen(keys.secretKey);"
            },
            "eval_rotate_key": {
                "type": "api",
                "content": "cc->EvalRotateKeyGen(keys.secretKey, {1, 2, -1});"
            },
            "encoding": {
                "type": "concept",
                "content": "Encoding converts vectors of real numbers into plaintexts suitable for encryption."
            },
            "plaintext": {
                "type": "api",
                "content": "Plaintext ptxt = cc->MakeCKKSPackedPlaintext(data);"
            },
            "encryption": {
                "type": "concept",
                "content": "Encryption converts plaintexts to ciphertexts for secure computation."
            },
            "encrypt": {
                "type": "api",
                "content": "Ciphertext<DCRTPoly> cipher = cc->Encrypt(keys.publicKey, ptxt);"
            },
            "decryption": {
                "type": "concept",
                "content": "Decryption converts ciphertexts back to plaintexts to reveal results."
            },
            "decrypt": {
                "type": "api",
                "content": "Plaintext result;\ncc->Decrypt(keys.secretKey, cipher, &result);"
            },
            "noise_management": {
                "type": "concept",
                "content": "Each operation in CKKS increases noise. Multiplications increase noise significantly more than additions."
            },
            "relinearization": {
                "type": "concept",
                "content": "Relinearization is used to manage ciphertext size after multiplication. It's automatically applied in EvalMult operations."
            },
            "rescaling": {
                "type": "concept",
                "content": "Rescaling reduces the scale of ciphertexts to manage precision and noise growth. It's automatically handled in CKKS operations."
            },
            "addition": {
                "type": "operation",
                "content": "Addition performs element-wise addition on encrypted vectors."
            },
            "eval_add": {
                "type": "api",
                "content": "Ciphertext<DCRTPoly> cResult = cc->EvalAdd(c1, c2);"
            },
            "add_plain": {
                "type": "api",
                "content": "Ciphertext<DCRTPoly> cResult = cc->EvalAdd(c1, ptxt);"
            },
            "multiplication": {
                "type": "operation",
                "content": "Multiplication performs element-wise multiplication on encrypted vectors."
            },
            "eval_mult": {
                "type": "api",
                "content": "Ciphertext<DCRTPoly> cResult = cc->EvalMult(c1, c2);"
            },
            "mult_plain": {
                "type": "api",
                "content": "Ciphertext<DCRTPoly> cResult = cc->EvalMult(c1, ptxt);"
            },
            "dot_product": {
                "type": "operation",
                "content": "Dot product computes the inner product of two encrypted vectors."
            },
            "eval_inner_product": {
                "type": "api",
                "content": "Ciphertext<DCRTPoly> cResult = cc->EvalInnerProduct(c1, c2, batchSize);"
            },
            "manual_dot_product": {
                "type": "code",
                "content": "// Manual dot product implementation\nCiphertext<DCRTPoly> product = cc->EvalMult(c1, c2);\nCiphertext<DCRTPoly> result = product;\n// Sum all elements\nfor (int i = 1; i < batchSize; i *= 2) {\n    auto rotated = cc->EvalRotate(product, i);\n    result = cc->EvalAdd(result, rotated);\n}"
            },
            "rotation": {
                "type": "concept",
                "content": "Rotation shifts elements in the encrypted vector by a specified offset."
            },
            "eval_rotate": {
                "type": "api",
                "content": "Ciphertext<DCRTPoly> cResult = cc->EvalRotate(c, offset);"
            },
            "matrix_multiplication": {
                "type": "operation",
                "content": "Matrix multiplication involves encoding matrices and using a combination of rotations and multiplications."
            },
            "matrix_encoding": {
                "type": "concept",
                "content": "Matrices can be encoded in row-major or column-major format for CKKS operations."
            },
            "matrix_mult_implementation": {
                "type": "code",
                "content": "// Matrix multiplication implementation (simplified)\n// Assuming A (m×n) is encrypted, B (n×p) is in plaintext\n// For each column j of B\nfor (int j = 0; j < p; j++) {\n    // Extract column j from B\n    vector<double> bCol = extractColumn(B, j);\n    // Create plaintext\n    Plaintext bColPt = cc->MakeCKKSPackedPlaintext(bCol);\n    // Initialize result column\n    Ciphertext<DCRTPoly> resultCol;\n    // For each row i of A (already encrypted in cA)\n    for (int i = 0; i < m; i++) {\n        // Extract row i from A using rotations\n        Ciphertext<DCRTPoly> aRow = cA;\n        for (int k = 0; k < i; k++) {\n            aRow = cc->EvalRotate(aRow, n);\n        }\n        // Multiply with column j of B\n        auto temp = cc->EvalMult(aRow, bColPt);\n        // Add to result\n        if (i == 0) {\n            resultCol = temp;\n        } else {\n            resultCol = cc->EvalAdd(resultCol, temp);\n        }\n    }\n    // Store result column\n    if (j == 0) {\n        result = resultCol;\n    } else {\n        // Combine columns\n        // Implementation depends on result representation\n    }\n}"
            },
            "convolution": {
                "type": "operation",
                "content": "Convolution slides a kernel over an encrypted signal, performing multiplications and additions."
            },
            "convolution_implementation": {
                "type": "code",
                "content": "// For each kernel position\nfor (int i = 0; i < kernelSize; i++) {\n    auto rotated = cc->EvalRotate(c1, i);\n    auto term = cc->EvalMult(rotated, kernel[i]);\n    cResult = cc->EvalAdd(cResult, term);\n}"
            },
            "precision_management": {
                "type": "concept",
                "content": "CKKS operations introduce approximation errors. Managing precision is important for accurate results."
            },
            "parameter_examples": {
                "type": "code",
                "content": "// Parameter examples for different operations\n// Low-depth operations (addition, simple multiplication)\nparameters.SetMultiplicativeDepth(1);\nparameters.SetScalingModSize(30);\n// Medium-depth operations (dot product, simple convolution)\nparameters.SetMultiplicativeDepth(3);\nparameters.SetScalingModSize(40);\n// High-depth operations (matrix multiplication, complex convolution)\nparameters.SetMultiplicativeDepth(5);\nparameters.SetScalingModSize(50);"
            },
            "rotation_indices": {
                "type": "concept",
                "content": "For operations requiring rotations, you need to generate rotation keys for specific indices. For example, dot products need indices for powers of 2."
            },
            "dot_product_rotation_setup": {
                "type": "code",
                "content": "// Generate rotation keys for dot product\nvector<int> rotationIndices;\nfor (int i = 1; i < batchSize; i *= 2) {\n    rotationIndices.push_back(i);\n}\ncc->EvalRotateKeyGen(keys.secretKey, rotationIndices);"
            }
        }
        
        # Create graph edges (relationships)
        edges = [
            {"source": "ckks", "target": "parameters", "relation": "uses", "weight": 0.9},
            {"source": "parameters", "target": "parameter_selection", "relation": "elaborated_by", "weight": 0.8},
            {"source": "parameters", "target": "scaling_factor", "relation": "includes", "weight": 0.7},
            {"source": "parameters", "target": "batch_size", "relation": "includes", "weight": 0.7},
            {"source": "parameters", "target": "crypto_context", "relation": "creates", "weight": 0.8},
            {"source": "parameters", "target": "parameter_examples", "relation": "illustrated_by", "weight": 0.6},
            {"source": "crypto_context", "target": "keypair", "relation": "generates", "weight": 0.8},
            {"source": "keypair", "target": "eval_mult_key", "relation": "enables", "weight": 0.7},
            {"source": "keypair", "target": "eval_rotate_key", "relation": "enables", "weight": 0.7},
            {"source": "encoding", "target": "plaintext", "relation": "produces", "weight": 0.8},
            {"source": "plaintext", "target": "encrypt", "relation": "inputs_to", "weight": 0.8},
            {"source": "encrypt", "target": "addition", "relation": "enables", "weight": 0.7},
            {"source": "encrypt", "target": "multiplication", "relation": "enables", "weight": 0.7},
            {"source": "encrypt", "target": "dot_product", "relation": "enables", "weight": 0.6},
            {"source": "encrypt", "target": "matrix_multiplication", "relation": "enables", "weight": 0.6},
            {"source": "encrypt", "target": "convolution", "relation": "enables", "weight": 0.6},
            {"source": "addition", "target": "eval_add", "relation": "uses", "weight": 0.9},
            {"source": "addition", "target": "add_plain", "relation": "uses", "weight": 0.7},
            {"source": "multiplication", "target": "eval_mult_key", "relation": "requires", "weight": 0.8},
            {"source": "multiplication", "target": "eval_mult", "relation": "uses", "weight": 0.9},
            {"source": "multiplication", "target": "mult_plain", "relation": "uses", "weight": 0.7},
            {"source": "multiplication", "target": "noise_management", "relation": "affected_by", "weight": 0.7},
            {"source": "multiplication", "target": "relinearization", "relation": "uses", "weight": 0.6},
            {"source": "multiplication", "target": "rescaling", "relation": "uses", "weight": 0.6},
            {"source": "dot_product", "target": "eval_mult", "relation": "uses", "weight": 0.8},
            {"source": "dot_product", "target": "rotation", "relation": "uses", "weight": 0.8},
            {"source": "dot_product", "target": "eval_inner_product", "relation": "uses", "weight": 0.9},
            {"source": "dot_product", "target": "manual_dot_product", "relation": "implemented_by", "weight": 0.9},
            {"source": "dot_product", "target": "dot_product_rotation_setup", "relation": "requires", "weight": 0.8},
            {"source": "rotation", "target": "eval_rotate_key", "relation": "requires", "weight": 0.8},
            {"source": "rotation", "target": "eval_rotate", "relation": "uses", "weight": 0.9},
            {"source": "rotation", "target": "rotation_indices", "relation": "requires", "weight": 0.7},
            {"source": "matrix_multiplication", "target": "matrix_encoding", "relation": "requires", "weight": 0.8},
            {"source": "matrix_multiplication", "target": "eval_mult", "relation": "uses", "weight": 0.8},
            {"source": "matrix_multiplication", "target": "eval_rotate", "relation": "uses", "weight": 0.8},
            {"source": "matrix_multiplication", "target": "matrix_mult_implementation", "relation": "implemented_by", "weight": 0.9},
            {"source": "matrix_multiplication", "target": "rotation_indices", "relation": "requires", "weight": 0.7},
            {"source": "convolution", "target": "eval_rotate", "relation": "uses", "weight": 0.8},
            {"source": "convolution", "target": "eval_mult", "relation": "uses", "weight": 0.8},
            {"source": "convolution", "target": "convolution_implementation", "relation": "implemented_by", "weight": 0.9},
            {"source": "convolution", "target": "rotation_indices", "relation": "requires", "weight": 0.7},
            {"source": "multiplication", "target": "precision_management", "relation": "affected_by", "weight": 0.6},
            {"source": "dot_product", "target": "precision_management", "relation": "affected_by", "weight": 0.6},
            {"source": "matrix_multiplication", "target": "precision_management", "relation": "affected_by", "weight": 0.7},
            {"source": "convolution", "target": "precision_management", "relation": "affected_by", "weight": 0.7},
            
            # Bidirectional edges for more comprehensive traversal
            {"source": "eval_add", "target": "addition", "relation": "implements", "weight": 0.9},
            {"source": "eval_mult", "target": "multiplication", "relation": "implements", "weight": 0.9},
            {"source": "eval_inner_product", "target": "dot_product", "relation": "implements", "weight": 0.9},
            {"source": "matrix_mult_implementation", "target": "matrix_multiplication", "relation": "implements", "weight": 0.9},
            {"source": "convolution_implementation", "target": "convolution", "relation": "implements", "weight": 0.9},
            {"source": "parameter_selection", "target": "parameters", "relation": "informs", "weight": 0.8},
            {"source": "scaling_factor", "target": "parameters", "relation": "part_of", "weight": 0.7},
            {"source": "batch_size", "target": "parameters", "relation": "part_of", "weight": 0.7},
            {"source": "rotation_indices", "target": "rotation", "relation": "informs", "weight": 0.7},
            {"source": "precision_management", "target": "scaling_factor", "relation": "affects", "weight": 0.6},
            {"source": "dot_product_rotation_setup", "target": "dot_product", "relation": "supports", "weight": 0.8}
        ]
        
        # Combine into a graph
        graph = {
            "nodes": nodes,
            "edges": edges
        }
        
        return graph
    
    def _bidirectional_graph_search(self, start_node: str, max_depth: int = 3, min_weight: float = 0.0) -> List[Dict[str, Any]]:
        """
        Perform bidirectional graph search from a starting node with relevance weighting.
        
        Args:
            start_node: Starting node ID
            max_depth: Maximum path depth
            min_weight: Minimum edge weight to consider
            
        Returns:
            List of nodes with their content and relevance score
        """
        if start_node not in self.knowledge_graph["nodes"]:
            return []
        
        # Track visited nodes and their scores
        visited = {}
        queue = [(start_node, 1.0, 0)]  # (node_id, score, depth)
        
        while queue:
            node_id, score, depth = queue.pop(0)
            
            # Skip if we've seen this node with a better score
            if node_id in visited and visited[node_id]["score"] >= score:
                continue
                
            # Get node information
            node_data = self.knowledge_graph["nodes"][node_id]
            base_relevance = self.relevance_weights.get(node_id, 0.5)
            
            # Store the node with its score
            visited[node_id] = {
                "id": node_id,
                "type": node_data["type"],
                "content": node_data["content"],
                "score": score * base_relevance,
                "depth": depth
            }
            
            # If we've reached max depth, don't explore further
            if depth >= max_depth:
                continue
            
            # Find all edges from this node
            outgoing_edges = [
                edge for edge in self.knowledge_graph["edges"]
                if edge["source"] == node_id and edge.get("weight", 0.5) >= min_weight
            ]
            
            # Follow each edge
            for edge in outgoing_edges:
                target = edge["target"]
                edge_weight = edge.get("weight", 0.5)
                # Calculate new score with decay
                new_score = score * edge_weight * (0.8 ** depth)
                queue.append((target, new_score, depth + 1))
        
        # Convert to list and sort by score
        result = list(visited.values())
        result.sort(key=lambda x: x["score"], reverse=True)
        
        return result
    
    def _retrieve_relevant_documentation(self) -> List[str]:
        """
        Retrieve documentation by intelligent graph search.
        
        Returns:
            List of relevant documentation chunks
        """
        relevant_nodes = []
        
        # Start with the specific operation
        if self.operation_type in self.knowledge_graph["nodes"]:
            operation_nodes = self._bidirectional_graph_search(self.operation_type, max_depth=3, min_weight=0.6)
            relevant_nodes.extend(operation_nodes)
        
        # Add CKKS general information with lower depth
        ckks_nodes = self._bidirectional_graph_search("ckks", max_depth=2, min_weight=0.7)
        relevant_nodes.extend(ckks_nodes)
        
        # Add parameter setup information if not already included
        if not any(n["id"] == "parameters" for n in relevant_nodes):
            param_nodes = self._bidirectional_graph_search("parameters", max_depth=2, min_weight=0.7)
            relevant_nodes.extend(param_nodes)
        
        # Add encoding and encryption information if not already included
        if not any(n["id"] == "encoding" for n in relevant_nodes):
            encoding_nodes = self._bidirectional_graph_search("encoding", max_depth=2, min_weight=0.7)
            relevant_nodes.extend(encoding_nodes)
        
        if not any(n["id"] == "encryption" for n in relevant_nodes):
            encryption_nodes = self._bidirectional_graph_search("encryption", max_depth=2, min_weight=0.7)
            relevant_nodes.extend(encryption_nodes)
        
        # Add decryption information if not already included
        if not any(n["id"] == "decryption" for n in relevant_nodes):
            decryption_nodes = self._bidirectional_graph_search("decryption", max_depth=1, min_weight=0.7)
            relevant_nodes.extend(decryption_nodes)
        
        # Remove duplicates by keeping highest score for each node
        unique_nodes = {}
        for node in relevant_nodes:
            node_id = node["id"]
            if node_id not in unique_nodes or unique_nodes[node_id]["score"] < node["score"]:
                unique_nodes[node_id] = node
        
        # Sort by score
        sorted_nodes = sorted(unique_nodes.values(), key=lambda x: x["score"], reverse=True)
        
        # Extract content, prioritizing more relevant nodes
        # Limit to top 25 nodes to prevent overwhelming the LLM
        relevant_docs = [node["content"] for node in sorted_nodes[:25]]
        
        return relevant_docs
    
    def generate_rag_prompt(self) -> str:
        """
        Generate a RAG-enhanced prompt for code generation with contextual organization.
        
        Returns:
            Complete prompt string with retrieved documentation
        """
        # Get the relevant documentation
        docs = self._retrieve_relevant_documentation()
        
        # Organize documentation into sections
        intro_docs = []
        parameter_docs = []
        implementation_docs = []
        operation_specific_docs = []
        
        # Simple categorization based on content
        for doc in docs:
            if "parameter" in doc.lower() or "scaling" in doc.lower() or "depth" in doc.lower():
                parameter_docs.append(doc)
            elif self.operation_type in doc.lower():
                operation_specific_docs.append(doc)
            elif "implementation" in doc.lower() or "for each" in doc.lower() or "algorithm" in doc.lower():
                implementation_docs.append(doc)
            else:
                intro_docs.append(doc)
        
        # Format sections
        sections = []
        
        if intro_docs:
            sections.append("GENERAL CKKS CONCEPTS:\n" + "\n\n".join(intro_docs))
            
        if parameter_docs:
            sections.append("PARAMETER SELECTION:\n" + "\n\n".join(parameter_docs))
            
        if operation_specific_docs:
            sections.append(f"{self.operation_type.upper()} OPERATION DETAILS:\n" + "\n\n".join(operation_specific_docs))
            
        if implementation_docs:
            sections.append("IMPLEMENTATION GUIDANCE:\n" + "\n\n".join(implementation_docs))
        
        # Combine sections
        context = "\n\n" + "\n\n".join(sections)
        
        # Generate the base prompt
        operation_descriptions = {
            "addition": "element-wise addition of two encrypted vectors",
            "multiplication": "element-wise multiplication of two encrypted vectors",
            "dot_product": "dot product (inner product) of two encrypted vectors",
            "matrix_multiplication": "matrix multiplication of two encrypted matrices",
            "convolution": "1D convolution of an encrypted signal with a kernel"
        }
        
        description = operation_descriptions.get(self.operation_type, "")
        
        # Add canonical code example for the operation
        code_example = OPENFHE_EXAMPLES.get(self.operation_type, None)
        code_section = ""
        if code_example:
            code_section = f"\n\nREFERENCE IMPLEMENTATION (OpenFHE, {self.operation_type}):\n```cpp\n{code_example}\n```"
        
        # Add common compilation issues to avoid
        compilation_guidance = """
IMPORTANT COMPILATION REQUIREMENTS:
1. Include ALL necessary headers: #include "openfhe.h", <iostream>, <vector>, etc.
2. Use the correct namespace: 'using namespace lbcrypto;'
3. Always call EvalMultKeyGen() before using EvalMult operations
4. Always call EvalRotateKeyGen() with specific indices before using EvalRotate or EvalSum operations
5. Ensure SetMultiplicativeDepth is sufficient for your operation's complexity
6. The SetBatchSize must be at least as large as your input vectors
7. Always check vector sizes are compatible before operations
8. Call cc->Enable(PKE), cc->Enable(KEYSWITCH), cc->Enable(LEVELEDSHE) after creating CryptoContext
9. For operations using rotation, also call cc->Enable(ADVANCEDSHE)
10. SetLength() on plaintexts before accessing them after decryption
"""

        # Add OpenFHE API-specific guidance based on common errors
        api_guidance = f"""
CRITICAL OPENFHE API USAGE NOTES:

1. SetLength() must be called on plaintexts, not on the CryptoContext:
   CORRECT:   plaintextResult->SetLength(vectorSize);
   INCORRECT: cryptoContext->SetLength(vectorSize);

2. EvalRotateKeyGen takes a secret key and vector of indices:
   CORRECT:   cryptoContext->EvalRotateKeyGen(keyPair.secretKey, {{1, 2, -1}});
   INCORRECT: cryptoContext->EvalRotateKeyGen(0, 0);

3. For dot product, EvalInnerProduct requires 3 parameters:
   CORRECT:   cryptoContext->EvalInnerProduct(ciphertext1, ciphertext2, vectorSize);
   INCORRECT: cryptoContext->EvalInnerProduct(ciphertext1, ciphertext2);

4. EvalSum's second parameter must be an integer size:
   CORRECT:   cryptoContext->EvalSum(ciphertextMult, vectorSize);
   INCORRECT: cryptoContext->EvalSum(ciphertextSum, rotated);

5. There is no EvalMultConstant in OpenFHE, use EvalMult with a scalar:
   CORRECT:   cryptoContext->EvalMult(ciphertext, 2.5);
   INCORRECT: cryptoContext->EvalMultConstant(ciphertext, 2.5);

6. EvalRelinearize should be Relinearize in current OpenFHE versions:
   CORRECT:   cryptoContext->Relinearize(ciphertext);
   INCORRECT: cryptoContext->EvalRelinearize(ciphertext);

7. EvalMult's first parameter must always be a ciphertext:
   CORRECT:   auto ciphertextZero = cryptoContext->Encrypt(keyPair.publicKey, zeroPlaintext);
              cryptoContext->EvalMult(ciphertextZero, scalar);
   INCORRECT: cryptoContext->EvalMult(0.0, ciphertext);
   
8. For operations using rotation ({', '.join(['dot_product', 'matrix_multiplication', 'convolution'])}):
   - You MUST call Enable(ADVANCEDSHE)
   - You MUST generate rotation keys with EvalRotateKeyGen or EvalSumKeyGen
"""

        # Add operation-specific guidance
        operation_guidance = {
            "dot_product": """
For dot product, implement using one of these approaches:
1. Use EvalInnerProduct with three parameters:
   auto result = cryptoContext->EvalInnerProduct(ciphertext1, ciphertext2, vectorSize);
   
2. Or manually with element-wise multiplication and sum:
   auto product = cryptoContext->EvalMult(ciphertext1, ciphertext2); 
   auto result = cryptoContext->EvalSum(product, vectorSize);
""",
            "convolution": """
For convolution:
1. Do NOT use EvalMultConstant or initialize with a scalar directly
2. Instead, create a zero ciphertext by encrypting a zero plaintext:
   auto zeroPlaintext = cryptoContext->MakeCKKSPackedPlaintext({0.0});
   auto ciphertextZero = cryptoContext->Encrypt(keyPair.publicKey, zeroPlaintext);
3. Always use proper rotation indices when generating rotation keys
""",
            "matrix_multiplication": """
For matrix multiplication:
1. Carefully track matrix dimensions during the implementation
2. Generate rotation keys with appropriate indices for your matrix dimensions
3. Flatten matrices using consistent ordering (row-major or column-major)
"""
        }
        
        # Add operation-specific guidance if available
        op_specific = operation_guidance.get(self.operation_type, "")
        
        prompt = f"""
You are an expert in Fully Homomorphic Encryption (FHE) using the CKKS scheme in OpenFHE.

I will provide you with relevant documentation and context for implementing a specific CKKS operation.

RELEVANT DOCUMENTATION:{context}

TASK: Implement CKKS {self.operation_type} operation using OpenFHE.

OPERATION DESCRIPTION:
This operation performs {description}.

{compilation_guidance}

{api_guidance}

{op_specific}

Generate a complete C++ program that:
1. Includes necessary headers and namespaces
2. Sets up appropriate parameters for CKKS, considering operation complexity
3. Generates all required keys (encryption, evaluation, rotation)
4. Creates test data, encodes and encrypts it
5. Performs the {self.operation_type} operation on encrypted data
6. Decrypts and verifies results against plaintext computation
7. Handles precision appropriately for CKKS approximate arithmetic

Your code should be complete, well-structured, and include clear comments explaining each step.
{code_section}
"""
        return prompt


class VectorRAG(BasicRAG):
    """
    Retrieval-Augmented Generation with vector embeddings:
    - Chunk reference docs & code examples
    - Embed with OpenAI embeddings
    - Index with FAISS
    - At runtime, similarity search to pull top chunks
    """
    def __init__(
        self,
        operation_type: str,
        knowledge_base_dir: Optional[str] = None,
        index_path: Optional[str] = None,
        embedding_model: str = "text-embedding-ada-002",
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        top_k: int = 5
    ):
        super().__init__(operation_type, knowledge_base_dir)
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k

        # Where to persist FAISS index + mapping
        kb_dir = Path(self.knowledge_base_dir)
        self.index_path = index_path or str(kb_dir / f"{operation_type}_vector.index")
        self.map_path   = self.index_path + ".pkl"

        if os.path.exists(self.index_path) and os.path.exists(self.map_path):
            self._load_index()
        else:
            self._build_index()

    def _chunk_text(self, text: str) -> List[str]:
        tokens = text.split()
        chunks = []
        i = 0
        while i < len(tokens):
            chunk = tokens[i:i+self.chunk_size]
            chunks.append(" ".join(chunk))
            i += self.chunk_size - self.chunk_overlap
        return chunks

    def _gather_all_chunks(self) -> List[str]:
        chunks = []
        # 1) from static documentation in BasicRAG
        for section in self.knowledge_base.values():
            if isinstance(section, dict):
                for sub in section.values():
                    for doc in sub:
                        chunks += self._chunk_text(doc)
            else:
                for doc in section:
                    chunks += self._chunk_text(doc)
        # 2) from code examples
        for code in OPENFHE_EXAMPLES.values():
            chunks += self._chunk_text(code)
        return chunks

    def _embed(self, texts: List[str]) -> np.ndarray:
        resp = openai.embeddings.create(
            model=self.embedding_model,
            input=texts
        )
        embs = [ np.array(item.embedding, dtype=np.float32) for item in resp.data ]
        return np.vstack(embs)


    def _build_index(self):
        # gather & embed
        self.chunks = self._gather_all_chunks()
        embeddings = self._embed(self.chunks)
        # build FAISS index
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)
        # save persistent files
        faiss.write_index(self.index, self.index_path)
        with open(self.map_path, "wb") as f:
            pickle.dump(self.chunks, f)

    def _load_index(self):
        self.index = faiss.read_index(self.index_path)
        with open(self.map_path, "rb") as f:
            self.chunks = pickle.load(f)

    def _search(self, query: str) -> List[str]:
        q_emb = self._embed([query])
        D, I = self.index.search(q_emb, self.top_k)
        return [ self.chunks[int(i)] for i in I[0] if i != -1 ]

    def generate_rag_prompt(self) -> str:
        # build a simple query
        top = self._search(f"CKKS {self.operation_type} operation reference")
        context = "\n\n".join(top)

        # rotation guidance only for ops that need it
        rotation_guidance = ""
        if self.operation_type in ["dot_product", "matrix_multiplication", "convolution"]:
            rotation_guidance = """
            IMPORTANT FOR ROTATIONS:
            1. You MUST enable advanced SHE:
            cc->Enable(ADVANCEDSHE);
            2. You MUST generate rotation keys *before* using any rotations or sums:
            - Option A: cc->EvalRotateKeyGen(keys.secretKey, {1, 2, /* … */});
            - Option B: cc->EvalSumKeyGen(keys.secretKey);
            """

        prompt = f"""
                You are an expert in FHE CKKS with OpenFHE.

                RELEVANT CONTEXT (from vector DB similarity search):
                {context}

                TASK: Implement CKKS {self.operation_type} operation using OpenFHE.

                Your program MUST:
                1. Include all necessary headers and `using namespace lbcrypto;`
                2. Set up parameters with CCParams<CryptoContextCKKSRNS>:
                    ```cpp
                    CCParams<CryptoContextCKKSRNS> parameters;
                    parameters.SetMultiplicativeDepth( /* e.g. 2+ */ );
                    parameters.SetScalingModSize( /* e.g. 40 */ );
                    parameters.SetBatchSize( /* >= your vector/matrix size */ );
                    parameters.SetSecurityLevel(SecurityLevel::HEStd_128_classic);
                    ```
                3. **Explicitly** create the CryptoContext:
                    ```cpp
                    CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);
                    cc->Enable(PKE);
                    cc->Enable(KEYSWITCH);
                    cc->Enable(LEVELEDSHE);
                    ```
                4. **If your operation requires rotations** (dot product / matrix multiplication / convolution), then also include:{rotation_guidance}
                5. Generate keys:
                    ```cpp
                    auto keys = cc->KeyGen();
                    cc->EvalMultKeyGen(keys.secretKey);
                    ```
                6. Encode & encrypt test data with `MakeCKKSPackedPlaintext` + `Encrypt`
                7. Perform the {self.operation_type} on ciphertexts (`EvalAdd`, `EvalMult`, etc.)
                8. Decrypt, call `plaintext->SetLength(...)`, use `GetRealPackedValue()`, and verify
                9. Include clear comments on every step

                Return a single, compilable C++ file.
                """
        return prompt
    

def get_rag_technique(operation_type: str, technique: str = "basic", knowledge_base_dir: Optional[str] = None,**kwargs):
    """Factory function to get the appropriate RAG technique."""
    if technique == "basic":
        return BasicRAG(operation_type, knowledge_base_dir)
    elif technique == "graph":
        return GraphRAG(operation_type, knowledge_base_dir)
    elif technique == "vector":
        return VectorRAG(operation_type, knowledge_base_dir, **kwargs)
    else:
        raise ValueError(f"Unsupported RAG technique: {technique}")