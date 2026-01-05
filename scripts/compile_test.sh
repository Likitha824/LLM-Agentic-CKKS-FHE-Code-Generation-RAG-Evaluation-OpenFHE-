#!/bin/bash
# Batch Compilability Testing Script - Runs only compilability tests for all operations

# Command line parameters
MODEL="${1:-openai}"
TECHNIQUE="${2:-self_improvement_iterative}"

# All operations to test
OPERATIONS=("addition" "multiplication" "dot_product" "matrix_multiplication" "convolution")

# Create results directory
mkdir -p "results/metrics/${MODEL}/${TECHNIQUE}"

echo "======================================================================"
echo "BATCH COMPILABILITY TESTING for $MODEL with $TECHNIQUE"
echo "======================================================================"

# Run compilability tests for all operations
for OPERATION in "${OPERATIONS[@]}"; do
  echo "--------------------------------------------------------------------"
  echo "Testing compilability for $OPERATION..."
  echo "--------------------------------------------------------------------"
  
  DIR_PATH="generated_code/$MODEL/$TECHNIQUE/$OPERATION"
  
  # Check if directory exists
  if [ ! -d "$DIR_PATH" ]; then
    echo "WARNING: Directory $DIR_PATH does not exist. Skipping..."
    continue
  fi
  
  # Initialize results
  TOTAL_FILES=0
  COMPILABLE_COUNT=0
  RESULTS="{\"results\": {"
  
  # Loop through all cpp files
  for CPP_FILE in "$DIR_PATH"/*.cpp; do
    # Check if files exist
    if [ ! -e "$CPP_FILE" ]; then
      echo "No .cpp files found in $DIR_PATH. Skipping..."
      break
    fi
    
    FILENAME=$(basename "$CPP_FILE")
    BASE_NAME="${FILENAME%.cpp}"
    OUTPUT_EXE="$DIR_PATH/${BASE_NAME}.exe"
    
    echo "====================================================="
    echo "Testing: $FILENAME"
    echo "====================================================="
  # Check if the file already defines _USE_MATH_DEFINES
  if grep -q "#define[[:space:]]\+_USE_MATH_DEFINES" "$CPP_FILE"; then
    MATH_DEFINE_OPTION=""
  else
    MATH_DEFINE_OPTION="-D_USE_MATH_DEFINES"
  fi
    # Try to compile; generate an executable with the same base name
    g++ "$CPP_FILE" \
       $MATH_DEFINE_OPTION  \
      -I/home/likit/fhelibraries/include \
      -I/home/likit/fhelibraries/include/openfhe \
      -I/home/likit/fhelibraries/include/openfhe/pke \
      -I/home/likit/fhelibraries/include/openfhe/core \
      -I/home/likit/fhelibraries/include/openfhe/binfhe \
      -I/home/likit/fhelibraries/include/openfhe/cereal \
      -L/home/likit/fhelibraries/lib \
      -lopenfhecore -lopenfhepke -lopenfhebinfhe \
      -std=c++17 \
      -o "$OUTPUT_EXE" > compile_output.txt 2>&1
    
    COMPILE_STATUS=$?
    
    if [ $COMPILE_STATUS -eq 0 ]; then
      echo "✅ COMPILATION SUCCESSFUL!"
      RESULTS="$RESULTS\"$FILENAME\": {\"compiles\": true, \"error\": \"\"},"
      COMPILABLE_COUNT=$((COMPILABLE_COUNT+1))
    else
      echo "❌ COMPILATION FAILED!"
      echo "Error details:"
      echo "-----------------------------------------------------"
      cat compile_output.txt
      echo "-----------------------------------------------------"
      
      # Escape special characters for JSON
      ERROR=$(cat compile_output.txt | sed 's/\\/\\\\/g' | sed 's/"/\\"/g' | tr '\n' ' ')
      RESULTS="$RESULTS\"$FILENAME\": {\"compiles\": false, \"error\": \"$ERROR\"},"
      
      # Remove the executable if compilation failed
      rm -f "$OUTPUT_EXE"
    fi
    
    TOTAL_FILES=$((TOTAL_FILES+1))
    
    # Clean up compile output; don't remove the executable
    rm -f compile_output.txt
    echo ""
  done
  
  # If we found files to compile
  if [ $TOTAL_FILES -gt 0 ]; then
    # Finalize JSON
    RESULTS="${RESULTS%,}}"
    PASS_AT_1=0
    if [ $COMPILABLE_COUNT -gt 0 ]; then
      PASS_AT_1=1
    fi
    
    RESULTS="$RESULTS, \"pass_at_1\": $PASS_AT_1, \"compilable_count\": $COMPILABLE_COUNT, \"total_samples\": $TOTAL_FILES}"
    
    # Save results
    echo "$RESULTS" > "results/metrics/${MODEL}/${TECHNIQUE}/compilability_${OPERATION}.json"
    
    # Print summary
    echo "****************************************************************"
    echo "SUMMARY: Compilability results for $MODEL with $TECHNIQUE on $OPERATION:"
    echo "****************************************************************"
    echo "Pass@1: $PASS_AT_1 ($([ $PASS_AT_1 -eq 1 ] && echo "PASS" || echo "FAIL"))"
    echo "Compilable: $COMPILABLE_COUNT/$TOTAL_FILES"
    echo ""
    echo "Detailed results:"
    for CPP_FILE in "$DIR_PATH"/*.cpp; do
      FILENAME=$(basename "$CPP_FILE")
      if grep -q "\"$FILENAME\": {\"compiles\": true" <<< "$RESULTS"; then
        echo "  - $FILENAME: ✅ PASS"
      else
        echo "  - $FILENAME: ❌ FAIL"
      fi
    done
  fi
done

echo "======================================================================"
echo "BATCH COMPILABILITY TESTING COMPLETE for $MODEL with $TECHNIQUE"
echo "======================================================================"
echo "All results saved to results/metrics/"