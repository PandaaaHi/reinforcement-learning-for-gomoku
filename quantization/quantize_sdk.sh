#!bin/bash

GREEN='\033[0;32m'
NC='\033[0m'

### conversion
echo "${GREEN}[INFO] conversion ${NC}"
INPUT_NETWORK="./outputs_quantsim/w8a16.aimet/w8a16.onnx"
QUANTIZATION_OVERRIDES="./outputs_quantsim/w8a16.aimet/w8a16.encodings"
OUTPUT_PATH="./outputs_sdk/w8a16.dlc"
qairt-converter \
    --input_network "${INPUT_NETWORK}" \
    --quantization_overrides "${QUANTIZATION_OVERRIDES}" \
    --output_path "${OUTPUT_PATH}"

### quantization
# echo "${GREEN}[INFO] quantization ${NC}"
# INPUT_DLC="./outputs_sdk/w8a16.dlc"
# OUTPUT_DLC="./outputs_sdk/w8a16.dlc.quantized"
# qairt-quantizer \
#     --input_dlc "${INPUT_DLC}" \
#     --output_dlc "${OUTPUT_DLC}" \
#     --float_fallback

### compilation
# echo "${GREEN}[INFO] compilation ${NC}"
# MODEL="/home/panda/data/qualcomm/qairt/2.48.0.260626/lib/x86_64-linux-clang/libQnnModelDlc.so"
# BACKEND="/home/panda/data/qualcomm/qairt/2.48.0.260626/lib/x86_64-linux-clang/libQnnHtp.so"
# DLC_PATH="./outputs_sdk/w8a16.dlc"
# OUTPUT_DIR="./outputs_sdk/"
# BINARY_FILE="w8a16"
# qnn-context-binary-generator \
#     --model "${MODEL}" \
#     --backend "${BACKEND}" \
#     --dlc_path "${DLC_PATH}" \
#     --output_dir "${OUTPUT_DIR}" \
#     --binary_file "${BINARY_FILE}"

### execution
# echo "${GREEN}[INFO] execution ${NC}"
# BACKEND="/home/panda/data/qualcomm/qairt/2.48.0.260626/lib/x86_64-linux-clang/libQnnHtp.so"
# RETRIEVE_CONTEXT="./outputs_sdk/w8a16.bin"
# INPUT_LIST="./data/states_path.txt"
# OUTPUT_DIR="./outputs_sdk/"
# qnn-net-run \
#     --backend "${BACKEND}" \
#     --retrieve_context "${RETRIEVE_CONTEXT}" \
#     --input_list "${INPUT_LIST}" \
#     --output_dir "${OUTPUT_DIR}"