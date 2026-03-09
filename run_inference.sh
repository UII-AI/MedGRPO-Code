#!/bin/bash
# MedGPRO Qwen2.5-VL Inference Script
# Multi-GPU parallel inference for medical video understanding

export TZ=America/New_York
set -e

# ============================================================
# CONFIGURATION - Edit these
# ============================================================
MODEL_PATH="/path/to/your/model"          # HuggingFace format checkpoint
DATA_PATH="/path/to/test_data.json"       # Input JSON (SFT format)
OUTPUT_DIR="./results"
N_GPUS=6
GPUS=(0 1 2 3 4 5)
BATCH_SIZE=6
GPU_MEM_UTIL=0.85
MAX_NEW_TOKENS=256
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${OUTPUT_DIR}/logs"

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

echo "=============================================="
echo "MedGPRO Inference"
echo "=============================================="
echo "Model:   ${MODEL_PATH}"
echo "Data:    ${DATA_PATH}"
echo "Output:  ${OUTPUT_DIR}"
echo "GPUs:    ${GPUS[*]} (${N_GPUS} GPUs)"
echo "Batch:   ${BATCH_SIZE}"
echo "=============================================="

# Step 1: Split data across GPUs
echo "[1/4] Splitting data across ${N_GPUS} GPUs..."
python3 "${SCRIPT_DIR}/utils/split_data_balanced.py" "${DATA_PATH}" "${N_GPUS}"
echo "  Done"

# Step 2: Parallel inference
echo "[2/4] Launching parallel inference..."
declare -a PIDS

for i in "${!GPUS[@]}"; do
    GPU_ID=${GPUS[$i]}
    CUDA_VISIBLE_DEVICES=${GPU_ID} python3 "${SCRIPT_DIR}/inference/vllm_infer.py" \
        --model_path "${MODEL_PATH}" \
        --data_path "${DATA_PATH%.json}_gpu${i}.json" \
        --output_path "${OUTPUT_DIR}/results_gpu${GPU_ID}.json" \
        --batch_size "${BATCH_SIZE}" \
        --max_pixels_per_frame $((48*28*28)) \
        --min_pixels_per_frame $((8*28*28)) \
        --max_new_tokens "${MAX_NEW_TOKENS}" \
        --gpu_memory_utilization "${GPU_MEM_UTIL}" \
        2>&1 | tee "${LOG_DIR}/gpu${GPU_ID}.log" &
    PIDS[$i]=$!
    echo "  GPU ${GPU_ID}: PID ${PIDS[$i]}"
done

# Step 3: Wait
echo "[3/4] Waiting for all GPUs..."
ALL_SUCCESS=true
for i in "${!PIDS[@]}"; do
    GPU_ID=${GPUS[$i]}
    wait ${PIDS[$i]} && echo "  GPU ${GPU_ID} done" || { echo "  GPU ${GPU_ID} FAILED"; ALL_SUCCESS=false; }
done

[ "$ALL_SUCCESS" = false ] && { echo "One or more GPUs failed."; exit 1; }

# Step 4: Merge
echo "[4/4] Merging results..."
python3 "${SCRIPT_DIR}/utils/merge_results_manual.py" "${OUTPUT_DIR}/results.json" "${N_GPUS}"

echo "=============================================="
echo "Done! Output: ${OUTPUT_DIR}/results.json"
echo "=============================================="
