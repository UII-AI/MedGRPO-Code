#!/bin/bash
# MedGPRO Qwen2.5-VL One-Shot Inference Script
# Same as run_inference.sh but adds a per-task one-shot example
# (system message) to teach the leaderboard's expected output format.
# CVS examples are randomized per-sample (seeded by index) to avoid
# all-zero answer bias.
#
# One-shot examples are loaded from inference/oneshot_examples.json
# (curated, version-controlled — no train data dependency).

export TZ=America/New_York
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================================================
# CONFIGURATION - Edit these
# ============================================================
MODEL_PATH="${SCRIPT_DIR}/models/uAI-NEXUS-MedVLM-1.0a-7B-RL"
DATA_PATH="/root/code/MedVidBench/cleaned_test_data_11_04.json"
EXAMPLES_PATH="${SCRIPT_DIR}/inference/oneshot_examples.json"
OUTPUT_DIR="${SCRIPT_DIR}/results/oneshot"
N_GPUS=1          # Set to 1 for single-GPU, or N for multi-GPU
GPUS=(0)           # List of GPU IDs to use, e.g. (0 1 2 3)
BATCH_SIZE=6
GPU_MEM_UTIL=0.85
MAX_NEW_TOKENS=256
# ============================================================
LOG_DIR="${OUTPUT_DIR}/logs"

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

echo "=============================================="
echo "MedGPRO One-Shot Inference"
echo "=============================================="
echo "Model:    ${MODEL_PATH}"
echo "Data:     ${DATA_PATH}"
echo "Examples: ${EXAMPLES_PATH}"
echo "Output:   ${OUTPUT_DIR}"
echo "GPUs:     ${GPUS[*]} (${N_GPUS} GPU(s))"
echo "Batch:    ${BATCH_SIZE}"
echo "=============================================="

if [ "${N_GPUS}" -eq 1 ]; then
    # ── Single-GPU: run directly, no split/merge needed ──────────────────
    GPU_ID=${GPUS[0]}
    echo "[1/1] Running one-shot inference on GPU ${GPU_ID}..."
    CUDA_VISIBLE_DEVICES=${GPU_ID} python3 "${SCRIPT_DIR}/inference/vllm_infer_oneshot.py" \
        --model_path "${MODEL_PATH}" \
        --data_path "${DATA_PATH}" \
        --examples_path "${EXAMPLES_PATH}" \
        --output_path "${OUTPUT_DIR}/results.json" \
        --batch_size "${BATCH_SIZE}" \
        --max_pixels_per_frame $((48*28*28)) \
        --min_pixels_per_frame $((8*28*28)) \
        --max_new_tokens "${MAX_NEW_TOKENS}" \
        --gpu_memory_utilization "${GPU_MEM_UTIL}" \
        2>&1 | tee "${LOG_DIR}/gpu${GPU_ID}.log"
else
    # ── Multi-GPU: split → parallel inference → merge ────────────────────
    echo "[1/4] Splitting data across ${N_GPUS} GPUs..."
    python3 "${SCRIPT_DIR}/utils/split_data_balanced.py" "${DATA_PATH}" "${N_GPUS}"
    echo "  Done"

    echo "[2/4] Launching parallel inference..."
    declare -a PIDS
    for i in "${!GPUS[@]}"; do
        GPU_ID=${GPUS[$i]}
        CUDA_VISIBLE_DEVICES=${GPU_ID} python3 "${SCRIPT_DIR}/inference/vllm_infer_oneshot.py" \
            --model_path "${MODEL_PATH}" \
            --data_path "${DATA_PATH%.json}_gpu${i}.json" \
            --examples_path "${EXAMPLES_PATH}" \
            --output_path "${OUTPUT_DIR}/results_gpu${i}.json" \
            --batch_size "${BATCH_SIZE}" \
            --max_pixels_per_frame $((48*28*28)) \
            --min_pixels_per_frame $((8*28*28)) \
            --max_new_tokens "${MAX_NEW_TOKENS}" \
            --gpu_memory_utilization "${GPU_MEM_UTIL}" \
            2>&1 | tee "${LOG_DIR}/gpu${GPU_ID}.log" &
        PIDS[$i]=$!
        echo "  GPU ${GPU_ID}: PID ${PIDS[$i]}"
    done

    echo "[3/4] Waiting for all GPUs..."
    ALL_SUCCESS=true
    for i in "${!PIDS[@]}"; do
        GPU_ID=${GPUS[$i]}
        wait ${PIDS[$i]} && echo "  GPU ${GPU_ID} done" || { echo "  GPU ${GPU_ID} FAILED"; ALL_SUCCESS=false; }
    done
    [ "$ALL_SUCCESS" = false ] && { echo "One or more GPUs failed."; exit 1; }

    echo "[4/4] Merging results..."
    python3 "${SCRIPT_DIR}/utils/merge_results_manual.py" "${OUTPUT_DIR}/results.json" "${N_GPUS}"
fi

echo "=============================================="
echo "Done! Output: ${OUTPUT_DIR}/results.json"
echo "Convert: python3 utils/convert_to_submission.py ${OUTPUT_DIR}/results.json ${OUTPUT_DIR}/submission.json"
echo "=============================================="
