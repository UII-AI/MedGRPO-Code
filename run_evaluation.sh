#!/bin/bash
# MedGPRO Evaluation Script
# Evaluate inference results across all tasks

export TZ=America/New_York

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RESULTS_FILE="${1:-./results/results.json}"
TASKS="${2:-}"   # Optional: space-separated task list (tal stg next_action dvc cvs_assessment skill_assessment)

echo "=============================================="
echo "MedGPRO Evaluation"
echo "=============================================="
echo "Results: ${RESULTS_FILE}"
echo "=============================================="

CMD="python3 ${SCRIPT_DIR}/evaluation/evaluate_all_final.py ${RESULTS_FILE}"
if [ -n "${TASKS}" ]; then
    CMD="${CMD} --tasks ${TASKS}"
fi

eval $CMD
