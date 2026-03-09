# MedGPRO-Inference

Inference and evaluation pipeline for **MedGPRO** (Qwen2.5-VL 7B fine-tuned on medical video understanding via GRPO/DAPO).

## Directory Structure

```
MedGPRO-Inference/
├── inference/
│   ├── vllm_infer.py              # VLLM batch inference engine
│   └── vision_process_medical.py  # Medical video frame processing (RC box support)
├── evaluation/
│   ├── evaluate_all_final.py      # Main evaluation script (all tasks)
│   ├── eval_tal.py                # Temporal Action Localization
│   ├── eval_stg.py                # Spatiotemporal Grounding
│   ├── eval_next_action.py        # Next Action Prediction
│   ├── eval_dvc.py                # Dense Video Captioning
│   ├── eval_skill_assessment.py   # Surgical Skill Assessment
│   ├── eval_cvs_assessment.py     # CVS Assessment
│   ├── eval_rc_vs.py              # Region Caption & Video Summary
│   └── dataset_utils.py           # Dataset utilities
├── utils/
│   ├── split_data_balanced.py     # Split data across GPUs (balanced by task)
│   └── merge_results_manual.py    # Merge per-GPU results
├── run_inference.sh               # End-to-end inference launcher
├── run_evaluation.sh              # Evaluation launcher
└── requirements.txt
```

## Input Data Format

The inference script expects JSON files in **SFT format**:

```json
[
  {
    "conversations": [
      {"from": "human", "value": "<video>\nQuestion text?"},
      {"from": "gpt",   "value": "Ground truth answer"}
    ],
    "video": ["frame_0001.jpg", "frame_0002.jpg", ...],
    "metadata": {
      "fps": "1.0",
      "video_id": "...",
      "input_video_start_frame": "0",
      "input_video_end_frame": "100"
    },
    "qa_type": "tal",
    "data_source": "AVOS",
    "struc_info": {...},
    "is_RC": false,
    "RC_info": {}
  }
]
```

**Supported tasks** (`qa_type`): `tal`, `stg`, `next_action`, `dense_captioning`, `video_summary`, `region_caption`, `skill_assessment`, `cvs_assessment`

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run inference (multi-GPU)

Edit `run_inference.sh` to set `MODEL_PATH`, `DATA_PATH`, `N_GPUS`, then:

```bash
bash run_inference.sh
# Output: results/results.json
```

### 3. Run evaluation

```bash
bash run_evaluation.sh results/results.json
```

Or evaluate specific tasks:

```bash
bash run_evaluation.sh results/results.json "tal stg next_action"
```

## Manual Usage

### Single-GPU inference

```bash
CUDA_VISIBLE_DEVICES=0 python3 inference/vllm_infer.py \
    --model_path /path/to/model \
    --data_path test_data.json \
    --output_path results.json \
    --batch_size 4 \
    --gpu_memory_utilization 0.85 \
    --max_new_tokens 256
```

### Multi-GPU inference (manual)

```bash
# Split data
python3 utils/split_data_balanced.py test_data.json 4

# Run on each GPU in parallel
CUDA_VISIBLE_DEVICES=0 python3 inference/vllm_infer.py --data_path test_data_gpu0.json --output_path results/results_gpu0.json ... &
CUDA_VISIBLE_DEVICES=1 python3 inference/vllm_infer.py --data_path test_data_gpu1.json --output_path results/results_gpu1.json ... &
CUDA_VISIBLE_DEVICES=2 python3 inference/vllm_infer.py --data_path test_data_gpu2.json --output_path results/results_gpu2.json ... &
CUDA_VISIBLE_DEVICES=3 python3 inference/vllm_infer.py --data_path test_data_gpu3.json --output_path results/results_gpu3.json ... &
wait

# Merge results
python3 utils/merge_results_manual.py results/results.json 4
```

### Evaluate results

```bash
python3 evaluation/evaluate_all_final.py results/results.json
```

## Output Format

Inference saves results as a dict keyed by sequential index:

```json
{
  "0": {
    "metadata": {"fps": "1.0", "video_id": "...", ...},
    "qa_type": "tal",
    "struc_info": {...},
    "question": "When does cutting happen?",
    "gnd": "0.0–10.0 seconds.",
    "answer": "<model prediction>",
    "data_source": "AVOS"
  },
  "1": {...}
}
```

## Notes

- `vision_process_medical.py` handles per-sample FPS and draws bounding boxes for region caption tasks (`is_RC=True`)
- Uses greedy decoding (`temperature=0.0`) by default
- Model must be in HuggingFace format; use the FSDP→HF converter if starting from a GRPO checkpoint
