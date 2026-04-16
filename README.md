# MedGRPO Inference (CVPR 2026)

Reference inference pipeline for **uAI-NEXUS-MedVLM-1.0a-7B-RL** — a medical video-understanding model trained with SFT + MedGRPO on Qwen2.5-VL-7B.

- **📄 Paper**: [arXiv:2512.06581](https://arxiv.org/abs/2512.06581)
- **🌐 Project Page**: [uii-ai.github.io/MedGRPO](https://uii-ai.github.io/MedGRPO/)
- **🤗 Model**: [UII-AI/uAI-NEXUS-MedVLM-1.0a-7B-RL](https://huggingface.co/UII-AI/uAI-NEXUS-MedVLM-1.0a-7B-RL)
- **🤗 Dataset**: [UII-AI/MedVidBench](https://huggingface.co/datasets/UII-AI/MedVidBench)
- **🎮 Demo**: [UII-AI/MedGRPO-Demo](https://huggingface.co/spaces/UII-AI/MedGRPO-Demo)
- **📊 Leaderboard**: [UII-AI/MedVidBench-Leaderboard](https://huggingface.co/spaces/UII-AI/MedVidBench-Leaderboard)

## Directory Structure

```
MedGPRO-Inference/
├── inference/
│   ├── vllm_infer.py              # VLLM batch inference engine
│   └── vision_process_medical.py  # Medical video frame processing (RC box support)
├── utils/
│   ├── split_data_balanced.py     # Split data across GPUs (balanced by task)
│   ├── merge_results_manual.py    # Merge per-GPU results
│   └── convert_to_submission.py   # Convert results.json → leaderboard submission
├── results/                       # Inference outputs (gitignored)
├── run_inference.sh               # Launcher (single- and multi-GPU)
└── requirements.txt
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download the model

```bash
huggingface-cli download UII-AI/uAI-NEXUS-MedVLM-1.0a-7B-RL \
    --local-dir models/uAI-NEXUS-MedVLM-1.0a-7B-RL
```

### 3. Download the test data

```bash
huggingface-cli download UII-AI/MedVidBench cleaned_test_data_11_04.json \
    --repo-type dataset --local-dir .
```

### 4. Run inference

```bash
bash run_inference.sh
# Output: results/results.json
```

Edit the `CONFIGURATION` block at the top of `run_inference.sh` to set `MODEL_PATH`, `DATA_PATH`, `N_GPUS`, and `GPUS`.

## Input Data Format

The inference script accepts the **SFT-style** JSON format (the same format used during training, with answers optional):

```json
[
  {
    "conversations": [
      {"from": "human", "value": "<video>\nQuestion text?"}
    ],
    "video": ["frame_0001.jpg", "frame_0002.jpg", "..."],
    "metadata": {
      "fps": "1.0",
      "video_id": "...",
      "input_video_start_frame": "0",
      "input_video_end_frame": "100"
    },
    "qa_type": "tal",
    "data_source": "AVOS",
    "struc_info": {},
    "is_RC": false,
    "RC_info": {}
  }
]
```

Ground-truth `{"from": "gpt", ...}` turns are ignored if present — the same file works for training data and test data.

**Supported `qa_type` values**: `tal`, `stg`, `next_action`, `dense_captioning`, `video_summary`, `region_caption`, `skill_assessment`, `cvs_assessment`.

## Usage

### Single-GPU

```bash
CUDA_VISIBLE_DEVICES=0 python3 inference/vllm_infer.py \
    --model_path ./models/uAI-NEXUS-MedVLM-1.0a-7B-RL \
    --data_path test_data.json \
    --output_path results.json \
    --batch_size 4 \
    --gpu_memory_utilization 0.85 \
    --max_new_tokens 256
```

### Multi-GPU

Use the launcher (recommended — handles splitting, parallel launch, and merging):

```bash
# Set N_GPUS and GPUS in run_inference.sh, then:
bash run_inference.sh
```

Or run the steps manually:

```bash
# 1. Split data across GPUs (balanced by task type)
python3 utils/split_data_balanced.py test_data.json 4

# 2. Launch parallel inference (one command per GPU)
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i python3 inference/vllm_infer.py \
      --model_path ./models/uAI-NEXUS-MedVLM-1.0a-7B-RL \
      --data_path test_data_gpu${i}.json \
      --output_path results/results_gpu${i}.json \
      --batch_size 4 --gpu_memory_utilization 0.85 --max_new_tokens 256 &
done
wait

# 3. Merge per-GPU results
python3 utils/merge_results_manual.py results/results.json 4
```

## Output Format

`vllm_infer.py` writes a dict keyed by sequential index:

```json
{
  "0": {
    "metadata": {"fps": "1.0", "video_id": "...", "...": "..."},
    "qa_type": "tal",
    "struc_info": {},
    "question": "When does cutting happen?",
    "answer": "<model prediction>",
    "data_source": "AVOS"
  },
  "1": {"...": "..."}
}
```

## Leaderboard Submission

The [MedVidBench Leaderboard](https://huggingface.co/spaces/UII-AI/MedVidBench-Leaderboard) expects a list with `{id, qa_type, prediction}` per entry. Convert `results.json` with:

```bash
python3 utils/convert_to_submission.py results/results.json submission.json
```

Example `submission.json`:

```json
[
  {"id": "video_001", "qa_type": "tal", "prediction": "The action starts at 5.2s and ends at 12.7s."},
  {"id": "video_002", "qa_type": "stg", "prediction": "..."}
]
```

Then upload `submission.json` via the leaderboard's **Step 1: Submit & Evaluate** form.

## Implementation Details

- `vision_process_medical.py` handles per-sample FPS and draws bounding boxes for region-caption samples (`is_RC: true`).
- Decoding is greedy (`temperature=0.0`) by default.
- The model path must be a local directory in HuggingFace format (use `huggingface-cli download` as shown above).

## Citation

If you use our model or benchmark (MedVidBench / uAI-NEXUS-MedVLM), please cite our paper:

```bibtex
@inproceedings{su2026medgrpo,
  title     = {{MedGRPO}: Multi-Task Reinforcement Learning for Heterogeneous Medical Video Understanding},
  author    = {Su, Yuhao and Choudhuri, Anwesa and Gao, Zhongpai and Planche, Benjamin and
               Nguyen, Van Nguyen and Zheng, Meng and Shen, Yuhan and Innanje, Arun and
               Chen, Terrence and Elhamifar, Ehsan and Wu, Ziyan},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2026}
}
```
