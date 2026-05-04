"""
One-shot variant of vllm_infer.py.

Mirrors the system-instruction one-shot pattern from
/root/code/LlamaFactory/medical_finetune/eval/qwen3_6_vllm_infer.py
to teach a non-finetuned model the leaderboard's expected output format
(fixes CVS_acc=0, SA_acc=0, STG_mIoU=0 caused by format drift).

One-shot examples are baked into inference/oneshot_examples.json
(curated, version-controlled — no dependency on private train data).

Usage:
    python3 inference/vllm_infer_oneshot.py \\
        --model_path models/Lingshu-7B \\
        --data_path /root/code/MedVidBench/cleaned_test_data_11_04.json \\
        --output_path results/lingshu7b_oneshot/results.json

    # Override the bundled examples file:
    python3 inference/vllm_infer_oneshot.py ... \\
        --examples_path /path/to/custom_examples.json
"""

import argparse
import json
import os
import random
import sys
import time
from typing import Dict, List, Tuple

import tqdm
from PIL import Image
from vllm import LLM, SamplingParams

# Reuse helpers from base inference script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vllm_infer import group_data_by_type  # noqa: E402
from vision_process_medical import process_vision_info_medical  # noqa: E402


# ─── One-shot example loading ─────────────────────────────────────────────────

DEFAULT_EXAMPLES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "oneshot_examples.json")


def load_oneshot_examples(path: str = None) -> dict:
    """Load curated {qa_type: {question, answer}} examples from JSON."""
    if path is None:
        path = DEFAULT_EXAMPLES_PATH
    print(f"[OneShotInfer] Loading one-shot examples from: {path}")
    with open(path) as f:
        examples = json.load(f)
    print(f"[OneShotInfer] Loaded {len(examples)} task types: {sorted(examples.keys())}")
    return examples


def _synthetic_cvs_example(idx: int) -> dict:
    """Randomized CVS one-shot to avoid all-zero answer bias from a fixed real example."""
    rng = random.Random(idx)
    a, b, c = rng.randint(0, 2), rng.randint(0, 2), rng.randint(0, 2)
    question = (
        "You are an expert surgical analyst. The video comes from Cholec80-CVS "
        "and is for evaluating Strasberg's Critical View of Safety. For this "
        "laparoscopic cholecystectomy procedure, evaluate the Critical View of "
        "Safety based on the three essential criteria: proper identification of "
        "two structures, adequate cystic plate exposure, and complete "
        "hepatocystic triangle clearance. Provide scores (0,1,2) for each criterion."
    )
    answer = f"Two structures: {a}, Cystic plate: {b}, Hepatocystic triangle: {c}"
    return {"question": question, "answer": answer}


def build_system_instruction(qa_type: str, oneshot_examples: dict, idx: int) -> str | None:
    if qa_type == "cvs_assessment":
        example = _synthetic_cvs_example(idx)
    else:
        example = oneshot_examples.get(qa_type)
    if example is None:
        return None
    return (
        "You are an expert medical video analyst. "
        "Below is an example of the expected question and answer format for this task.\n\n"
        "--- Example ---\n"
        f"Question: {example['question']}\n\n"
        f"Answer: {example['answer']}\n"
        "--- End Example ---\n\n"
        "Follow the same answer format exactly. Be concise and precise."
    )


# ─── Per-sample preprocessing (replaces vllm_infer.prepare_messages_for_vllm) ─

def prepare_messages_oneshot(data_dict: Dict, max_pixels: int, min_pixels: int):
    """Build the user message dict (video + question), same as vllm_infer.py."""
    convs = data_dict["conversations"]
    question = convs[0]["value"].replace("<video>\n", "")

    video_content = {
        "type": "video",
        "video": data_dict["video"],
        "sample_fps": float(data_dict["metadata"]["fps"]),
    }
    if max_pixels is not None:
        video_content["max_pixels"] = max_pixels
    if min_pixels is not None:
        video_content["min_pixels"] = min_pixels
    if data_dict.get("is_RC", False) and "RC_info" in data_dict:
        video_content["is_RC"] = True
        video_content["RC_info"] = data_dict["RC_info"]

    message = {
        "role": "user",
        "content": [video_content, {"type": "text", "text": question}],
    }
    return message, question


def preprocess_batch_oneshot(
    batch: List[Dict],
    processor,
    oneshot_examples: dict,
    max_pixels: int,
    min_pixels: int,
) -> Tuple[List[str], List[List[Image.Image]], List[Dict]]:
    """Build prompts (with optional system message containing the one-shot example)
    and process video frames (with RC bbox support). Returns (prompts, video_frames, metadata).
    """
    prompts, video_frames_list, metadata_list = [], [], []

    for data_dict in batch:
        message, question = prepare_messages_oneshot(data_dict, max_pixels, min_pixels)
        convs = data_dict["conversations"]
        gnd = convs[1]["value"] if len(convs) > 1 else None

        # Process video (RC boxes drawn here)
        image_inputs, video_inputs, video_kwargs = process_vision_info_medical(
            [message], return_video_kwargs=True
        )
        if not video_inputs:
            raise ValueError(f"No video frames for {data_dict.get('id', 'unknown')}")
        video_frames_list.append(video_inputs[0])

        # Build chat template input — text-only message that vllm will pair with video tokens
        text_user = {
            "role": "user",
            "content": f"<|vision_start|><|video_pad|><|vision_end|>{question}",
        }
        qa_type = data_dict.get("qa_type", "unknown")
        idx = data_dict.get("original_idx", 0)
        sys_instr = build_system_instruction(qa_type, oneshot_examples, idx)

        if sys_instr is not None:
            messages = [{"role": "system", "content": sys_instr}, text_user]
        else:
            messages = [text_user]

        prompt = processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompts.append(prompt)

        meta = {
            "original_idx": data_dict["original_idx"],
            "metadata": data_dict.get("metadata", None),
            "qa_type": qa_type,
            "struc_info": data_dict.get("struc_info", {}),
            "question": question,
            "data_source": data_dict.get("data_source", ""),
        }
        if gnd is not None:
            meta["gnd"] = gnd
        metadata_list.append(meta)

    return prompts, video_frames_list, metadata_list


def process_batch_oneshot(batch, llm, processor, sampling_params, oneshot_examples, max_pixels, min_pixels):
    prompts, video_frames_list, metadata_list = preprocess_batch_oneshot(
        batch, processor, oneshot_examples, max_pixels, min_pixels
    )
    vllm_inputs = [
        {"prompt": p, "multi_modal_data": {"video": v}}
        for p, v in zip(prompts, video_frames_list)
    ]
    outputs = llm.generate(vllm_inputs, sampling_params)

    batch_results = {}
    for output, metadata in zip(outputs, metadata_list):
        result = {
            "metadata": metadata["metadata"],
            "qa_type": metadata["qa_type"],
            "struc_info": metadata["struc_info"],
            "question": metadata["question"],
            "answer": output.outputs[0].text,
            "data_source": metadata["data_source"],
        }
        if "gnd" in metadata:
            result["gnd"] = metadata["gnd"]
        batch_results[metadata["original_idx"]] = result
    return batch_results


def process_type_group_oneshot(
    llm, processor, type_data, qa_type, batch_size, sampling_params,
    oneshot_examples, max_pixels, min_pixels,
):
    print(f"\n=== Processing QA type: {qa_type} ({len(type_data)} instances) ===")
    type_results = {}
    for i in tqdm.tqdm(range(0, len(type_data), batch_size), desc=f"Processing {qa_type}"):
        batch = type_data[i:i + batch_size]
        type_results.update(process_batch_oneshot(
            batch, llm, processor, sampling_params,
            oneshot_examples, max_pixels, min_pixels,
        ))
    return type_results


def main():
    parser = argparse.ArgumentParser(description="VLLM one-shot inference for Qwen2.5-VL medical videos")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--examples_path", type=str, default=None,
                        help="Path to oneshot examples JSON (default: inference/oneshot_examples.json)")
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_pixels_per_frame", type=int, default=48 * 28 * 28)
    parser.add_argument("--min_pixels_per_frame", type=int, default=8 * 28 * 28)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--limit_data", type=int, default=None)
    args = parser.parse_args()

    print("=" * 80)
    print("VLLM One-Shot Inference")
    print("=" * 80)
    print(f"Model:       {args.model_path}")
    print(f"Test data:   {args.data_path}")
    print(f"Examples:    {args.examples_path or DEFAULT_EXAMPLES_PATH}")
    print(f"Output:      {args.output_path}")
    print(f"Batch size:  {args.batch_size}")
    print("=" * 80)

    # Load test data and one-shot examples
    with open(args.data_path) as f:
        data_dicts = json.load(f)
    if args.limit_data:
        data_dicts = data_dicts[:args.limit_data]
        print(f"Limited to {len(data_dicts)} instances for testing")
    oneshot_examples = load_oneshot_examples(args.examples_path)

    type_groups, _ = group_data_by_type(data_dicts)

    print("\nInitializing VLLM...")
    t0 = time.time()
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        max_model_len=32768,
        limit_mm_per_prompt={"image": 10, "video": 1},
        enforce_eager=True,
    )
    from transformers import AutoProcessor, AutoTokenizer
    processor = AutoProcessor.from_pretrained(
        args.model_path, padding_side="left",
        max_pixels=args.max_pixels_per_frame,
        min_pixels=args.min_pixels_per_frame,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.chat_template is None:
        print("Model has no chat template, loading from base Qwen2.5-VL-7B-Instruct...")
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", trust_remote_code=True)
    processor.tokenizer = tokenizer
    print(f"VLLM initialized in {time.time() - t0:.2f}s")

    sampling_params = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens)

    all_results = {}
    inf_start = time.time()
    for qa_type in type_groups.keys():
        type_results = process_type_group_oneshot(
            llm=llm, processor=processor,
            type_data=type_groups[qa_type], qa_type=qa_type,
            batch_size=args.batch_size, sampling_params=sampling_params,
            oneshot_examples=oneshot_examples,
            max_pixels=args.max_pixels_per_frame,
            min_pixels=args.min_pixels_per_frame,
        )
        all_results.update(type_results)
        print(f"Completed {qa_type}: {len(type_results)} instances")

    inf_time = time.time() - inf_start
    sorted_results = dict(sorted(all_results.items()))

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(sorted_results, f, indent=4)

    print("\n" + "=" * 80)
    print("One-Shot Inference Complete!")
    print("=" * 80)
    print(f"Total instances: {len(sorted_results)}")
    print(f"Inference time:  {inf_time:.2f}s")
    print(f"Avg per inst:    {inf_time/len(sorted_results):.2f}s")
    print(f"Output:          {args.output_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
