#!/usr/bin/env python3
"""
VLLM-based inference for Qwen2.5-VL medical video understanding.
Integrates medical vision processing with RC box drawing and VLLM for efficient inference.
"""

import os
import sys
import json
import argparse
import time
from collections import defaultdict
from typing import List, Dict, Any, Tuple
import tqdm
from PIL import Image

from vllm import LLM, SamplingParams

# Import vision processing from the same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vision_process_medical import process_vision_info_medical


def group_data_by_type(data_dicts: List[Dict]) -> Tuple[Dict[str, List], Dict[str, int]]:
    """
    Group data instances by their qa_type.
    Returns a dictionary where keys are qa_types and values are lists of data instances.
    """
    type_groups = defaultdict(list)
    type_counts = defaultdict(int)

    for idx, data_dict in enumerate(data_dicts):
        qa_type = data_dict.get('qa_type', 'unknown')
        # Add original index to track position in final output
        data_dict['original_idx'] = idx
        type_groups[qa_type].append(data_dict)
        type_counts[qa_type] += 1

    print("Found QA types and their counts:")
    for qa_type, count in type_counts.items():
        print(f"  {qa_type}: {count} instances")

    return dict(type_groups), dict(type_counts)


def prepare_messages_for_vllm(data_dict: Dict, max_pixels: int = None, min_pixels: int = None) -> Dict:
    """
    Prepare a single data instance into VLLM-compatible message format.
    Applies medical vision processing with RC box drawing support.
    """
    convs = data_dict['conversations']
    question = convs[0]['value'].replace("<video>\n", "")

    # Build video content with CORRECT parameter name
    video_content = {
        "type": "video",
        "video": data_dict['video'],  # List of frame paths
        "sample_fps": float(data_dict['metadata']['fps']),  # CRITICAL: Use "sample_fps"
    }

    # Add pixel settings if provided (controls video token count)
    if max_pixels is not None:
        video_content["max_pixels"] = max_pixels
    if min_pixels is not None:
        video_content["min_pixels"] = min_pixels

    # Add RC info if present (for region_caption tasks)
    if data_dict.get('is_RC', False) and 'RC_info' in data_dict:
        video_content['is_RC'] = True
        video_content['RC_info'] = data_dict['RC_info']

    # Build message in Qwen2.5-VL format
    message = {
        "role": "user",
        "content": [
            video_content,
            {"type": "text", "text": question},
        ],
    }

    return message, question


def preprocess_batch_videos(batch: List[Dict], processor, max_pixels: int = None, min_pixels: int = None) -> Tuple[List[str], List[List[Image.Image]], List[Dict]]:
    """
    Preprocess a batch of videos with medical vision processing (includes RC box drawing).
    Returns: (prompts, preprocessed_videos, metadata_list)
    """
    prompts = []
    video_frames_list = []
    metadata_list = []

    for data_dict in batch:
        # Prepare message
        message, question = prepare_messages_for_vllm(data_dict, max_pixels=max_pixels, min_pixels=min_pixels)

        # Get ground truth answer (if available)
        convs = data_dict['conversations']
        gnd = convs[1]['value'] if len(convs) > 1 else None

        # Apply medical vision processing (unified function with RC support)
        messages_list = [message]
        image_inputs, video_inputs, video_kwargs = process_vision_info_medical(
            messages_list,
            return_video_kwargs=True
        )

        # video_inputs is a list of PIL Images with timestamps overlaid
        if video_inputs and len(video_inputs) > 0:
            video_frames = video_inputs[0]  # First (and only) video in this message
            video_frames_list.append(video_frames)
        else:
            raise ValueError(f"No video frames found for data_dict: {data_dict.get('id', 'unknown')}")

        # Apply chat template to get the prompt (use tokenizer which has the chat template)
        # Create a text-only message for the chat template (video placeholder + question)
        text_message = {
            "role": "user",
            "content": f"<|vision_start|><|video_pad|><|vision_end|>{question}"
        }
        prompt = processor.tokenizer.apply_chat_template(
            [text_message],
            tokenize=False,
            add_generation_prompt=True
        )
        prompts.append(prompt)

        # Store metadata (exclude 'gnd' for test data without ground truth)
        meta = {
            'original_idx': data_dict['original_idx'],
            'metadata': data_dict.get('metadata', None),
            'qa_type': data_dict.get('qa_type', None),
            'struc_info': data_dict.get('struc_info', None),
            'question': question,
            'data_source': data_dict.get('data_source', None),
        }
        # Only include ground truth if available (training data)
        if gnd is not None:
            meta['gnd'] = gnd
        metadata_list.append(meta)

    return prompts, video_frames_list, metadata_list


def process_batch_vllm(
    batch: List[Dict],
    llm: LLM,
    processor,
    sampling_params: SamplingParams,
    max_pixels: int = None,
    min_pixels: int = None
) -> Dict[int, Dict]:
    """
    Process a batch using VLLM with custom preprocessing.
    """
    # Preprocess videos with timestamp overlays
    prompts, video_frames_list, metadata_list = preprocess_batch_videos(batch, processor, max_pixels=max_pixels, min_pixels=min_pixels)

    # Prepare VLLM inputs
    vllm_inputs = []
    for prompt, video_frames in zip(prompts, video_frames_list):
        vllm_inputs.append({
            "prompt": prompt,
            "multi_modal_data": {
                "video": video_frames  # List of PIL Images
            }
        })

    # Run VLLM inference
    outputs = llm.generate(vllm_inputs, sampling_params)

    # Process outputs
    batch_results = {}
    for output, metadata in zip(outputs, metadata_list):
        generated_text = output.outputs[0].text

        result = {
            'metadata': metadata['metadata'],
            'qa_type': metadata['qa_type'],
            'struc_info': metadata['struc_info'],
            'question': metadata['question'],
            'answer': generated_text,
            'data_source': metadata['data_source'],
        }
        # Only include ground truth if available (training data)
        if 'gnd' in metadata:
            result['gnd'] = metadata['gnd']
        batch_results[metadata['original_idx']] = result

    return batch_results


def process_type_group(
    llm: LLM,
    processor,
    type_data: List[Dict],
    qa_type: str,
    batch_size: int,
    sampling_params: SamplingParams,
    max_pixels: int = None,
    min_pixels: int = None
) -> Dict[int, Dict]:
    """
    Process a single QA type group.
    """
    type_results = {}

    print(f"\n=== Processing QA type: {qa_type} ({len(type_data)} instances) ===")

    # Process in batches
    for i in tqdm.tqdm(range(0, len(type_data), batch_size), desc=f"Processing {qa_type}"):
        batch = type_data[i:i + batch_size]
        batch_results = process_batch_vllm(batch, llm, processor, sampling_params, max_pixels=max_pixels, min_pixels=min_pixels)
        type_results.update(batch_results)

    return type_results


def main():
    parser = argparse.ArgumentParser(description="VLLM inference for Qwen2.5-VL medical videos")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--data_path", type=str, required=True, help="Path to input JSON data")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save output JSON")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for inference")
    parser.add_argument("--max_pixels_per_frame", type=int, default=48*28*28, help="Max pixels per frame")
    parser.add_argument("--min_pixels_per_frame", type=int, default=8*28*28, help="Min pixels per frame")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Tensor parallel size (num GPUs)")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8, help="GPU memory utilization")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Max tokens to generate")
    parser.add_argument("--limit_data", type=int, default=None, help="Limit number of data instances (for testing)")

    args = parser.parse_args()

    print("="*80)
    print("VLLM Inference Configuration")
    print("="*80)
    print(f"Model: {args.model_path}")
    print(f"Data: {args.data_path}")
    print(f"Output: {args.output_path}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max pixels per frame: {args.max_pixels_per_frame}")
    print(f"Min pixels per frame: {args.min_pixels_per_frame}")
    print(f"Tensor parallel size: {args.tensor_parallel_size}")
    print(f"GPU memory utilization: {args.gpu_memory_utilization}")
    print("="*80)

    # Load data
    with open(args.data_path, 'r') as f:
        data_dicts = json.load(f)

    if args.limit_data:
        data_dicts = data_dicts[:args.limit_data]
        print(f"Limited to {len(data_dicts)} instances for testing")

    # Group by type
    type_groups, type_counts = group_data_by_type(data_dicts)

    # Initialize VLLM
    print("\nInitializing VLLM...")
    start_time = time.time()

    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        max_model_len=32768,
        limit_mm_per_prompt={"image": 10, "video": 1},
        enforce_eager=True,  # Helps with memory stability
    )

    # Get processor and tokenizer for chat template
    from transformers import AutoProcessor, AutoTokenizer
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        padding_side="left",
        max_pixels=args.max_pixels_per_frame,
        min_pixels=args.min_pixels_per_frame,
    )

    # Load tokenizer - try model path first, fallback to base Qwen2.5-VL if no chat template
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.chat_template is None:
        print("Model has no chat template, loading from base Qwen2.5-VL-7B-Instruct...")
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", trust_remote_code=True)

    # Store tokenizer in processor for use in preprocess function
    processor.tokenizer = tokenizer

    print(f"VLLM initialized in {time.time() - start_time:.2f}s")

    # Sampling params
    sampling_params = SamplingParams(
        temperature=0.0,  # Greedy decoding
        max_tokens=args.max_new_tokens,
    )

    # Process each type sequentially
    all_results = {}
    inference_start = time.time()

    for qa_type in type_groups.keys():
        type_data = type_groups[qa_type]
        type_results = process_type_group(
            llm=llm,
            processor=processor,
            type_data=type_data,
            qa_type=qa_type,
            batch_size=args.batch_size,
            sampling_params=sampling_params,
            max_pixels=args.max_pixels_per_frame,
            min_pixels=args.min_pixels_per_frame
        )
        all_results.update(type_results)
        print(f"Completed {qa_type}: {len(type_results)} instances")

    inference_time = time.time() - inference_start

    # Sort results by original index
    sorted_results = dict(sorted(all_results.items()))

    # Save results
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, 'w') as f:
        json.dump(sorted_results, f, indent=4)

    print("\n" + "="*80)
    print("Inference Complete!")
    print("="*80)
    print(f"Total instances: {len(sorted_results)}")
    print(f"Inference time: {inference_time:.2f}s")
    print(f"Average time per instance: {inference_time/len(sorted_results):.2f}s")
    print(f"Output saved to: {args.output_path}")
    print("="*80)


if __name__ == "__main__":
    main()
