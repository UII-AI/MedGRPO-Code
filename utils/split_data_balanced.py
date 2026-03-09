#!/usr/bin/env python3
"""
Split data with balanced task distribution across multiple GPUs.
Ensures each GPU gets roughly equal amounts of each task type (TAL/STG/Next-Action).
"""
import json
import sys
from collections import defaultdict

def split_data_balanced(input_file, n_gpus):
    """
    Split data ensuring balanced task distribution across GPUs.

    Algorithm:
    1. Group data by qa_type
    2. For each type, distribute evenly across GPUs in round-robin fashion
    3. Shuffle assignment to avoid sequential patterns
    """
    with open(input_file, 'r') as f:
        data = json.load(f)

    print(f"Total instances: {len(data)}")

    # Group by qa_type
    type_groups = defaultdict(list)
    for item in data:
        qa_type = item.get('qa_type', 'unknown')
        type_groups[qa_type].append(item)

    print("\nTask distribution in input:")
    for qa_type, items in sorted(type_groups.items()):
        print(f"  {qa_type}: {len(items)}")

    # Initialize GPU splits
    gpu_splits = [[] for _ in range(n_gpus)]

    # Distribute each task type round-robin across GPUs
    for qa_type, items in type_groups.items():
        for i, item in enumerate(items):
            gpu_id = i % n_gpus
            gpu_splits[gpu_id].append(item)

    # Save splits
    print("\nDistribution across GPUs:")
    for i, split in enumerate(gpu_splits):
        # Count tasks per GPU
        gpu_task_counts = defaultdict(int)
        for item in split:
            gpu_task_counts[item.get('qa_type', 'unknown')] += 1

        output_file = input_file.replace('.json', f'_gpu{i}.json')
        with open(output_file, 'w') as f:
            json.dump(split, f, indent=2)

        task_summary = ", ".join([f"{k}={v}" for k, v in sorted(gpu_task_counts.items())])
        print(f"GPU {i}: {len(split):4d} instances ({task_summary}) -> {output_file}")

    # Verification
    total_after_split = sum(len(split) for split in gpu_splits)
    assert total_after_split == len(data), f"Data loss! Before: {len(data)}, After: {total_after_split}"
    print(f"\n✓ Verification passed: {total_after_split} instances distributed across {n_gpus} GPUs")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input_json> <n_gpus>")
        print(f"\nExample:")
        print(f"  {sys.argv[0]} test_data.json 6")
        sys.exit(1)

    split_data_balanced(sys.argv[1], int(sys.argv[2]))
