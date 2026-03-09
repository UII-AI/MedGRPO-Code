#!/usr/bin/env python3
"""Merge results from multiple GPUs with proper reindexing"""
import json
import sys

def merge_results(base_output, n_gpus):
    """
    Merge results from multiple GPU files into a single output.
    Handles dict format with string keys ("0", "1", ...) and reindexes sequentially.
    """
    all_results = {}
    current_idx = 0

    for i in range(n_gpus):
        gpu_file = base_output.replace('.json', f'_gpu{i}.json')
        try:
            with open(gpu_file, 'r') as f:
                gpu_results = json.load(f)

            # Handle both list and dict formats
            if isinstance(gpu_results, list):
                # List format: convert to dict
                for item in gpu_results:
                    all_results[str(current_idx)] = item
                    current_idx += 1
                print(f"Loaded {len(gpu_results)} results from {gpu_file} (list format)")

            elif isinstance(gpu_results, dict):
                # Dict format: reindex from GPU's keys to sequential
                # Sort by integer value of keys to maintain order
                sorted_keys = sorted(gpu_results.keys(), key=lambda x: int(x))
                for old_key in sorted_keys:
                    all_results[str(current_idx)] = gpu_results[old_key]
                    current_idx += 1
                print(f"Loaded {len(gpu_results)} results from {gpu_file} (dict format, reindexed)")

            else:
                print(f"Warning: {gpu_file} has unexpected format (not list or dict), skipping")

        except FileNotFoundError:
            print(f"Warning: {gpu_file} not found, skipping")
        except json.JSONDecodeError as e:
            print(f"Warning: {gpu_file} is not valid JSON: {e}, skipping")

    # Save merged results
    with open(base_output, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nMerge complete!")
    print(f"  Total results: {len(all_results)}")
    print(f"  Output file: {base_output}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <base_output_json> <n_gpus>")
        sys.exit(1)

    merge_results(sys.argv[1], int(sys.argv[2]))
