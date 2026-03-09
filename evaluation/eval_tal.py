"""Temporal Action Localization Evaluation Script for Multiple Datasets."""

import json
import sys
from collections import defaultdict
import numpy as np

# Import evaluation functions from the old script
sys.path.append('/root/code/Qwen2.5-VL/my_eval_old')
import eval_tag as old_eval_tag


def detect_dataset_from_video_id(video_id):
    """Detect dataset from video ID patterns."""
    video_id = str(video_id).lower()
    
    # AVOS dataset - YouTube video IDs
    if len(video_id) == 11 and any(c.isalpha() for c in video_id):
        return "AVOS"
    
    # CoPESD dataset - numerical IDs with parts
    if "_part" in video_id and video_id.replace("_part", "").split("_")[0].isdigit():
        return "CoPESD"
    
    # CholecT50 dataset
    if "video" in video_id.lower() and any(c.isdigit() for c in video_id):
        return "CholecT50"
    
    # NurViD dataset - specific patterns
    if any(keyword in video_id for keyword in ["nur", "nursing", "medical"]):
        return "NurViD"
    
    return "Unknown"


def detect_dataset_from_question(question):
    """Detect dataset from question text patterns."""
    question_lower = question.lower()
    
    if "avos" in question_lower:
        return "AVOS"
    elif "copesd" in question_lower:
        return "CoPESD"
    elif "cholect50" in question_lower or "cholec" in question_lower:
        return "CholecT50"
    elif "nurvid" in question_lower or "nursing" in question_lower:
        return "NurViD"
    
    # Check for dataset-specific action patterns
    if any(action in question_lower for action in ["cutting", "tying", "suturing"]):
        return "AVOS"
    elif "forceps" in question_lower and "knife" in question_lower:
        return "CoPESD"
    
    return "Unknown"


def group_records_by_dataset(data):
    """Group TAL records by dataset."""
    dataset_records = defaultdict(list)
    
    for idx, record in data.items():
        if record.get("qa_type") != "tal":
            continue
            
        # Get dataset from data_source field first, fallback to detection if needed
        dataset = record.get("data_source", "Unknown")
        if dataset == "Unknown" or not dataset:
            dataset = detect_dataset_from_video_id(record["metadata"]["video_id"])
            if dataset == "Unknown":
                dataset = detect_dataset_from_question(record["question"])
        
        # Extract required data
        question = record['question'].strip()
        raw_answer = record['answer'].strip()
        answer_segments = old_eval_tag.extract_segments_from_text(raw_answer)
        
        # Handle different struc_info formats
        if isinstance(record['struc_info'], list):
            # New format - list of action dictionaries
            spans = []
            for action_info in record['struc_info']:
                spans.extend(action_info.get('spans', []))

            # If struc_info is empty list, parse from 'gnd' field
            if not spans and 'gnd' in record:
                raw_gnd = record['gnd'].strip()
                spans = old_eval_tag.extract_segments_from_text(raw_gnd)
        else:
            # Old format - direct spans
            spans = record['struc_info'].get('spans', [])
        
        fps = float(record['metadata']['fps'])
        
        # Convert from seconds to frames
        for segment in answer_segments:
            segment['start'] = float(segment['start'] * fps)
            segment['end'] = float(segment['end'] * fps)
        for span in spans:
            span['start'] = float(span['start'] * fps)
            span['end'] = float(span['end'] * fps)
        
        record_data = {
            "question": question,
            "prediction": answer_segments,
            "ground_truth": spans,
            "fps": fps,
            "video_id": record["metadata"]["video_id"]
        }
        
        dataset_records[dataset].append(record_data)
    
    return dataset_records


def evaluate_dataset_tal(dataset_name, dataset_records, tiou_thresholds=[0.3, 0.5, 0.7]):
    """Evaluate temporal action localization for a specific dataset."""
    print(f"\n=== Temporal Action Localization Evaluation for {dataset_name} ===")
    print(f"Number of records: {len(dataset_records)}")
    
    if not dataset_records:
        print("No records found for this dataset.")
        return {}
    
    # Group by FPS for detailed analysis
    fps_grouped = defaultdict(list)
    for record in dataset_records:
        fps_grouped[record["fps"]].append(record)
    
    # Evaluate per FPS
    all_results = {}
    for fps_value in sorted(fps_grouped.keys()):
        fps_records = fps_grouped[fps_value]
        print(f"\n--- FPS: {fps_value} ({len(fps_records)} records) ---")
        
        # Evaluate at different IoU thresholds
        for tiou_thresh in tiou_thresholds:
            results = old_eval_tag.evaluate_tal_record(fps_records, tiou_thresh=tiou_thresh)
            key = f"IoU_{tiou_thresh:.1f}"
            if key not in all_results:
                all_results[key] = {}
            all_results[key][fps_value] = results
            
            old_eval_tag.pretty_print_summary(results, f"TAL @IoU={tiou_thresh} (fps={fps_value})")
    
    # Overall evaluation for this dataset
    if len(fps_grouped) > 1:
        print(f"\n--- Overall {dataset_name} (all FPS combined) ---")
        
        overall_results = {}
        for tiou_thresh in tiou_thresholds:
            results = old_eval_tag.evaluate_tal_record(dataset_records, tiou_thresh=tiou_thresh)
            overall_results[f"IoU_{tiou_thresh:.1f}"] = results
            old_eval_tag.pretty_print_summary(results, f"TAL @IoU={tiou_thresh} (all fps)")
        
        return overall_results
    
    # Return results for single FPS
    single_fps_results = {}
    for key, fps_dict in all_results.items():
        if len(fps_dict) == 1:
            single_fps_results[key] = list(fps_dict.values())[0]
    
    return single_fps_results


def main():
    """Main evaluation function."""
    if len(sys.argv) > 1:
        output_file = sys.argv[1]
    else:
        output_file = "/root/code/Qwen2.5-VL/inference_results/qa_instances_08_15_type_grouped_results_baseline.json"
    
    print(f"Loading results from: {output_file}")
    
    with open(output_file, "r") as f:
        infer_output = json.load(f)
    
    # Group records by dataset
    dataset_records = group_records_by_dataset(infer_output)
    
    print(f"\nFound datasets: {list(dataset_records.keys())}")
    for dataset, records in dataset_records.items():
        print(f"  {dataset}: {len(records)} TAL records")
    
    # Evaluate each dataset
    all_results = {}
    for dataset_name, records in dataset_records.items():
        if records:  # Only evaluate if we have records
            results = evaluate_dataset_tal(dataset_name, records)
            all_results[dataset_name] = results
    
    # Print summary
    print(f"\n{'='*60}")
    print("TEMPORAL ACTION LOCALIZATION EVALUATION SUMMARY")
    print(f"{'='*60}")
    
    for dataset_name, results in all_results.items():
        if results:
            print(f"\n{dataset_name}:")
            for iou_key, metrics in results.items():
                if isinstance(metrics, dict):
                    print(f"  {iou_key}:")
                    for metric_name, value in metrics.items():
                        print(f"    {metric_name}: {value:.4f}")
    
    return all_results


if __name__ == "__main__":
    main()
