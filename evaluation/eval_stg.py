"""Spatial-Temporal Grounding Evaluation Script for Multiple Datasets."""

import json
import sys
from collections import defaultdict
import numpy as np

# Import evaluation functions from the old script  
sys.path.insert(0, '/root/code/Qwen2.5-VL')
sys.path.insert(0, '/root/code/Qwen2.5-VL/my_eval_old')

# Set PYTHONPATH to help with imports
import os
os.environ['PYTHONPATH'] = '/root/code/Qwen2.5-VL:' + os.environ.get('PYTHONPATH', '')

# Use importlib to avoid naming conflicts
import importlib.util
spec = importlib.util.spec_from_file_location("old_eval_stg", "/root/code/Qwen2.5-VL/my_eval_old/eval_stg.py")
old_eval_stg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old_eval_stg)


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


def post_process_pred_flexible(prediction_text):
    """
    Flexible post-processing for STG predictions that handles malformed brackets.
    
    Handles cases like:
    - 1365, 55, 1630, 357) -> [1365, 55, 1630, 357]
    - [1376, 0, 1919, 305 -> [1376, 0, 1919, 305]
    - [1365, 55, 1630, 357) -> [1365, 55, 1630, 357]
    """
    import re
    
    try:
        # First try the original post-processing
        return old_eval_stg.post_process_pred(prediction_text)
    except Exception:
        # If that fails, apply flexible parsing
        print(f"[Flexible parsing] Processing outlier: {prediction_text}")
        
        # Fix common bracket issues
        fixed_text = prediction_text
        
        # Replace mismatched closing parenthesis with closing bracket
        fixed_text = re.sub(r'(\d+)\s*\)', r'\1]', fixed_text)
        
        # Ensure opening bracket exists if we have numbers but no opening bracket
        if re.search(r'\d+\s*,.*\d+', fixed_text) and not fixed_text.strip().startswith('['):
            # Find the first number and add opening bracket
            fixed_text = re.sub(r'^([^0-9]*?)(\d+)', r'\1[\2', fixed_text)
        
        # Ensure closing bracket exists if we have numbers but no closing bracket
        if re.search(r'\d+\s*,.*\d+', fixed_text) and not fixed_text.strip().endswith(']'):
            # Add closing bracket at the end after the last number
            fixed_text = re.sub(r'(\d+)([^0-9]*)$', r'\1]\2', fixed_text)
        
        # Clean up multiple brackets
        fixed_text = re.sub(r'\]\]', ']', fixed_text)
        fixed_text = re.sub(r'\[\[', '[', fixed_text)
        
        print(f"[Flexible parsing] Fixed to: {fixed_text}")
        
        try:
            # Try processing the fixed text
            return old_eval_stg.post_process_pred(fixed_text)
        except Exception as e:
            print(f"[Flexible parsing] Still failed after fixing: {e}")
            # Return empty result as fallback
            return {}


def group_records_by_dataset(data):
    """Group STG records by dataset."""
    dataset_records = defaultdict(list)
    
    for idx, record in data.items():
        if record.get("qa_type") != "stg":
            continue
            
        # Detect dataset using common utility
        from dataset_utils import get_dataset_name
        dataset = get_dataset_name(record)
        
        # Extract required data
        question = record['question'].strip()
        processed_pred = post_process_pred_flexible(record['answer'].strip())
        
        # Handle different struc_info formats
        struc_info = record['struc_info']
        if isinstance(struc_info, list) and len(struc_info) > 0:
            # Take the first item if it's a list
            struc_item = struc_info[0]
            if isinstance(struc_item, dict) and 'bbox_dict' in struc_item:
                gt_dict = struc_item['bbox_dict']
            else:
                gt_dict = struc_item
        elif isinstance(struc_info, list) and len(struc_info) == 0:
            # Empty struc_info - parse from 'gnd' field
            if 'gnd' in record:
                raw_gnd = record['gnd'].strip()
                gt_dict = post_process_pred_flexible(raw_gnd)
            else:
                gt_dict = {}
        elif isinstance(struc_info, dict):
            if 'bbox_dict' in struc_info:
                gt_dict = struc_info['bbox_dict']
            else:
                gt_dict = struc_info
        else:
            gt_dict = struc_info
        
        fps = float(record['metadata']['fps']) if 'metadata' in record and 'fps' in record['metadata'] else 1.0
        
        record_data = {
            "question": question,
            "processed_pred": processed_pred,
            "gt_dict": gt_dict,
            "fps": fps,
            "video_id": record["metadata"]["video_id"]
        }
        
        dataset_records[dataset].append(record_data)
    
    return dataset_records


def evaluate_dataset_stg(dataset_name, dataset_records):
    """Evaluate spatial-temporal grounding for a specific dataset."""
    print(f"\n=== Spatial-Temporal Grounding Evaluation for {dataset_name} ===")
    print(f"Number of records: {len(dataset_records)}")
    
    if not dataset_records:
        print("No records found for this dataset.")
        return {}
    
    # Group by FPS for detailed analysis
    fps_grouped = defaultdict(list)
    for record in dataset_records:
        fps_grouped[record["fps"]].append(record)
    
    # Evaluate per FPS
    all_ious = []
    fps_results = {}
    
    for fps_value in sorted(fps_grouped.keys()):
        fps_records = fps_grouped[fps_value]
        print(f"\n--- FPS: {fps_value} ({len(fps_records)} records) ---")
        
        fps_ious = []
        valid_records = 0
        
        for record in fps_records:
            processed_pred = record["processed_pred"]
            gt_dict = record["gt_dict"]
            
            # Convert prediction list to dict using GT keys if needed
            if isinstance(processed_pred, list):
                key_list = list(gt_dict.keys())
                processed_pred = {key: box for key, box in zip(key_list[:len(processed_pred)], processed_pred)}
            
            pred_boxes = []
            gt_boxes = []
            
            # Process boxes
            for i, key in enumerate(gt_dict.keys()):
                gt_boxes.append(gt_dict[key])
                key_str = f"{float(key):.1f}"
                pred_box = processed_pred.get(key_str, [0, 0, 0, 0])
                if pred_box == [0, 0, 0, 0] and i > 0:
                    pred_box = pred_boxes[i - 1]  # Use previous box if current is invalid
                pred_boxes.append(pred_box)
            
            # Validate boxes
            valid_pred_boxes = []
            valid_gt_boxes = []
            for pred_box, gt_box in zip(pred_boxes, gt_boxes):
                if old_eval_stg.is_valid_box(pred_box) and old_eval_stg.is_valid_box(gt_box):
                    valid_pred_boxes.append(pred_box)
                    valid_gt_boxes.append(gt_box)
            
            if valid_pred_boxes and valid_gt_boxes:
                pred_boxes_array = np.array(valid_pred_boxes)
                gt_boxes_array = np.array(valid_gt_boxes)
                iou = old_eval_stg.compute_iou_batch(pred_boxes_array, gt_boxes_array)
                
                if len(iou) > 0:
                    mean_iou = iou.mean()
                    fps_ious.append(mean_iou)
                    all_ious.append(mean_iou)
                    valid_records += 1
                else:
                    print(f"Empty IoU for record with video_id {record['video_id']}")
            else:
                print(f"Invalid boxes for record with video_id {record['video_id']}")
        
        # Compute FPS-specific metrics
        if fps_ious:
            fps_mean_iou = sum(fps_ious) / len(fps_ious)
            print(f"Mean IoU: {fps_mean_iou:.4f} (from {valid_records} valid records)")
            fps_results[fps_value] = {
                "mean_iou": fps_mean_iou,
                "valid_records": valid_records,
                "total_records": len(fps_records)
            }
        else:
            print("No valid IoU scores computed")
            fps_results[fps_value] = {
                "mean_iou": 0.0,
                "valid_records": 0,
                "total_records": len(fps_records)
            }
    
    # Overall evaluation for this dataset
    overall_results = fps_results.copy()
    if len(fps_grouped) > 1 and all_ious:
        overall_mean_iou = sum(all_ious) / len(all_ious)
        print(f"\n--- Overall {dataset_name} (all FPS combined) ---")
        print(f"Mean IoU: {overall_mean_iou:.4f} (from {len(all_ious)} valid records)")
        overall_results["overall"] = {
            "mean_iou": overall_mean_iou,
            "valid_records": len(all_ious),
            "total_records": len(dataset_records)
        }
    
    return overall_results


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
        print(f"  {dataset}: {len(records)} STG records")
    
    # Evaluate each dataset
    all_results = {}
    for dataset_name, records in dataset_records.items():
        if records:  # Only evaluate if we have records
            results = evaluate_dataset_stg(dataset_name, records)
            all_results[dataset_name] = results
    
    # Print summary
    print(f"\n{'='*60}")
    print("SPATIAL-TEMPORAL GROUNDING EVALUATION SUMMARY")
    print(f"{'='*60}")
    
    for dataset_name, results in all_results.items():
        if results:
            print(f"\n{dataset_name}:")
            
            # Print per-FPS results
            for fps_key, metrics in results.items():
                if fps_key == "overall":
                    continue
                print(f"  FPS {fps_key}: IoU = {metrics['mean_iou']:.4f} "
                      f"({metrics['valid_records']}/{metrics['total_records']} valid)")
            
            # Print overall result if available
            if "overall" in results:
                overall = results["overall"]
                print(f"  Overall: IoU = {overall['mean_iou']:.4f} "
                      f"({overall['valid_records']}/{overall['total_records']} valid)")
    
    return all_results


if __name__ == "__main__":
    main()
