"""Dense Video Captioning Evaluation Script for Multiple Datasets."""

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
spec = importlib.util.spec_from_file_location("old_eval_dvc", "/root/code/Qwen2.5-VL/my_eval_old/eval_dvc.py")
old_eval_dvc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old_eval_dvc)


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
    """Group DVC records by dataset."""
    dataset_records = defaultdict(list)
    
    for idx, record in data.items():
        qa_type = record.get("qa_type", "")
        if not any(dvc_type in qa_type for dvc_type in ["dc", "dense_captioning"]):
            continue
            
        # Get dataset from data_source field first, fallback to detection if needed
        dataset = record.get("data_source", "Unknown")
        if dataset == "Unknown" or not dataset:
            dataset = detect_dataset_from_video_id(record["metadata"]["video_id"])
            if dataset == "Unknown":
                dataset = detect_dataset_from_question(record["question"])
        
        # Extract required data
        question = record['question']
        raw_answer = record['answer']
        
        # Handle different struc_info formats
        if isinstance(record['struc_info'], list) and len(record['struc_info']) > 0:
            if isinstance(record['struc_info'][0], list):
                # Format: [[{segments...}]]
                gnd = record['struc_info'][0]
            elif isinstance(record['struc_info'][0], dict) and 'dc_segments' in record['struc_info'][0]:
                # NurViD format: [{'dc_segments': [...]}]
                gnd = record['struc_info'][0]['dc_segments']
            else:
                # Format: [{segments...}]
                gnd = record['struc_info']
        else:
            gnd = record['struc_info']
            
        fps = float(record['metadata']['fps'])
        
        # Process prediction
        processed_answer = old_eval_dvc.process_raw_output(raw_answer)
        overlaps = old_eval_dvc.check_for_overlaps(processed_answer)
        if overlaps:
            processed_answer = old_eval_dvc.flatten_overlapping_segments(processed_answer, caption_strategy="longest")
        
        # Convert to frame-based coordinates
        if isinstance(gnd, list):
            for g in gnd:
                if isinstance(g, dict) and 'start' in g and 'end' in g:
                    g['start'] = int(g['start'] * fps)
                    g['end'] = int(g['end'] * fps)
        
        if isinstance(processed_answer, list):
            for p in processed_answer:
                if isinstance(p, dict) and 'start' in p and 'end' in p:
                    p['start'] = int(p['start'] * fps)
                    p['end'] = int(p['end'] * fps)
        
        record_data = {
            "question": question,
            "gnd": gnd,
            "pred": processed_answer,
            "fps": fps,
            "video_id": record["metadata"]["video_id"]
        }
        
        dataset_records[dataset].append(record_data)
    
    return dataset_records


def prepare_eval_arrays(dc_records):
    """Prepare evaluation arrays for dense captioning evaluation."""
    predicted_segments = []
    gt_segments = []
    predicted_captions = []
    gt_captions = []
    splits = []
    keys = []
    
    for idx, item in enumerate(dc_records):
        keys.append(str(idx))
        
        gt_seg = []
        gt_cap = []
        gnd = item["gnd"]
        if isinstance(gnd, list):
            for g in gnd:
                if isinstance(g, dict) and 'start' in g and 'end' in g and 'caption' in g:
                    gt_seg.append([g["start"], g["end"]])
                    gt_cap.append(g["caption"])
        
        pred_seg = []
        pred_cap = []
        pred = item["pred"]
        if isinstance(pred, list):
            for p in pred:
                if isinstance(p, dict) and 'start' in p and 'end' in p and 'caption' in p:
                    pred_seg.append([p["start"], p["end"]])
                    pred_cap.append(p["caption"])
        
        if gt_seg:  # Only add if we have valid segments
            gt_segments.append(np.array(gt_seg))
            gt_captions.append(gt_cap)
            splits.append(np.ones(len(gt_seg), dtype=int))
            predicted_segments.append(np.array(pred_seg))
            predicted_captions.append(pred_cap)
    
    return predicted_segments, gt_segments, predicted_captions, gt_captions, splits, keys


def evaluate_dataset_dvc(dataset_name, dataset_records, iou_thresholds=(0.3, 0.5, 0.7), skip_caption=False):
    """Evaluate dense video captioning for a specific dataset."""
    print(f"\n=== Dense Captioning Evaluation for {dataset_name} ===")
    print(f"Number of records: {len(dataset_records)}")
    
    if not dataset_records:
        print("No records found for this dataset.")
        return {}
    
    # Group by FPS for detailed analysis
    fps_grouped = defaultdict(list)
    for record in dataset_records:
        fps_grouped[record["fps"]].append(record)
    
    # Evaluate per FPS
    all_metrics = []
    for fps_value in sorted(fps_grouped.keys()):
        fps_records = fps_grouped[fps_value]
        print(f"\n--- FPS: {fps_value} ({len(fps_records)} records) ---")
        
        predicted_segments, gt_segments, predicted_captions, gt_captions, splits, keys = prepare_eval_arrays(fps_records)
        
        try:
            metrics = old_eval_dvc.evaluate_dense_captions(
                predicted_segments,
                gt_segments,
                predicted_captions,
                gt_captions,
                splits,
                keys,
                iou_thresholds,
                tmponly=skip_caption
            )
        except (KeyError, IndexError) as e:
            print(f"Warning: Evaluation failed for FPS {fps_value} due to key mapping issue: {e}")
            # Create empty metrics structure
            metrics = {
                'CIDER': {'tIoU=0.3': {'Precision': 0.0, 'Recall': 0.0, 'F1': 0.0}, 
                         'tIoU=0.5': {'Precision': 0.0, 'Recall': 0.0, 'F1': 0.0},
                         'tIoU=0.7': {'Precision': 0.0, 'Recall': 0.0, 'F1': 0.0}},
                'METEOR': {'tIoU=0.3': {'Precision': 0.0, 'Recall': 0.0, 'F1': 0.0},
                          'tIoU=0.5': {'Precision': 0.0, 'Recall': 0.0, 'F1': 0.0},
                          'tIoU=0.7': {'Precision': 0.0, 'Recall': 0.0, 'F1': 0.0}},
                'SODA': {'Average across tIoUs': 0.0}
            }
        
        try:
            old_eval_dvc.print_dense_caption_metrics_summary(metrics)
        except Exception as e:
            print(f"Warning: Could not print metrics summary: {e}")
            print("Metrics structure:", metrics)
        all_metrics.append(metrics)
    
    # Overall evaluation for this dataset
    if len(fps_grouped) > 1:
        print(f"\n--- Overall {dataset_name} (all FPS combined) ---")
        predicted_segments, gt_segments, predicted_captions, gt_captions, splits, keys = prepare_eval_arrays(dataset_records)
        
        try:
            overall_metrics = old_eval_dvc.evaluate_dense_captions(
                predicted_segments,
                gt_segments,
                predicted_captions,
                gt_captions,
                splits,
                keys,
                iou_thresholds,
                tmponly=skip_caption
            )
        except (KeyError, IndexError) as e:
            print(f"Warning: Overall evaluation failed due to key mapping issue: {e}")
            # Create empty metrics structure
            overall_metrics = {
                'CIDER': {'tIoU=0.3': {'Precision': 0.0, 'Recall': 0.0, 'F1': 0.0}, 
                         'tIoU=0.5': {'Precision': 0.0, 'Recall': 0.0, 'F1': 0.0},
                         'tIoU=0.7': {'Precision': 0.0, 'Recall': 0.0, 'F1': 0.0}},
                'METEOR': {'tIoU=0.3': {'Precision': 0.0, 'Recall': 0.0, 'F1': 0.0},
                          'tIoU=0.5': {'Precision': 0.0, 'Recall': 0.0, 'F1': 0.0},
                          'tIoU=0.7': {'Precision': 0.0, 'Recall': 0.0, 'F1': 0.0}},
                'SODA': {'Average across tIoUs': 0.0}
            }
        
        try:
            old_eval_dvc.print_dense_caption_metrics_summary(overall_metrics)
        except Exception as e:
            print(f"Warning: Could not print overall metrics summary: {e}")
            print("Overall metrics structure:", overall_metrics)
        return overall_metrics
    
    return all_metrics[0] if all_metrics else {}


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
        print(f"  {dataset}: {len(records)} DVC records")
    
    # Evaluate each dataset
    all_results = {}
    for dataset_name, records in dataset_records.items():
        if records:  # Only evaluate if we have records
            results = evaluate_dataset_dvc(dataset_name, records)
            all_results[dataset_name] = results
    
    # Print summary
    print(f"\n{'='*60}")
    print("DENSE VIDEO CAPTIONING EVALUATION SUMMARY")
    print(f"{'='*60}")
    
    for dataset_name, results in all_results.items():
        if results:
            print(f"\n{dataset_name}:")
            key_metrics = ['CIDER', 'METEOR', 'Precision_Mean', 'Recall_Mean', 'F1_Score', 'SODA_c_1']
            for metric in key_metrics:
                if metric in results:
                    if isinstance(results[metric], list) and results[metric]:
                        avg_val = np.mean(results[metric])
                        print(f"  {metric}: {avg_val:.4f}")
                    elif isinstance(results[metric], (int, float)):
                        print(f"  {metric}: {results[metric]:.4f}")
    
    return all_results


if __name__ == "__main__":
    main()
