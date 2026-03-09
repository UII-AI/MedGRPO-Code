"""Region Caption and Video Summary Evaluation Script for Multiple Datasets."""

import json
import sys
from collections import defaultdict

# Import evaluation functions directly
sys.path.append('/root/code/Qwen2.5-VL')
from captioning_metrics.cider import Cider
from captioning_metrics.meteor import Meteor
from captioning_metrics.ptbtokenizer import PTBTokenizer

# Import dataset utilities
from dataset_utils import get_dataset_name


def detect_dataset_from_video_id(video_id):
    """Detect dataset from video ID patterns."""
    video_id = str(video_id).lower()
    
    # AVOS dataset - YouTube video IDs
    if len(video_id) == 11 and any(c.isalpha() for c in video_id):
        return "AVOS"
    
    # CoPESD dataset - numerical IDs with parts
    if "_part" in video_id and video_id.replace("_part", "").split("_")[0].isdigit():
        return "CoPESD"
    
    # CholecTrack20 dataset - VID + number pattern
    if video_id.startswith("vid") and any(c.isdigit() for c in video_id):
        return "CholecTrack20"
    
    # Cholec80-CVS dataset - video + number pattern
    if video_id.startswith("video") and any(c.isdigit() for c in video_id):
        return "Cholec80-CVS"
        
    # JIGSAWS dataset - knot tying patterns
    if "knot_tying" in video_id or "needle_passing" in video_id or "suturing" in video_id:
        return "JIGSAWS"
    
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
    elif "cholect50" in question_lower or "cholec-t50" in question_lower:
        return "CholecT50"
    elif "cholectrack20" in question_lower or "cholec-track20" in question_lower:
        return "CholecTrack20"
    elif "cholec80-cvs" in question_lower or "critical view of safety" in question_lower:
        return "Cholec80-CVS"
    elif "jigsaws" in question_lower or "robotic bench-top" in question_lower:
        return "JIGSAWS"
    elif "nurvid" in question_lower or "nursing" in question_lower:
        return "NurViD"
    elif "laparoscopic cholecystectomy" in question_lower:
        return "CholecTrack20"
    
    # Check for dataset-specific patterns
    if any(action in question_lower for action in ["cutting", "tying", "suturing"]) and "open surgery" in question_lower:
        return "AVOS"
    elif "forceps" in question_lower and "knife" in question_lower:
        return "CoPESD"
    
    return "Unknown"


def group_records_by_dataset(data, qa_types):
    """Group RC/VS records by dataset."""
    dataset_records = defaultdict(lambda: defaultdict(list))
    
    for idx, record in data.items():
        qa_type = record.get("qa_type", "")
        if not any(target_type in qa_type for target_type in ["region_caption", "video_summary"]):
            continue
            
        # Detect dataset
        dataset = get_dataset_name(record)
        
        # Determine which type this is
        if "region_caption" in qa_type:
            task_type = "region_caption"
        elif "video_summary" in qa_type:
            task_type = "video_summary"
        else:
            task_type = qa_type
        
        record_data = {
            "question": record["question"],
            "answer": record["answer"],
            "gnd": record["gnd"],
            "video_id": record["metadata"]["video_id"]
        }
        
        dataset_records[dataset][task_type].append(record_data)
    
    return dataset_records


def evaluate_caption_task(task_name, records):
    """Evaluate a captioning task (RC or VS) using CIDER and METEOR."""
    if not records:
        print(f"No {task_name} records found.")
        return {}
    
    print(f"\n--- {task_name} Evaluation ({len(records)} records) ---")
    
    # Extract predictions and ground truths
    preds = [item['answer'] for item in records]
    gnds = [item['gnd'] for item in records]
    
    # Prepare dictionaries for evaluation
    gt_dict = {str(i): [{'caption': gt}] for i, gt in enumerate(gnds)}
    pred_dict = {str(i): [{'caption': pred}] for i, pred in enumerate(preds)}
    
    # Tokenize
    tokenizer = PTBTokenizer()
    gt_tokenized = tokenizer.tokenize(gt_dict)
    pred_tokenized = tokenizer.tokenize(pred_dict)
    
    # Initialize scorers
    cider_scorer = Cider()
    meteor_scorer = Meteor()
    
    # Compute scores
    cider_score, _ = cider_scorer.compute_score(gt_tokenized, pred_tokenized)
    meteor_score, _ = meteor_scorer.compute_score(gt_tokenized, pred_tokenized)
    
    # Output results
    print(f"CIDER:  {cider_score:.4f}")
    print(f"METEOR: {meteor_score:.4f}")
    
    # Clean up METEOR subprocess
    with meteor_scorer.lock:
        meteor_scorer.meteor_p.stdin.close()
        meteor_scorer.meteor_p.stdout.close()
        meteor_scorer.meteor_p.kill()
        meteor_scorer.meteor_p.wait()
    
    del cider_scorer
    del meteor_scorer
    del tokenizer
    
    return {
        "CIDER": cider_score,
        "METEOR": meteor_score,
        "num_records": len(records)
    }


def evaluate_dataset_rc_vs(dataset_name, dataset_records):
    """Evaluate region caption and video summary for a specific dataset."""
    print(f"\n=== Region Caption & Video Summary Evaluation for {dataset_name} ===")
    
    results = {}
    
    # Evaluate Region Caption if available
    if "region_caption" in dataset_records:
        rc_records = dataset_records["region_caption"]
        results["region_caption"] = evaluate_caption_task("Region Caption", rc_records)
    
    # Evaluate Video Summary if available  
    if "video_summary" in dataset_records:
        vs_records = dataset_records["video_summary"]
        results["video_summary"] = evaluate_caption_task("Video Summary", vs_records)
    
    return results


def main():
    """Main evaluation function."""
    if len(sys.argv) > 1:
        output_file = sys.argv[1]
    else:
        output_file = "/root/code/Qwen2.5-VL/inference_results/qa_instances_08_15_type_grouped_results_baseline.json"
    
    print(f"Loading results from: {output_file}")
    
    with open(output_file, "r") as f:
        infer_output = json.load(f)
    
    # Group records by dataset for RC and VS tasks
    qa_types = ["region_caption", "video_summary"]
    dataset_records = group_records_by_dataset(infer_output, qa_types)
    
    # Print what we found
    print(f"\nFound datasets:")
    total_rc = 0
    total_vs = 0
    for dataset, records in dataset_records.items():
        rc_count = len(records.get("region_caption", []))
        vs_count = len(records.get("video_summary", []))
        total_rc += rc_count
        total_vs += vs_count
        print(f"  {dataset}: {rc_count} RC, {vs_count} VS records")
    
    print(f"\nTotal: {total_rc} Region Caption, {total_vs} Video Summary records")
    
    if total_rc == 0 and total_vs == 0:
        print("No Region Caption or Video Summary records found!")
        return
    
    # Evaluate each dataset
    all_results = {}
    for dataset_name, records in dataset_records.items():
        if records:  # Only evaluate if we have records
            results = evaluate_dataset_rc_vs(dataset_name, records)
            all_results[dataset_name] = results
    
    # Print summary
    print(f"\n{'='*60}")
    print("REGION CAPTION & VIDEO SUMMARY EVALUATION SUMMARY")
    print(f"{'='*60}")
    
    for dataset_name, results in all_results.items():
        if results:
            print(f"\n{dataset_name}:")
            
            if "region_caption" in results:
                rc = results["region_caption"]
                print(f"  Region Caption ({rc['num_records']} records):")
                print(f"    CIDER: {rc['CIDER']:.4f}")
                print(f"    METEOR: {rc['METEOR']:.4f}")
            
            if "video_summary" in results:
                vs = results["video_summary"]
                print(f"  Video Summary ({vs['num_records']} records):")
                print(f"    CIDER: {vs['CIDER']:.4f}")
                print(f"    METEOR: {vs['METEOR']:.4f}")


if __name__ == "__main__":
    main()
