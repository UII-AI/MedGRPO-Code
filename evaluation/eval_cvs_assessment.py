"""CVS (Clinical Video Summary) Assessment Evaluation Script for Multiple Datasets."""

import json
import sys
from collections import defaultdict
import numpy as np


def detect_dataset_from_video_id(video_id):
    """Detect dataset from video ID patterns."""
    video_id = str(video_id).lower()
    
    # Cholec80_CVS dataset - patterns like "video05", "video10", etc.
    if video_id.startswith("video") and video_id[5:].isdigit():
        return "Cholec80_CVS"
    
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
    
    # Cholec80_CVS dataset - look for CVS-specific terms
    if any(pattern in question_lower for pattern in ["cholec80-cvs", "strasberg", "critical view", "cvs", "cystic plate", "hepatocystic triangle"]):
        return "Cholec80_CVS"
    
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


def parse_cvs_scores(cvs_text):
    """Parse CVS assessment text into component scores from format like 'Two structures: 0, Cystic plate: 0, Hepatocystic triangle: 0'"""
    import re
    
    # Split by commas first, then parse each part
    parts = cvs_text.split(',')
    components = {}
    
    for part in parts:
        part = part.strip().lower()
        
        # Map text patterns to standard component names
        if 'two structures' in part:
            match = re.search(r'two structures?:\s*(\d+)', part)
            if match:
                components['two_structures'] = int(match.group(1))
        elif 'cystic plate' in part:
            match = re.search(r'cystic plate:\s*(\d+)', part)
            if match:
                components['cystic_plate'] = int(match.group(1))
        elif 'hepatocystic triangle' in part:
            match = re.search(r'hepatocystic triangle:\s*(\d+)', part)
            if match:
                components['hepatocystic_triangle'] = int(match.group(1))
    
    return components


def calculate_cvs_total_score(components):
    """Calculate total CVS score from components."""
    if not components:
        return None
    
    # CVS scoring: each component can be 0, 1, or 2
    # Total ranges from 0 to 6
    total = sum(components.values())
    return total


def normalize_cvs_rating(rating_text):
    """Normalize CVS rating text to standard format."""
    rating_text = rating_text.strip()
    
    # First try to parse as CVS component scores
    components = parse_cvs_scores(rating_text)
    if components:
        total_score = calculate_cvs_total_score(components)
        if total_score is not None:
            # Convert total score to rating category
            if total_score <= 1:
                return "poor"
            elif total_score <= 3:
                return "fair"
            elif total_score <= 5:
                return "good"
            else:
                return "excellent"
    
    # Fallback to simple text matching
    rating_text_lower = rating_text.lower()
    rating_mappings = {
        "poor": "poor",
        "bad": "poor",
        "low": "poor", 
        "inadequate": "poor",
        "fair": "fair",
        "average": "fair",
        "moderate": "fair",
        "good": "good",
        "satisfactory": "good",
        "adequate": "good", 
        "excellent": "excellent",
        "great": "excellent",
        "outstanding": "excellent",
        "superior": "excellent",
        "1": "poor",
        "2": "fair",
        "3": "good", 
        "4": "excellent",
        "5": "excellent"
    }
    
    for key, value in rating_mappings.items():
        if key in rating_text_lower:
            return value
    
    return rating_text


def calculate_balanced_accuracy(per_class_correct, per_class_total):
    """Calculate balanced accuracy across classes."""
    if not per_class_total:
        return 0.0
    
    # Calculate recall for each class
    recalls = []
    for class_name in per_class_total:
        if per_class_total[class_name] > 0:
            recall = per_class_correct[class_name] / per_class_total[class_name]
            recalls.append(recall)
    
    # Balanced accuracy is the mean of per-class recalls
    if recalls:
        return np.mean(recalls)
    else:
        return 0.0


def group_records_by_dataset(data):
    """Group CVS assessment records by dataset."""
    dataset_records = defaultdict(list)
    
    for idx, record in data.items():
        if record.get("qa_type") != "cvs_assessment":
            continue
            
        # Get dataset from data_source field if available (preferred method)
        dataset = record.get("data_source", "Unknown")
        
        # Fallback to detection methods if data_source is not available
        if dataset == "Unknown" or not dataset:
            dataset = detect_dataset_from_video_id(record["metadata"]["video_id"])
            if dataset == "Unknown":
                dataset = detect_dataset_from_question(record["question"])
        
        record_data = {
            "question": record["question"],
            "answer": record["answer"],
            "gnd": record["gnd"],
            "video_id": record["metadata"]["video_id"],
            "struc_info": record.get("struc_info", [])
        }
        
        dataset_records[dataset].append(record_data)
    
    return dataset_records


def evaluate_cvs_assessment(records):
    """Evaluate CVS assessment using accuracy metric."""
    if not records:
        return {"accuracy": 0.0, "correct": 0, "total": 0}
    
    correct = 0
    total = 0
    per_rating_correct = defaultdict(int)
    per_rating_total = defaultdict(int)
    
    # Per-component evaluation
    component_correct = defaultdict(int)
    component_total = defaultdict(int)
    component_mae = defaultdict(float)  # Mean Absolute Error for components
    
    for record in records:
        # Parse predicted component scores from answer text
        pred_components = parse_cvs_scores(record["answer"])
        
        # Get ground truth component scores from struc_info if available
        gnd_components = None
        if record.get("struc_info") and len(record["struc_info"]) > 0:
            gnd_components = record["struc_info"][0].get("cvs_scores", {})
            # Remove non-component fields
            gnd_components = {k: v for k, v in gnd_components.items() 
                            if k in ['two_structures', 'cystic_plate', 'hepatocystic_triangle']}
        
        # Fallback to parsing ground truth text
        if not gnd_components:
            gnd_components = parse_cvs_scores(record["gnd"])
        
        # Evaluate each component
        for component_name in gnd_components:
            if component_name in pred_components:
                gnd_score = gnd_components[component_name]
                pred_score = pred_components[component_name]
                
                component_total[component_name] += 1
                
                # Exact match accuracy
                if pred_score == gnd_score:
                    component_correct[component_name] += 1
                
                # Mean Absolute Error
                component_mae[component_name] += abs(pred_score - gnd_score)
        
        # Overall evaluation (using total scores)
        pred_total = sum(pred_components.values()) if pred_components else 0
        gnd_total = sum(gnd_components.values()) if gnd_components else 0
        
        # Convert total scores to ratings for overall accuracy
        pred_rating = "poor" if pred_total <= 1 else "fair" if pred_total <= 3 else "good" if pred_total <= 5 else "excellent"
        gnd_rating = "poor" if gnd_total <= 1 else "fair" if gnd_total <= 3 else "good" if gnd_total <= 5 else "excellent"
        
        per_rating_total[gnd_rating] += 1
        total += 1
        
        if pred_rating == gnd_rating:
            correct += 1
            per_rating_correct[gnd_rating] += 1
    
    accuracy = correct / total if total > 0 else 0.0
    
    # Calculate per-rating accuracies
    per_rating_accuracies = {}
    for rating in per_rating_total:
        rating_correct = per_rating_correct[rating]
        rating_total = per_rating_total[rating]
        rating_accuracy = rating_correct / rating_total if rating_total > 0 else 0.0
        per_rating_accuracies[rating] = {
            "accuracy": rating_accuracy,
            "correct": rating_correct,
            "total": rating_total
        }
    
    # Calculate balanced accuracy for components only
    component_balanced_acc = calculate_balanced_accuracy(component_correct, component_total)
    
    # Calculate per-component metrics
    per_component_metrics = {}
    for component in component_total:
        component_acc = component_correct[component] / component_total[component] if component_total[component] > 0 else 0.0
        component_mae_avg = component_mae[component] / component_total[component] if component_total[component] > 0 else 0.0
        per_component_metrics[component] = {
            "accuracy": component_acc,
            "correct": component_correct[component],
            "total": component_total[component],
            "mae": component_mae_avg
        }
    
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "per_rating": per_rating_accuracies,
        "per_component": per_component_metrics,
        "component_balanced_accuracy": component_balanced_acc
    }


def evaluate_dataset_cvs_assessment(dataset_name, dataset_records):
    """Evaluate CVS assessment for a specific dataset."""
    print(f"\n=== CVS Assessment Evaluation for {dataset_name} ===")
    print(f"Number of records: {len(dataset_records)}")
    
    if not dataset_records:
        print("No records found for this dataset.")
        return {}
    
    # Evaluate the dataset
    results = evaluate_cvs_assessment(dataset_records)
    
    # Print overall results
    print(f"Overall Accuracy: {results['accuracy']:.4f} ({results['correct']}/{results['total']})")
    
    # Print per-rating results
    if "per_rating" in results and results["per_rating"]:
        print("\nPer-rating Accuracy:")
        for rating, metrics in results["per_rating"].items():
            print(f"  {rating}: {metrics['accuracy']:.4f} ({metrics['correct']}/{metrics['total']})")
    
    # Print per-component results with balanced accuracy
    if "per_component" in results and results["per_component"]:
        print(f"\nComponent Balanced Accuracy: {results.get('component_balanced_accuracy', 0.0):.4f}")
        print("\nPer-component Performance:")
        component_display_names = {
            'two_structures': 'Two structures',
            'cystic_plate': 'Cystic plate', 
            'hepatocystic_triangle': 'Hepatocystic triangle'
        }
        
        for component, metrics in results["per_component"].items():
            display_name = component_display_names.get(component, component)
            print(f"  {display_name}:")
            print(f"    Accuracy: {metrics['accuracy']:.4f} ({metrics['correct']}/{metrics['total']})")
            print(f"    Mean Absolute Error: {metrics['mae']:.3f}")
    
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
    
    # Group records by dataset
    dataset_records = group_records_by_dataset(infer_output)
    
    print(f"\nFound datasets: {list(dataset_records.keys())}")
    for dataset, records in dataset_records.items():
        print(f"  {dataset}: {len(records)} CVS assessment records")
    
    if not any(dataset_records.values()):
        print("No CVS assessment records found!")
        return
    
    # Evaluate each dataset
    all_results = {}
    for dataset_name, records in dataset_records.items():
        if records:  # Only evaluate if we have records
            results = evaluate_dataset_cvs_assessment(dataset_name, records)
            all_results[dataset_name] = results
    
    # Print summary
    print(f"\n{'='*60}")
    print("CVS ASSESSMENT EVALUATION SUMMARY")
    print(f"{'='*60}")
    
    for dataset_name, results in all_results.items():
        if results:
            print(f"\n{dataset_name}:")
            print(f"  Overall Accuracy: {results['accuracy']:.4f} ({results['correct']}/{results['total']})")


if __name__ == "__main__":
    main()
