"""Next Action Prediction Evaluation Script for Multiple Datasets."""

import json
import sys
from collections import defaultdict
import numpy as np

# Import evaluation functions and data from the old script
sys.path.insert(0, '/root/code/Qwen2.5-VL')
sys.path.insert(0, '/root/code/Qwen2.5-VL/my_eval_old')

# Set PYTHONPATH to help with imports
import os
os.environ['PYTHONPATH'] = '/root/code/Qwen2.5-VL:' + os.environ.get('PYTHONPATH', '')

# Use importlib to avoid naming conflicts
import importlib.util
spec = importlib.util.spec_from_file_location("old_eval_next_action", "/root/code/Qwen2.5-VL/my_eval_old/eval_next_action.py")
old_eval_next_action = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old_eval_next_action)

try:
    from sentence_transformers import SentenceTransformer, util
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: sentence-transformers not available. Falling back to exact matching only.")


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


def calculate_balanced_accuracy(per_class_correct, per_class_total, action_list=None):
    """Calculate balanced accuracy across classes, excluding missing actions."""
    if not per_class_total:
        return 0.0
    
    # Calculate recall for each class that appears in the test set
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
    """Group next action records by dataset."""
    dataset_records = defaultdict(list)
    
    for idx, record in data.items():
        if record.get("qa_type") != "next_action":
            continue
            
        # Detect dataset using common utility
        from dataset_utils import get_dataset_name
        dataset = get_dataset_name(record)
        
        # Extract procedure for NurViD
        procedure = None
        if dataset == "NurViD":
            # Try to extract procedure from question or metadata
            question_lower = record["question"].lower()
            for proc_name in old_eval_next_action.NURVID_PROCEDURE_ACTIONS.keys():
                if proc_name.lower() in question_lower:
                    procedure = proc_name
                    break
        
        record_data = {
            "answer": record["answer"],
            "gnd": record["gnd"],
            "question": record["question"],
            "video_id": record["metadata"]["video_id"],
            "procedure": procedure
        }
        
        dataset_records[dataset].append(record_data)
    
    return dataset_records


def evaluate_dataset_next_action(dataset_name, dataset_records):
    """Evaluate next action prediction for a specific dataset."""
    print(f"\n=== Next Action Prediction Evaluation for {dataset_name} ===")
    print(f"Number of records: {len(dataset_records)}")
    
    if not dataset_records:
        print("No records found for this dataset.")
        return {}
    
    # For NurViD, handle procedure-specific evaluation
    if dataset_name == "NurViD":
        return evaluate_nurvid_procedures(dataset_records)
    else:
        return evaluate_single_dataset(dataset_name, dataset_records)


def evaluate_nurvid_procedures(dataset_records):
    """Evaluate NurViD dataset with procedure-specific handling."""
    # Group records by procedure
    procedure_records = defaultdict(list)
    for record in dataset_records:
        procedure = record.get("procedure", "Unknown")
        procedure_records[procedure].append(record)
    
    print(f"Found {len(procedure_records)} procedures in NurViD data:")
    for proc, records in procedure_records.items():
        print(f"  {proc}: {len(records)} records")
    
    # Evaluate each procedure separately
    total_correct = 0
    total_records = 0
    procedure_results = {}
    
    for procedure, records in procedure_records.items():
        print(f"\n--- Evaluating {procedure} ---")
        
        # Get action list for this procedure
        try:
            actions = old_eval_next_action.get_action_list_for_dataset("NurViD", procedure)
            CLASS_MAP = old_eval_next_action.create_class_map_for_dataset(actions)
            
            # Load SentenceTransformer model for semantic similarity
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                semantic_class_eval_model = SentenceTransformer('all-MiniLM-L6-v2')
                class_embeddings = semantic_class_eval_model.encode(actions, convert_to_tensor=True)
            else:
                semantic_class_eval_model = None
                class_embeddings = None
            
            # Evaluate
            procedure_correct = 0
            procedure_total = 0
            per_class_correct = defaultdict(int)
            per_class_total = defaultdict(int)
            
            for record in records:
                pred_text = old_eval_next_action.normalize_action_text(record['answer'], "NurViD")
                gnd_text = old_eval_next_action.normalize_action_text(record['gnd'], "NurViD")
                
                # Skip if ground truth not in action list
                if gnd_text not in CLASS_MAP:
                    print(f"Warning: Ground truth '{gnd_text}' not found in {procedure} action list")
                    continue
                
                # Determine prediction class
                if pred_text in CLASS_MAP:
                    pred_idx = CLASS_MAP[pred_text]
                else:
                    # Use semantic similarity as fallback
                    if SENTENCE_TRANSFORMERS_AVAILABLE and semantic_class_eval_model is not None:
                        pred_emb = semantic_class_eval_model.encode(pred_text, convert_to_tensor=True)
                        sim_scores = util.cos_sim(pred_emb, class_embeddings)[0]
                        pred_idx = sim_scores.argmax().item()
                        print(f"Using semantic similarity for prediction: '{pred_text}' -> '{actions[pred_idx]}'")
                    else:
                        # No semantic similarity available, mark as incorrect
                        pred_idx = -1
                
                gnd_idx = CLASS_MAP[gnd_text]
                per_class_total[gnd_text] += 1
                
                if pred_idx == gnd_idx:
                    procedure_correct += 1
                    per_class_correct[gnd_text] += 1
                procedure_total += 1
            
            # Procedure accuracy
            if procedure_total > 0:
                procedure_accuracy = procedure_correct / procedure_total
                procedure_balanced_acc = calculate_balanced_accuracy(per_class_correct, per_class_total, actions)
                
                print(f"{procedure} accuracy: {procedure_accuracy:.4f} ({procedure_correct}/{procedure_total})")
                print(f"{procedure} balanced accuracy: {procedure_balanced_acc:.4f}")
                
                total_correct += procedure_correct
                total_records += procedure_total
                
                procedure_results[procedure] = {
                    "accuracy": procedure_accuracy,
                    "balanced_accuracy": procedure_balanced_acc,
                    "correct": procedure_correct,
                    "total": procedure_total
                }
                
                # Per-class accuracy for this procedure
                print(f"\nPer-class accuracy for {procedure}:")
                for action in actions:
                    total_cls = per_class_total[action]
                    correct_cls = per_class_correct[action]
                    if total_cls > 0:
                        acc = correct_cls / total_cls
                        print(f"  {action:40s}: {acc:.4f} ({correct_cls}/{total_cls})")
                    else:
                        print(f"  {action:40s}: N/A (0 samples)")
            else:
                print(f"No valid records for {procedure}")
                procedure_results[procedure] = {"accuracy": 0.0, "balanced_accuracy": 0.0, "correct": 0, "total": 0}
                
        except Exception as e:
            print(f"Error evaluating {procedure}: {e}")
            procedure_results[procedure] = {"accuracy": 0.0, "balanced_accuracy": 0.0, "correct": 0, "total": 0}
    
    # Overall accuracy
    overall_results = procedure_results.copy()
    if total_records > 0:
        overall_accuracy = total_correct / total_records
        print(f"\n=== Overall NurViD Accuracy ===")
        print(f"Overall accuracy: {overall_accuracy:.4f} ({total_correct}/{total_records})")
        overall_results["overall"] = {
            "accuracy": overall_accuracy,
            "correct": total_correct,
            "total": total_records
        }
    
    return overall_results


def get_action_list_for_dataset_extended(dataset_name):
    """Get action list for dataset, including newer datasets not in old script."""
    if dataset_name == "EgoSurgery":
        # EgoSurgery phases extracted from the data
        return ['closing', 'closure', 'design', 'dissection', 'dressing', 'hemostasis', 'incision', 'irrigation', 'preparation']
    else:
        # Use the old script for supported datasets
        return old_eval_next_action.get_action_list_for_dataset(dataset_name)

def evaluate_single_dataset(dataset_name, dataset_records):
    """Evaluate a single dataset (AVOS, CholecT50, CoPESD, EgoSurgery)."""
    actions = get_action_list_for_dataset_extended(dataset_name)
    CLASS_MAP = old_eval_next_action.create_class_map_for_dataset(actions)
    
    print(f"Using action list for {dataset_name}: {actions}")
    
    # Load SentenceTransformer model
    if SENTENCE_TRANSFORMERS_AVAILABLE:
        semantic_class_eval_model = SentenceTransformer('all-MiniLM-L6-v2')
        class_embeddings = semantic_class_eval_model.encode(actions, convert_to_tensor=True)
    else:
        semantic_class_eval_model = None
        class_embeddings = None
    
    # Evaluate
    next_action_correct = 0
    next_action_total = 0
    per_class_correct = defaultdict(int)
    per_class_total = defaultdict(int)
    
    for record in dataset_records:
        pred_text = old_eval_next_action.normalize_action_text(record['answer'], dataset_name)
        gnd_text = old_eval_next_action.normalize_action_text(record['gnd'], dataset_name)
        
        # Skip if ground truth not in CLASS_MAP
        if gnd_text not in CLASS_MAP:
            print(f"Warning: Ground truth '{gnd_text}' not found in {dataset_name} action list")
            continue
        
        # Determine prediction class
        if pred_text in CLASS_MAP:
            pred_idx = CLASS_MAP[pred_text]
        else:
            # Use semantic similarity as fallback
            if SENTENCE_TRANSFORMERS_AVAILABLE and semantic_class_eval_model is not None:
                pred_emb = semantic_class_eval_model.encode(pred_text, convert_to_tensor=True)
                sim_scores = util.cos_sim(pred_emb, class_embeddings)[0]
                pred_idx = sim_scores.argmax().item()
                print(f"Using semantic similarity for prediction: '{pred_text}' -> '{actions[pred_idx]}'")
            else:
                # No semantic similarity available, mark as incorrect
                pred_idx = -1
        
        gnd_idx = CLASS_MAP[gnd_text]
        per_class_total[gnd_text] += 1
        
        if pred_idx == gnd_idx:
            next_action_correct += 1
            per_class_correct[gnd_text] += 1
        next_action_total += 1
    
    # Final accuracy
    results = {}
    if next_action_total > 0:
        accuracy = next_action_correct / next_action_total
        balanced_acc = calculate_balanced_accuracy(per_class_correct, per_class_total, actions)
        
        print(f"Overall accuracy: {accuracy:.4f} ({next_action_correct}/{next_action_total})")
        print(f"Balanced accuracy: {balanced_acc:.4f}")
        
        results["overall"] = {
            "accuracy": accuracy,
            "balanced_accuracy": balanced_acc,
            "correct": next_action_correct,
            "total": next_action_total
        }
        
        print(f"\nPer-class accuracy:")
        per_class_results = {}
        for action in actions:
            total_cls = per_class_total[action]
            correct_cls = per_class_correct[action]
            if total_cls > 0:
                acc = correct_cls / total_cls
                print(f"{action:40s}: {acc:.4f} ({correct_cls}/{total_cls})")
                per_class_results[action] = {"accuracy": acc, "correct": correct_cls, "total": total_cls}
            else:
                print(f"{action:40s}: N/A (0 samples)")
                per_class_results[action] = {"accuracy": 0.0, "correct": 0, "total": 0}
        
        results["per_class"] = per_class_results
    else:
        print("No valid records found!")
        results["overall"] = {"accuracy": 0.0, "balanced_accuracy": 0.0, "correct": 0, "total": 0}
    
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
        print(f"  {dataset}: {len(records)} next action records")
    
    # Evaluate each dataset
    all_results = {}
    for dataset_name, records in dataset_records.items():
        if records:  # Only evaluate if we have records
            results = evaluate_dataset_next_action(dataset_name, records)
            all_results[dataset_name] = results
    
    # Print summary
    print(f"\n{'='*60}")
    print("NEXT ACTION PREDICTION EVALUATION SUMMARY")
    print(f"{'='*60}")
    
    for dataset_name, results in all_results.items():
        if results and "overall" in results:
            print(f"\n{dataset_name}:")
            overall = results["overall"]
            print(f"  Overall Accuracy: {overall['accuracy']:.4f} ({overall['correct']}/{overall['total']})")
            if "balanced_accuracy" in overall:
                print(f"  Balanced Accuracy: {overall['balanced_accuracy']:.4f}")
    
    return all_results


if __name__ == "__main__":
    main()
