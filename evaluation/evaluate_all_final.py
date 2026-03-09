"""Main Evaluation Script for All Tasks and Multiple Datasets."""

import json
import sys
import argparse
from collections import defaultdict

# Import task-specific evaluation modules using importlib to avoid path conflicts
import importlib.util
import os

_EVAL_DIR = os.path.dirname(os.path.abspath(__file__))

def load_eval_module(module_name):
    """Load evaluation module from the same directory as this script."""
    module_path = os.path.join(_EVAL_DIR, f"{module_name}.py")
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def analyze_output_file(output_file):
    """Analyze the output file to determine what tasks and datasets are present."""
    print(f"Analyzing output file: {output_file}")
    
    with open(output_file, "r") as f:
        data = json.load(f)
    
    # Count different QA types
    qa_type_counts = defaultdict(int)
    dataset_counts = defaultdict(int)
    
    # Handle both dict and list formats
    if isinstance(data, dict):
        records = data.values()
    elif isinstance(data, list):
        records = data
    else:
        print(f"Unexpected data format: {type(data)}")
        return {}, {}
    
    for record in records:
        qa_type = record.get("qa_type", "unknown")
        qa_type_counts[qa_type] += 1
        
        # Get dataset from data_source field if available
        dataset = record.get("data_source", "Unknown")
        
        # Fallback to detection methods if data_source is not available
        if dataset == "Unknown" or not dataset:
            video_id = record.get("metadata", {}).get("video_id", "")
            dataset = detect_dataset_from_video_id(video_id)
            if dataset == "Unknown":
                dataset = detect_dataset_from_question(record.get("question", ""))
        
        dataset_counts[dataset] += 1
    
    print(f"\nFound QA types:")
    for qa_type, count in qa_type_counts.items():
        print(f"  {qa_type}: {count} records")
    
    print(f"\nFound datasets:")
    for dataset, count in dataset_counts.items():
        print(f"  {dataset}: {count} records")
    
    return qa_type_counts, dataset_counts


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





def print_evaluation_results_csv_with_real_results(output_file, tasks, all_task_results):
    """Print evaluation results in CSV format with real captured results."""
    print(f"\n{'='*80}")
    print(f"EVALUATION RESULTS SUMMARY (NEW CSV FORMAT) - WITH REAL RESULTS")
    print(f"{'='*80}")
    
    # Convert the task results to the format expected by the internal function
    converted_results = {}
    
    # Load the data to get FPS information
    with open(output_file, "r") as f:
        data = json.load(f)
    
    # Group records by dataset, fps, and task to match structure
    dataset_fps_task_stats = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {
        'count': 0, 'videos': set()
    })))
    
    # Handle both dict and list formats
    if isinstance(data, dict):
        records = data.values()
    elif isinstance(data, list):
        records = data
    else:
        print(f"Unexpected data format in print_evaluation_results_csv_with_real_results: {type(data)}")
        return
    
    for record in records:
        qa_type = record.get("qa_type", "unknown")
        dataset = record.get("data_source", "Unknown")
        
        # Fallback to detection methods if data_source is not available
        if dataset == "Unknown" or not dataset:
            video_id = record.get("metadata", {}).get("video_id", "")
            dataset = detect_dataset_from_video_id(video_id)
            if dataset == "Unknown":
                dataset = detect_dataset_from_question(record.get("question", ""))
        
        fps = record.get("metadata", {}).get("fps", "unknown")
        video_id = record.get("metadata", {}).get("video_id", "unknown")
        
        # Map qa_type to task name for consistency
        task_name = "unknown"
        if any("dense_captioning" in qa_type or qa_type == "dc" for _ in [qa_type]):
            task_name = "dvc"
        elif qa_type == "tal":
            task_name = "tal"
        elif qa_type == "next_action":
            task_name = "next_action"
        elif qa_type == "stg":
            task_name = "stg"
        elif "region_caption" in qa_type:
            task_name = "rc"
        elif "video_summary" in qa_type:
            task_name = "vs"
        elif qa_type == "skill_assessment":
            task_name = "skill_assessment"
        elif qa_type == "cvs_assessment":
            task_name = "cvs_assessment"
        
        # Only include tasks that were evaluated
        if task_name in tasks or task_name == "unknown":
            dataset_fps_task_stats[dataset][fps][task_name]['count'] += 1
            dataset_fps_task_stats[dataset][fps][task_name]['videos'].add(video_id)
    
    # Convert real evaluation results to expected format
    for task_name, task_results in all_task_results.items():
        for dataset_name, dataset_results in task_results.items():
            # For each FPS in this dataset
            for fps in dataset_fps_task_stats[dataset_name].keys():
                if task_name in dataset_fps_task_stats[dataset_name][fps]:
                    eval_key = f"{dataset_name}_{task_name}_{fps}"
                    
                    # Extract metrics based on task type
                    if task_name == "dvc":
                        # DVC format: extract CIDER, METEOR, Precision_Mean, Recall_Mean, F1_Score
                        metrics = []
                        if isinstance(dataset_results, dict):
                            metrics.append(dataset_results.get('CIDER', 0.0))
                            metrics.append(dataset_results.get('METEOR', 0.0))
                            metrics.append(dataset_results.get('Precision_Mean', 0.0))
                            metrics.append(dataset_results.get('Recall_Mean', 0.0))
                            metrics.append(dataset_results.get('F1_Score', 0.0))
                            metrics.append(dataset_results.get('SODA_c_1', 0.0))
                        converted_results[eval_key] = {'metrics': metrics}
                        
                    elif task_name == "tal":
                        # TAL format: extract precision and recall at different IoU thresholds
                        metrics = []
                        if isinstance(dataset_results, dict):
                            # Look for IoU thresholds
                            metrics.append(dataset_results.get('0.3', {}).get('Precision', 0.0))
                            metrics.append(dataset_results.get('0.3', {}).get('Recall', 0.0))
                            metrics.append(dataset_results.get('0.5', {}).get('Precision', 0.0))
                            metrics.append(dataset_results.get('0.5', {}).get('Recall', 0.0))
                            metrics.append(dataset_results.get('mAP@0.5', 0.0))
                        converted_results[eval_key] = {'metrics': metrics}
                        
                    elif task_name == "next_action":
                        # Next Action format: extract overall accuracy
                        metrics = []
                        if isinstance(dataset_results, dict) and 'overall' in dataset_results:
                            overall = dataset_results['overall']
                            metrics.append(overall.get('accuracy', 0.0))
                            metrics.append(0.0)  # Per_class_avg placeholder
                            metrics.append(0.0)  # Weighted_F1 placeholder
                        converted_results[eval_key] = {'metrics': metrics}
                        
                    elif task_name == "stg":
                        # STG format: extract IoU metrics
                        metrics = []
                        if isinstance(dataset_results, dict):
                            # Use overall metrics if available
                            if 'overall' in dataset_results:
                                overall = dataset_results['overall']
                                mean_iou = overall.get('mean_iou', 0.0)
                                metrics = [mean_iou, mean_iou, mean_iou, mean_iou]  # IoU@0.3, 0.5, 0.7, mIoU
                            else:
                                # Use FPS-specific metrics
                                fps_result = dataset_results.get(str(fps), {})
                                mean_iou = fps_result.get('mean_iou', 0.0)
                                metrics = [mean_iou, mean_iou, mean_iou, mean_iou]
                        converted_results[eval_key] = {'metrics': metrics}
    
    # Use the existing function but pass the converted real evaluation results
    print_evaluation_results_csv_internal(output_file, tasks, converted_results)


def print_evaluation_results_csv(output_file, tasks):
    """Print evaluation results in new CSV format: Dataset → Task → Metrics."""
    print(f"\n{'='*80}")
    print(f"EVALUATION RESULTS SUMMARY (NEW CSV FORMAT)")
    print(f"{'='*80}")
    
    # Call internal function with empty evaluation results (for analyze-only mode)
    print_evaluation_results_csv_internal(output_file, tasks, {})


def print_evaluation_results_csv_internal(output_file, tasks, evaluation_results):
    """Internal function to print CSV results with optional real evaluation results."""
    # Load the data to analyze structure
    with open(output_file, "r") as f:
        data = json.load(f)
    
    # Define metrics for each task type (these will be populated from actual evaluation results)
    task_metrics = {
        'dvc': ['CIDER', 'METEOR', 'Precision@0.5', 'Recall@0.5', 'F1_Score'],
        'tal': ['Precision@0.3', 'Recall@0.3', 'Precision@0.5', 'Recall@0.5', 'mAP@0.5'],
        'next_action': ['Accuracy', 'Per_class_avg', 'Weighted_F1'],
        'stg': ['IoU@0.3', 'IoU@0.5', 'IoU@0.7', 'mIoU'],
        'rc': ['BLEU4', 'METEOR', 'CIDEr', 'ROUGE_L'],
        'vs': ['BLEU4', 'METEOR', 'CIDEr', 'ROUGE_L'],
        'skill_assessment': ['Accuracy', 'Macro_F1', 'Weighted_F1'],
        'cvs_assessment': ['Accuracy', 'Precision', 'Recall', 'F1_Score']
    }
    
    # Group records by dataset, fps, and task
    dataset_fps_task_stats = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {
        'count': 0, 'videos': set()
    })))
    
    # Handle both dict and list formats
    if isinstance(data, dict):
        records = data.values()
    elif isinstance(data, list):
        records = data
    else:
        print(f"Unexpected data format in print_evaluation_results_csv_internal: {type(data)}")
        return
    
    for record in records:
        qa_type = record.get("qa_type", "unknown")
        dataset = record.get("data_source", "Unknown")
        
        # Fallback to detection methods if data_source is not available
        if dataset == "Unknown" or not dataset:
            video_id = record.get("metadata", {}).get("video_id", "")
            dataset = detect_dataset_from_video_id(video_id)
            if dataset == "Unknown":
                dataset = detect_dataset_from_question(record.get("question", ""))
        
        fps = record.get("metadata", {}).get("fps", "unknown")
        video_id = record.get("metadata", {}).get("video_id", "unknown")
        
        # Map qa_type to task name for consistency
        task_name = "unknown"
        if any("dense_captioning" in qa_type or qa_type == "dc" for _ in [qa_type]):
            task_name = "dvc"
        elif qa_type == "tal":
            task_name = "tal"
        elif qa_type == "next_action":
            task_name = "next_action"
        elif qa_type == "stg":
            task_name = "stg"
        elif "region_caption" in qa_type:
            task_name = "rc"
        elif "video_summary" in qa_type:
            task_name = "vs"
        elif qa_type == "skill_assessment":
            task_name = "skill_assessment"
        elif qa_type == "cvs_assessment":
            task_name = "cvs_assessment"
        
        # Only include tasks that were evaluated
        if task_name in tasks or task_name == "unknown":
            dataset_fps_task_stats[dataset][fps][task_name]['count'] += 1
            dataset_fps_task_stats[dataset][fps][task_name]['videos'].add(video_id)
    
    # Get all unique tasks that have data
    available_tasks = set()
    for dataset_stats in dataset_fps_task_stats.values():
        for fps_stats in dataset_stats.values():
            available_tasks.update(fps_stats.keys())
    
    # Print results for each dataset
    for dataset_name in sorted(dataset_fps_task_stats.keys()):
        print(f"\n{dataset_name}")
        
        # For each task in this dataset
        dataset_tasks = set()
        for fps_stats in dataset_fps_task_stats[dataset_name].values():
            dataset_tasks.update(fps_stats.keys())
        
        for task_name in sorted(dataset_tasks):
            print(f"{task_name}")
            
            # Print headers for this task
            metrics = task_metrics.get(task_name, ['Count', 'Videos'])
            header = "fps, qa_instances, " + ", ".join(metrics)
            print(header)
            
            # Store metrics for overall average calculation
            task_overall_metrics = []
            task_overall_count = 0
            
            # Print data rows for each FPS
            for fps in sorted(dataset_fps_task_stats[dataset_name].keys()):
                fps_stats = dataset_fps_task_stats[dataset_name][fps]
                
                if task_name in fps_stats:
                    task_stats = fps_stats[task_name]
                    count = task_stats['count']
                    video_count = len(task_stats['videos'])
                    
                    # Get real evaluation results if available
                    eval_key = f"{dataset_name}_{task_name}_{fps}"
                    if eval_key in evaluation_results:
                        values = evaluation_results[eval_key]['metrics']
                        task_overall_metrics.append(values)
                        task_overall_count += count
                        
                        # Format values as strings
                        value_strs = [f"{v:.3f}" if isinstance(v, float) else str(v) for v in values]
                        row = f"{fps}, {count}, " + ", ".join(value_strs)
                        print(row)
                    else:
                        print(f"No real results for {eval_key}, missing!!!")
            
            # Add overall average line if we have metrics
            if task_overall_metrics and task_overall_count > 0:
                # Calculate weighted average across all fps
                num_metrics = len(task_overall_metrics[0])
                overall_avg = [0.0] * num_metrics
                for metrics in task_overall_metrics:
                    for i, val in enumerate(metrics):
                        if isinstance(val, (int, float)):
                            overall_avg[i] += val
                
                # Average the metrics
                for i in range(num_metrics):
                    overall_avg[i] /= len(task_overall_metrics)
                
                avg_strs = [f"{v:.3f}" for v in overall_avg]
                avg_row = f"Overall, {task_overall_count}, " + ", ".join(avg_strs)
                print(avg_row)
    
    # Print combined summary
    print(f"\nCombined Summary")
    
    for task_name in sorted(available_tasks):
        print(f"{task_name}")
        
        # Aggregate across all datasets for this task
        task_fps_stats = defaultdict(lambda: {'count': 0, 'videos': set()})
        
        for dataset_stats in dataset_fps_task_stats.values():
            for fps, fps_stats in dataset_stats.items():
                if task_name in fps_stats:
                    task_fps_stats[fps]['count'] += fps_stats[task_name]['count']
                    task_fps_stats[fps]['videos'].update(fps_stats[task_name]['videos'])
        
        # Print headers
        metrics = task_metrics.get(task_name, ['Count', 'Videos'])
        header = "fps, qa_instances, " + ", ".join(metrics)
        print(header)
        
        # Store metrics for overall average calculation
        combined_task_metrics = []
        combined_task_count = 0
        
        # Print data rows
        for fps in sorted(task_fps_stats.keys()):
            fps_data = task_fps_stats[fps]
            count = fps_data['count']
            video_count = len(fps_data['videos'])
            

        
        # Add overall average line for combined summary
        if combined_task_metrics and combined_task_count > 0:
            # Calculate average across all fps for this task
            num_metrics = len(combined_task_metrics[0])
            combined_avg = [0.0] * num_metrics
            for metrics in combined_task_metrics:
                for i, val in enumerate(metrics):
                    if isinstance(val, (int, float)):
                        combined_avg[i] += val
            
            # Average the metrics
            for i in range(num_metrics):
                combined_avg[i] /= len(combined_task_metrics)
            
            avg_strs = [f"{v:.3f}" for v in combined_avg]
            avg_row = f"Overall, {combined_task_count}, " + ", ".join(avg_strs)
            print(avg_row)


def print_overall_evaluation_results(output_file, tasks, all_task_results, skip_caption=False):
    """Print evaluation results in overall mode (dataset-agnostic).

    For each task, computes metrics by processing individual samples across
    all datasets together, rather than averaging per-dataset metrics.
    """
    print(f"\n{'='*80}")
    print(f"EVALUATION RESULTS - OVERALL (Dataset-Agnostic)")
    print(f"{'='*80}")

    # Load the data to re-process at individual level
    with open(output_file, "r") as f:
        data = json.load(f)

    # Handle both dict and list formats
    if isinstance(data, dict):
        records = list(data.values())
    elif isinstance(data, list):
        records = data
    else:
        print(f"Unexpected data format: {type(data)}")
        return

    # For each task, collect all records across datasets and re-evaluate
    for task_name in sorted(tasks):
        print(f"\n{'='*80}")
        print(f"{task_name.upper()} - Overall Evaluation (All Datasets Combined)")
        print(f"{'='*80}")

        # Filter records for this task
        task_records = []
        for record in records:
            qa_type = record.get("qa_type", "unknown")

            # Map qa_type to task name
            mapped_task = None
            if any("dense_captioning" in qa_type or qa_type == "dc" for _ in [qa_type]):
                mapped_task = "dvc"
            elif qa_type == "tal":
                mapped_task = "tal"
            elif qa_type == "next_action":
                mapped_task = "next_action"
            elif qa_type == "stg":
                mapped_task = "stg"
            elif "region_caption" in qa_type:
                mapped_task = "rc"
            elif "video_summary" in qa_type:
                mapped_task = "vs"
            elif qa_type == "skill_assessment":
                mapped_task = "skill_assessment"
            elif qa_type == "cvs_assessment":
                mapped_task = "cvs_assessment"

            if mapped_task == task_name:
                task_records.append(record)

        if not task_records:
            print(f"No records found for {task_name}")
            continue

        print(f"Total samples: {len(task_records)}")

        # Re-run evaluation on all records together
        # Import and call the appropriate evaluation function
        try:
            if task_name == "tal":
                # Import the eval module
                module = load_eval_module("eval_tal")
                # Create a temporary dict with sequential keys
                temp_data = {str(i): record for i, record in enumerate(task_records)}
                # Get grouped records
                dataset_records_dict = module.group_records_by_dataset(temp_data)
                # Combine all records across datasets
                all_records = []
                for ds_records in dataset_records_dict.values():
                    all_records.extend(ds_records)
                # Evaluate as single dataset
                results = module.evaluate_dataset_tal("Overall", all_records)
                # Print results
                for iou_key, metrics in results.items():
                    if isinstance(metrics, dict):
                        print(f"\n{iou_key}:")
                        for metric_name, value in metrics.items():
                            print(f"  {metric_name}: {value:.4f}")
                    else:
                        print(f"{iou_key}: {metrics:.4f}")

            elif task_name == "stg":
                module = load_eval_module("eval_stg")
                temp_data = {str(i): record for i, record in enumerate(task_records)}
                dataset_records_dict = module.group_records_by_dataset(temp_data)
                all_records = []
                for ds_records in dataset_records_dict.values():
                    all_records.extend(ds_records)
                results = module.evaluate_dataset_stg("Overall", all_records)
                for key, value in results.items():
                    if isinstance(value, dict):
                        print(f"\n{key}:")
                        for metric_name, metric_value in value.items():
                            print(f"  {metric_name}: {metric_value:.4f}")
                    else:
                        print(f"{key}: {value:.4f}")

            elif task_name in ["rc", "vs"]:
                module = load_eval_module("eval_rc_vs")
                temp_data = {str(i): record for i, record in enumerate(task_records)}
                # Get the correct qa_types for filtering
                qa_types = ["region_caption"] if task_name == "rc" else ["video_summary"]
                dataset_records_dict = module.group_records_by_dataset(temp_data, qa_types)
                # Get the correct task key
                task_key = "region_caption" if task_name == "rc" else "video_summary"
                all_records = []
                for ds_task_records in dataset_records_dict.values():
                    if task_key in ds_task_records:
                        all_records.extend(ds_task_records[task_key])
                if all_records:
                    results = module.evaluate_caption_task(task_key.replace("_", " ").title(), all_records)
                    for metric_name, value in results.items():
                        print(f"{metric_name}: {value:.4f}")
                else:
                    print(f"No records found for {task_key}")

            elif task_name == "next_action":
                module = load_eval_module("eval_next_action")
                temp_data = {str(i): record for i, record in enumerate(task_records)}
                dataset_records_dict = module.group_records_by_dataset(temp_data)

                # For next_action, we need to evaluate per dataset (different action lists)
                # then aggregate the results - but suppress per-dataset output
                all_accuracies = []
                total_correct = 0
                total_samples = 0

                # Suppress output during per-dataset evaluation
                import io
                import contextlib

                for dataset_name, ds_records in dataset_records_dict.items():
                    if ds_records:
                        # Silently evaluate each dataset
                        with contextlib.redirect_stdout(io.StringIO()):
                            ds_results = module.evaluate_dataset_next_action(dataset_name, ds_records)
                        if "overall" in ds_results:
                            accuracy = ds_results["overall"].get("accuracy", 0.0)
                            all_accuracies.append(accuracy)
                            # Track weighted metrics
                            total_correct += int(accuracy * len(ds_records))
                            total_samples += len(ds_records)

                # Print only final aggregate metrics
                if all_accuracies:
                    macro_avg = sum(all_accuracies) / len(all_accuracies)
                    weighted_avg = total_correct / total_samples if total_samples > 0 else 0.0
                    print(f"\nMacro Average Accuracy (across {len(all_accuracies)} datasets): {macro_avg:.4f}")
                    print(f"Weighted Average Accuracy (across {total_samples} samples): {weighted_avg:.4f}")

            elif task_name == "dvc":
                module = load_eval_module("eval_dvc")
                temp_data = {str(i): record for i, record in enumerate(task_records)}
                dataset_records_dict = module.group_records_by_dataset(temp_data)
                # Combine all records across datasets
                all_records = []
                for ds_records in dataset_records_dict.values():
                    all_records.extend(ds_records)
                # Evaluate as single dataset
                results = module.evaluate_dataset_dvc("Overall", all_records, skip_caption=skip_caption)
                # Print results
                print(f"\nDense Video Captioning Metrics:")
                for metric_name, value in results.items():
                    if isinstance(value, (int, float)):
                        print(f"  {metric_name}: {value:.4f}")

            elif task_name == "cvs_assessment":
                module = load_eval_module("eval_cvs_assessment")
                temp_data = {str(i): record for i, record in enumerate(task_records)}
                dataset_records_dict = module.group_records_by_dataset(temp_data)
                # Combine all records across datasets
                all_records = []
                for ds_records in dataset_records_dict.values():
                    all_records.extend(ds_records)
                # Evaluate combined
                results = module.evaluate_cvs_assessment(all_records)
                # Print results
                print(f"\nCVS Assessment Metrics:")
                if "overall" in results:
                    for metric_name, value in results["overall"].items():
                        if isinstance(value, (int, float)):
                            print(f"  {metric_name}: {value:.4f}")
                else:
                    for metric_name, value in results.items():
                        if isinstance(value, (int, float)):
                            print(f"  {metric_name}: {value:.4f}")

            elif task_name == "skill_assessment":
                module = load_eval_module("eval_skill_assessment")
                temp_data = {str(i): record for i, record in enumerate(task_records)}
                dataset_records_dict = module.group_records_by_dataset(temp_data)
                # Combine all records across datasets
                all_records = []
                for ds_records in dataset_records_dict.values():
                    all_records.extend(ds_records)
                # Evaluate combined
                results = module.evaluate_skill_assessment(all_records)
                # Print results
                print(f"\nSkill Assessment Metrics:")
                if "overall" in results:
                    for metric_name, value in results["overall"].items():
                        if isinstance(value, (int, float)):
                            print(f"  {metric_name}: {value:.4f}")
                else:
                    for metric_name, value in results.items():
                        if isinstance(value, (int, float)):
                            print(f"  {metric_name}: {value:.4f}")

                # Print per-aspect MAE (for CSV parsing)
                if "per_aspect" in results and results["per_aspect"]:
                    print(f"\nPer-Aspect MAE:")
                    mae_values = []
                    for aspect_name in sorted(results["per_aspect"].keys()):
                        aspect_metrics = results["per_aspect"][aspect_name]
                        mae = aspect_metrics.get("mae", 0.0)
                        mae_values.append(mae)
                        # Use shortened aspect names for CSV compatibility
                        short_name = aspect_name.replace(" ", "_").replace("/", "_")
                        print(f"  MAE_{short_name}: {mae:.4f}")
                    # Print average MAE
                    if mae_values:
                        avg_mae = sum(mae_values) / len(mae_values)
                        print(f"  MAE_Average: {avg_mae:.4f}")

            else:
                print(f"Overall evaluation not implemented for {task_name} yet")

        except Exception as e:
            print(f"Error running overall evaluation for {task_name}: {e}")
            import traceback
            traceback.print_exc()


def _parse_metrics_from_output(output):
    """Parse leaderboard metrics from evaluation stdout.

    Mirrors app.py's parse_evaluation_output() logic.
    Returns dict with keys: tag_miou_03, tag_miou_05, stg_miou, nap_acc,
                             sa_acc, cvs_acc, dvc_f1, dvc_llm, vs_llm, rc_llm
    """
    metrics = {}
    lines = output.split('\n')
    current_task = None
    current_iou_section = None

    for line in lines:
        line = line.strip()

        # Detect task sections
        if "TAL" in line and "Overall" in line:
            current_task = "tal"
        elif "STG" in line and "Overall" in line:
            current_task = "stg"
        elif ("NEXT_ACTION" in line and "Overall" in line) or "Next Action" in line or ("next_action" in line.lower() and "overall" in line.lower()):
            current_task = "next_action"
        elif ("DVC" in line and "Overall" in line) or "Dense Video Captioning" in line:
            current_task = "dvc"
        elif ("RC" in line and "Overall" in line) or "Region Caption" in line:
            current_task = "rc"
        elif ("VS" in line and "Overall" in line) or "Video Summary" in line:
            current_task = "vs"
        elif ("SKILL" in line and "Overall" in line) or "Skill Assessment" in line:
            current_task = "skill_assessment"
        elif ("CVS" in line and "Overall" in line) or "CVS Assessment" in line:
            current_task = "cvs_assessment"

        if current_task == "tal":
            if "IoU_0.3:" in line:
                current_iou_section = "0.3"
            elif "IoU_0.5:" in line:
                current_iou_section = "0.5"

        if not current_task:
            continue

        try:
            if current_task == "tal":
                if "meanIoU@0.3" in line or "mIoU@0.3" in line:
                    metrics["tag_miou_03"] = float(line.split(":")[-1].strip())
                elif "meanIoU@0.5" in line or "mIoU@0.5" in line:
                    metrics["tag_miou_05"] = float(line.split(":")[-1].strip())
                elif current_iou_section and "meanIoU:" in line and "meanIoU@" not in line:
                    val = float(line.split(":")[-1].strip())
                    if current_iou_section == "0.3":
                        metrics["tag_miou_03"] = val
                    elif current_iou_section == "0.5":
                        metrics["tag_miou_05"] = val

            elif current_task == "stg" and ("mean_iou" in line.lower() or "miou" in line.lower() or "mean iou" in line.lower()):
                metrics["stg_miou"] = float(line.split(":")[-1].strip())

            elif current_task == "next_action" and "accuracy" in line.lower():
                metrics["nap_acc"] = float(line.split(":")[-1].strip())

            elif current_task == "dvc":
                if "caption_score" in line.lower() or "caption score" in line.lower():
                    metrics["dvc_llm"] = float(line.split(":")[-1].strip())
                elif "temporal_f1" in line.lower() or "temporal f1" in line.lower() or "f1_score" in line.lower():
                    metrics["dvc_f1"] = float(line.split(":")[-1].strip())

            elif current_task == "vs" and ("score" in line.lower() or "average" in line.lower()):
                metrics["vs_llm"] = float(line.split(":")[-1].strip())

            elif current_task == "rc" and ("score" in line.lower() or "average" in line.lower()):
                metrics["rc_llm"] = float(line.split(":")[-1].strip())

            elif current_task == "skill_assessment" and "accuracy:" in line.lower() and "aspect" not in line.lower() and "balanced" not in line.lower():
                metrics["sa_acc"] = float(line.split(":")[-1].split("(")[0].strip())

            elif current_task == "cvs_assessment" and "accuracy:" in line and "component_balanced" not in line:
                metrics["cvs_acc"] = float(line.split(":")[-1].strip())
        except (ValueError, IndexError):
            pass

    return metrics


def _print_leaderboard_summary(captured_output, skip_caption=False):
    """Print a clean leaderboard metrics summary parsed from evaluation stdout."""
    metrics = _parse_metrics_from_output(captured_output)

    print(f"\n{'='*80}", flush=True)
    print("LEADERBOARD METRICS SUMMARY", flush=True)
    print(f"{'='*80}", flush=True)

    METRIC_LABELS = [
        ("cvs_acc",     "CVS Assessment - Overall Evaluation",   "cvs_assessment",   "  accuracy: {v:.4f}"),
        ("nap_acc",     "Next Action - Overall Evaluation",  "next_action",     "  accuracy: {v:.4f}"),
        ("sa_acc",      "Skill Assessment - Overall Evaluation", "skill_assessment", "  Overall Accuracy: {v:.4f}"),
        ("stg_miou",    "STG - Overall Evaluation",          "stg",             "  mean_iou: {v:.4f}"),
        ("tag_miou_03", "TAL - Overall Evaluation",          "tal",             "  mIoU@0.3: {v:.4f}"),
        ("tag_miou_05", None,                                 None,              "  mIoU@0.5: {v:.4f}"),
        ("dvc_f1",      "Dense Video Captioning Metrics",    "dvc",             "  temporal_f1: {v:.4f}"),
    ]
    if not skip_caption:
        METRIC_LABELS += [
            ("dvc_llm",  None,                               None,   "  caption_score: {v:.4f}"),
            ("vs_llm",  "VS - Overall Evaluation",           "vs",   "  score: {v:.4f}"),
            ("rc_llm",  "RC - Overall Evaluation",           "rc",   "  score: {v:.4f}"),
        ]

    last_task = None
    for metric_key, header, task_tag, fmt in METRIC_LABELS:
        if metric_key not in metrics:
            continue
        v = metrics[metric_key]
        if header and task_tag != last_task:
            print(f"\n{header}", flush=True)
            last_task = task_tag
        print(fmt.format(v=v), flush=True)

    print(f"\n{'='*80}", flush=True)
    print("END LEADERBOARD METRICS SUMMARY", flush=True)
    print(f"{'='*80}\n", flush=True)


def _run_task_eval(task, output_file, skip_caption=False):
    """Helper function to run a single task evaluation.

    Args:
        task: Task name (e.g., 'tal', 'stg')
        output_file: Path to results JSON
        skip_caption: If True, skip caption scoring in DVC (avoids PTBTokenizer)

    Returns:
        Dictionary of evaluation results
    """
    import sys

    if task == "dvc":
        module = load_eval_module("eval_dvc")
        sys.argv = ["eval_script", output_file]
        with open(output_file, "r") as f:
            import json as _json
            infer_output = _json.load(f)
        dataset_records = module.group_records_by_dataset(infer_output)
        task_results = {}
        for ds_name, records in dataset_records.items():
            if records:
                task_results[ds_name] = module.evaluate_dataset_dvc(ds_name, records, skip_caption=skip_caption)
    elif task == "tal":
        module = load_eval_module("eval_tal")
        task_results = module.main()
    elif task == "next_action":
        module = load_eval_module("eval_next_action")
        task_results = module.main()
    elif task == "stg":
        module = load_eval_module("eval_stg")
        task_results = module.main()
    elif task == "rc":
        module = load_eval_module("eval_rc_vs")
        # Pass parameter to indicate RC-only evaluation
        sys.argv = ["eval_script", output_file, "--task", "rc"]
        task_results = module.main()
    elif task == "vs":
        module = load_eval_module("eval_rc_vs")
        # Pass parameter to indicate VS-only evaluation
        sys.argv = ["eval_script", output_file, "--task", "vs"]
        task_results = module.main()
    elif task == "skill_assessment":
        module = load_eval_module("eval_skill_assessment")
        task_results = module.main()
    elif task == "cvs_assessment":
        module = load_eval_module("eval_cvs_assessment")
        task_results = module.main()
    elif task == "gemini_structured":
        module = load_eval_module("eval_gemini_structured")
        task_results = module.main()
    elif task == "gpt_structured":
        module = load_eval_module("eval_gpt_structured")
        task_results = module.main()
    else:
        print(f"Unknown task: {task}")
        task_results = {}

    return task_results


def run_evaluation(output_file, tasks=None, grouping="per-dataset", silent_eval=False, skip_caption=False):
    """Run evaluation for specified tasks and capture real results.

    Args:
        output_file: Path to inference results JSON
        tasks: List of tasks to evaluate (None = auto-detect)
        grouping: 'per-dataset' or 'overall' - how to group results
        silent_eval: If True, suppress intermediate per-dataset output
        skip_caption: If True, skip RC and VS tasks (avoids PTBTokenizer)
    """
    # Analyze the file first
    qa_type_counts, dataset_counts = analyze_output_file(output_file)

    # Determine which tasks to run
    if tasks is None:
        # Run all available tasks based on what's in the file
        available_tasks = []
        
        # Check for dense captioning (various naming patterns)
        if any("dense_captioning" in qa_type or qa_type == "dc" for qa_type in qa_type_counts):
            available_tasks.append("dvc")
        
        # Check for TAL
        if qa_type_counts.get("tal", 0) > 0:
            available_tasks.append("tal")
            
        # Check for next action
        if qa_type_counts.get("next_action", 0) > 0:
            available_tasks.append("next_action")
            
        # Check for STG
        if qa_type_counts.get("stg", 0) > 0:
            available_tasks.append("stg")
            
        # Check for region caption and video summary (various naming patterns)
        if any("region_caption" in qa_type for qa_type in qa_type_counts):
            available_tasks.append("rc")
        if any("video_summary" in qa_type for qa_type in qa_type_counts):
            available_tasks.append("vs")
            
        # Check for skill assessment
        if qa_type_counts.get("skill_assessment", 0) > 0:
            available_tasks.append("skill_assessment")
            
        # Check for CVS assessment
        if qa_type_counts.get("cvs_assessment", 0) > 0:
            available_tasks.append("cvs_assessment")
        tasks = available_tasks

    if skip_caption:
        tasks = [t for t in tasks if t not in ("rc", "vs")]
        print(f"[--skip-caption] Skipping RC and VS (PTBTokenizer not required)")

    print(f"\nRunning evaluation for tasks: {tasks}")
    
    # Dictionary to store all evaluation results
    all_task_results = {}
    
    # Save original sys.argv to restore later
    original_argv = sys.argv.copy()

    # Redirect stdout if silent mode (for overall grouping)
    import io
    import contextlib

    try:
        # Run each task evaluation and capture returned results
        for task in tasks:
            if not silent_eval:
                print(f"\n{'='*80}")
                print(f"RUNNING {task.upper()} EVALUATION")
                print(f"{'='*80}")

            # Set sys.argv for the task-specific main function
            sys.argv = ["eval_script", output_file]
            
            # Load the module dynamically and call main to get results
            try:
                # Optionally suppress output from eval modules
                if silent_eval:
                    # Redirect stdout/stderr to devnull
                    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                        task_results = _run_task_eval(task, output_file, skip_caption=skip_caption)
                else:
                    task_results = _run_task_eval(task, output_file, skip_caption=skip_caption)

                # Store the results for this task
                all_task_results[task] = task_results if task_results else {}
                
            except Exception as e:
                print(f"Error running {task} evaluation: {e}")
                all_task_results[task] = {}
                
    finally:
        # Restore original sys.argv
        sys.argv = original_argv

    # Print results based on grouping mode
    if grouping == "overall":
        # Tee stdout to capture output for leaderboard summary parsing
        captured = io.StringIO()
        original_stdout = sys.stdout

        class _TeeWriter:
            def write(self, s):
                original_stdout.write(s)
                captured.write(s)
            def flush(self):
                original_stdout.flush()

        sys.stdout = _TeeWriter()
        try:
            print_overall_evaluation_results(output_file, tasks, all_task_results, skip_caption=skip_caption)
        finally:
            sys.stdout = original_stdout

        _print_leaderboard_summary(captured.getvalue(), skip_caption=skip_caption)
    else:  # per-dataset
        print_evaluation_results_csv_with_real_results(output_file, tasks, all_task_results)


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description="Evaluate multiple tasks on video understanding results")
    parser.add_argument("output_file",
                       help="Path to the JSON output file containing inference results")
    parser.add_argument("--tasks", nargs="+",
                       choices=["dvc", "tal", "next_action", "stg", "rc", "vs", "skill_assessment", "cvs_assessment", "gemini_structured", "gpt_structured"],
                       help="Specific tasks to evaluate (default: all available tasks)")
    parser.add_argument("--grouping", choices=["per-dataset", "overall"], default="overall",
                       help="Grouping strategy: 'per-dataset' shows results per dataset, 'overall' aggregates all datasets (default: per-dataset)")
    parser.add_argument("--analyze-only", action="store_true",
                       help="Only analyze the file structure without running evaluations")
    parser.add_argument("--structured", choices=["gemini", "gpt"],
                       help="Evaluate structured outputs from Gemini or GPT models")
    parser.add_argument("--skip-caption", action="store_true",
                       help="Skip RC and VS caption evaluation (avoids PTBTokenizer dependency)")

    args = parser.parse_args()

    if args.analyze_only:
        qa_type_counts, dataset_counts = analyze_output_file(args.output_file)
        # Print CSV-style results summary for analyze-only mode
        # Determine available tasks based on what's in the file
        available_tasks = []
        if any("dense_captioning" in qa_type or qa_type == "dc" for qa_type in qa_type_counts):
            available_tasks.append("dvc")
        if qa_type_counts.get("tal", 0) > 0:
            available_tasks.append("tal")
        if qa_type_counts.get("next_action", 0) > 0:
            available_tasks.append("next_action")
        if qa_type_counts.get("stg", 0) > 0:
            available_tasks.append("stg")
        if any("region_caption" in qa_type for qa_type in qa_type_counts):
            available_tasks.append("rc")
        if any("video_summary" in qa_type for qa_type in qa_type_counts):
            available_tasks.append("vs")
        if qa_type_counts.get("skill_assessment", 0) > 0:
            available_tasks.append("skill_assessment")
        if qa_type_counts.get("cvs_assessment", 0) > 0:
            available_tasks.append("cvs_assessment")

        print_evaluation_results_csv(args.output_file, available_tasks)
    else:
        # Handle structured evaluation
        # Enable silent mode when using overall grouping
        silent_eval = (args.grouping == "overall")

        if args.structured:
            tasks = [f"{args.structured}_structured"]
            run_evaluation(args.output_file, tasks, grouping=args.grouping, silent_eval=silent_eval)
        else:
            run_evaluation(args.output_file, args.tasks, grouping=args.grouping, silent_eval=silent_eval,
                           skip_caption=args.skip_caption)


if __name__ == "__main__":
    main()
