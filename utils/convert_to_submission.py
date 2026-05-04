"""
Convert inference output (dict format) to leaderboard submission format (list format).

Submission ID format (must match the leaderboard's extract_predictions.py):
    {video_id}&&{start_frame}&&{end_frame}&&{fps}

Source: MedVidBench-Leaderboard/evaluation/extract_predictions.py:43

Usage:
    python3 utils/convert_to_submission.py results/results.json submission.json
"""

import json
import sys


def convert(input_path: str, output_path: str) -> None:
    with open(input_path) as f:
        data = json.load(f)

    records = data.values() if isinstance(data, dict) else data

    submissions = []
    for record in records:
        metadata = record.get("metadata", {}) or {}
        video_id = metadata.get("video_id", "")
        start_frame = metadata.get("input_video_start_frame", "") or metadata.get("start_frame", "")
        end_frame = metadata.get("input_video_end_frame", "") or metadata.get("end_frame", "")
        fps = metadata.get("fps", "")

        submissions.append({
            "id": f"{video_id}&&{start_frame}&&{end_frame}&&{fps}",
            "qa_type": record.get("qa_type", ""),
            "prediction": record.get("answer", record.get("prediction", "")),
        })

    with open(output_path, "w") as f:
        json.dump(submissions, f, indent=2)

    unique_ids = len({s["id"] for s in submissions})
    print(f"Converted {len(submissions)} samples ({unique_ids} unique video clips) -> {output_path}")
    print("Note: duplicate ids are expected - same video clip can have multiple QA pairs")
    print("Upload at: https://huggingface.co/spaces/UII-AI/MedVidBench-Leaderboard")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 utils/convert_to_submission.py <input.json> <output.json>")
        sys.exit(1)
    convert(sys.argv[1], sys.argv[2])
