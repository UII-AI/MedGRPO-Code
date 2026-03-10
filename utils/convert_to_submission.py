"""
Convert inference output (dict format) to leaderboard submission format (list format).

Inference output:   dict keyed by index, uses "answer" field
Submission format:  list, uses "prediction" field + "id" field

Usage:
    python3 utils/convert_to_submission.py results/results.json submission.json
"""

import json
import sys


def convert(input_path: str, output_path: str):
    with open(input_path) as f:
        data = json.load(f)

    # Handle both dict (inference output) and list (already converted) formats
    records = list(data.values()) if isinstance(data, dict) else data

    submissions = []
    for record in records:
        submissions.append({
            "id": record.get("metadata", {}).get("video_id", ""),
            "qa_type": record["qa_type"],
            "prediction": record.get("answer", record.get("prediction", "")),
        })

    with open(output_path, "w") as f:
        json.dump(submissions, f, indent=2)

    print(f"✓ Converted {len(submissions)} samples → {output_path}")
    print(f"  Ready to submit to: https://huggingface.co/spaces/UIIAmerica/MedVidBench-Leaderboard")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 utils/convert_to_submission.py <input.json> <output.json>")
        sys.exit(1)
    convert(sys.argv[1], sys.argv[2])
