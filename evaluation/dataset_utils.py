"""Common dataset detection utilities for all evaluation scripts."""

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


def get_dataset_name(record):
    """Get dataset name from a record, preferring data_source field."""
    # First try to get dataset from data_source field
    dataset = record.get("data_source", "Unknown")
    if dataset != "Unknown" and dataset:
        return dataset
    
    # Fallback to detection methods if data_source is not available
    dataset_from_video_id = detect_dataset_from_video_id(record["metadata"]["video_id"])
    dataset_from_question = detect_dataset_from_question(record.get("question", ""))
    
    # Prefer question detection over video ID detection when both are not "Unknown"
    if dataset_from_question != "Unknown":
        return dataset_from_question
    else:
        return dataset_from_video_id
