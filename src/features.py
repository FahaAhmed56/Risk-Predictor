"""Input schema and validation for the student-support demo."""
FEATURES = ["attendance_pct", "study_hours_week", "previous_score", "assignments_completed_pct", "sleep_hours"]
RANGES = {
    "attendance_pct": (0, 100),
    "study_hours_week": (0, 80),
    "previous_score": (0, 100),
    "assignments_completed_pct": (0, 100),
    "sleep_hours": (0, 24),
}

def validate_record(record):
    if set(record) != set(FEATURES):
        raise ValueError(f"Expected exactly these fields: {FEATURES}")
    cleaned = {}
    for field in FEATURES:
        value = record[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{field} must be a number")
        low, high = RANGES[field]
        if not (low <= value <= high):
            raise ValueError(f"{field} must be between {low} and {high}")
        cleaned[field] = float(value)
    return cleaned
